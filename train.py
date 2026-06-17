import os
import argparse
import json
import math
import torch
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter

from dataset import get_cc3m_dataloader
from model import CoLLMStage1, resolve_attn_implementation, resolve_llm_quantization
from loss import collm_loss
from slerp import slerp
from text_synthesis import get_modification_texts


def get_nearest_neighbors(h_prime_detached):
    sim_matrix = h_prime_detached @ h_prime_detached.T
    sim_matrix.fill_diagonal_(-float("inf"))
    nn_indices = sim_matrix.argmax(dim=1)
    return h_prime_detached[nn_indices], nn_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_pattern",
        type=str,
        default="/home/rahul/shyam/aditya/training/cc3m_downloaded_80k_224/{00000..00011}.tar",
    )
    parser.add_argument("--llm_model", type=str, default="Salesforce/SFR-Embedding-2_R")
    parser.add_argument("--llm_dim", type=int, default=4096)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Contrastive batch size (logits are [B x B]). NN/Slerp also use this full batch.",
    )
    parser.add_argument(
        "--llm_micro_batch",
        type=int,
        default=8,
        help="Micro-batch for the 3 LLM query forwards; keeps peak VRAM low while B=32 for loss.",
    )
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--logit_scale_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument(
        "--llm_precision",
        type=str,
        default="auto",
        choices=["auto", "4bit", "bf16"],
        help="auto: 4-bit QLoRA when bitsandbytes is installed, else bf16.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["auto", "sdpa", "eager"],
        help="Attention backend for the LLM. auto resolves to sdpa.",
    )
    args = parser.parse_args()

    if args.batch_size % args.llm_micro_batch != 0:
        raise ValueError(
            f"batch_size ({args.batch_size}) must be divisible by "
            f"llm_micro_batch ({args.llm_micro_batch})"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn_impl = resolve_attn_implementation(args.attn_implementation)
    use_4bit = resolve_llm_quantization(args.llm_precision)
    print(f"Using device: {device}")
    print(f"Contrastive batch size: {args.batch_size}")
    print(f"LLM micro-batch size:   {args.llm_micro_batch}")
    print(f"LLM precision:          {'4-bit QLoRA' if use_4bit else 'bf16 (full weights)'}")
    print(f"Attention backend:      {attn_impl}")
    print(f"Learning rate:          {args.lr}")
    print(f"Logit scale LR:         {args.logit_scale_lr}")
    print(f"Max steps:              {args.max_steps}")
    print(f"Warmup ratio:           {args.warmup_ratio}")

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    step_logs = []

    dataloader = get_cc3m_dataloader(args.data_pattern, batch_size=args.batch_size)

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, padding_side="left")

    model = CoLLMStage1(
        tokenizer=tokenizer,
        llm_model_name=args.llm_model,
        llm_dim=args.llm_dim,
        lora_rank=args.lora_rank,
        llm_precision=args.llm_precision,
        attn_implementation=args.attn_implementation,
        device=device,
    )
    model.train()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,}")
    print(f"Total params:     {total_params:,}")

    logit_scale_params = [model.logit_scale]
    logit_scale_param_ids = {id(p) for p in logit_scale_params}
    main_params = [
        p
        for p in model.parameters()
        if p.requires_grad and id(p) not in logit_scale_param_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": main_params, "lr": args.lr, "weight_decay": 0.01},
            {"params": logit_scale_params, "lr": args.logit_scale_lr, "weight_decay": 0.0},
        ]
    )
    warmup_steps = max(1, int(args.max_steps * args.warmup_ratio))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    optimizer.zero_grad()

    print(f"Warmup steps:           {warmup_steps}")
    print("Starting training...")

    for step, batch in enumerate(dataloader):
        aug_images = batch["aug_images"].to(device, non_blocking=True)
        target_images = batch["target_images"].to(device, non_blocking=True)
        captions = batch["captions"]

        # --- Full-batch CLIP path (never micro-batched) ---
        with torch.no_grad():
            z = model.encode_target(target_images)          # [B, 768]

        h_prime = model.encode_reference(aug_images)        # [B, 1024]

        with torch.no_grad():
            nn_embeds, nn_indices = get_nearest_neighbors(h_prime.detach())

        captions_j = [captions[i.item()] for i in nn_indices]

        h_star = slerp(h_prime, nn_embeds, t=0.5)           # [B, 1024]
        mod_texts = get_modification_texts(captions, captions_j, synthesis_ratio=0.75)

        # --- LLM-only micro-batching; concat back to [B, D] for [B x B] loss ---
        c_v, c_w, c = model.forward_llm_queries_microbatched(
            h_star,
            captions,
            mod_texts,
            micro_batch=args.llm_micro_batch,
            device=model.llm_device,
        )

        # Single [B x B] contrastive loss — equivalent to one forward at batch_size=B.
        loss, loss_dict = collm_loss(c_v, c_w, c, z, model.logit_scale)
        if not torch.isfinite(loss):
            print(f"Skipping step {step}: non-finite loss {loss.item()}", flush=True)
            optimizer.zero_grad(set_to_none=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            model.logit_scale.clamp_(min=0.0, max=math.log(100.0))
        optimizer.zero_grad(set_to_none=True)

        temperature_val = torch.exp(model.logit_scale.float()).item()

        writer.add_scalar("Loss/Total", loss_dict["loss"], step)
        writer.add_scalar("Loss/Image_Only_cv", loss_dict["img_only"], step)
        writer.add_scalar("Loss/Text_Only_cw", loss_dict["txt_only"], step)
        writer.add_scalar("Loss/Composed_c", loss_dict["comp"], step)
        writer.add_scalar("Metrics/Temperature", temperature_val, step)
        writer.add_scalar("LR/Main", scheduler.get_last_lr()[0], step)
        writer.add_scalar("LR/Logit_Scale", scheduler.get_last_lr()[1], step)

        step_logs.append({
            "step": step,
            "loss": loss_dict["loss"],
            "loss_cv": loss_dict["img_only"],
            "loss_cw": loss_dict["txt_only"],
            "loss_c": loss_dict["comp"],
            "temperature": temperature_val,
        })

        if step % 10 == 0:
            print(
                f"Step {step:4d} | "
                f"Loss {loss_dict['loss']:.4f} | "
                f"L_v {loss_dict['img_only']:.4f} | "
                f"L_w {loss_dict['txt_only']:.4f} | "
                f"L_c {loss_dict['comp']:.4f} | "
                f"Temp {temperature_val:.2f}",
                flush=True,
            )

        if step > 0 and step % 1000 == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint_step_{step}")
            os.makedirs(ckpt_dir, exist_ok=True)

            model.clip.vision_model.save_pretrained(os.path.join(ckpt_dir, "clip_lora"))
            model.llm.save_pretrained(os.path.join(ckpt_dir, "llm_lora"))
            torch.save(
                {
                    "step": step,
                    "image_adapter": model.image_adapter.state_dict(),
                    "projection": model.projection.state_dict(),
                    "logit_scale": model.logit_scale.detach().cpu(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(ckpt_dir, "extra_modules.pt"),
            )
            print(f"Saved Checkpoint -> {ckpt_dir}")

        if step + 1 >= args.max_steps:
            print(f"Reached max_steps={args.max_steps}")
            break

    final_dir = os.path.join(args.output_dir, "checkpoint_final")
    os.makedirs(final_dir, exist_ok=True)

    model.clip.vision_model.save_pretrained(os.path.join(final_dir, "clip_lora"))
    model.llm.save_pretrained(os.path.join(final_dir, "llm_lora"))
    torch.save(
        {
            "image_adapter": model.image_adapter.state_dict(),
            "projection": model.projection.state_dict(),
            "logit_scale": model.logit_scale.detach().cpu(),
        },
        os.path.join(final_dir, "extra_modules.pt"),
    )

    print(f"\nFinal Checkpoint Saved -> {final_dir}")

    with open(os.path.join(args.log_dir, "training_logs.json"), "w") as f:
        json.dump(step_logs, f, indent=4)
    print(f"Logs saved -> {args.log_dir}/training_logs.json")

    writer.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
