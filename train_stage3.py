"""
Train CoLLM directly on MTCIR triplets from base CLIP and SFR models.

Trainable components:
  * optional CLIP vision LoRA
  * SFR LLM LoRA
  * image adapter
  * output projection
  * contrastive logit scale

The objective remains the paper's triplet objective L = L_cl(c_i, z_i).
"""

import argparse
import json
import math
import os
import time
from collections import Counter

import torch
import torch.nn.functional as F
from peft import set_peft_model_state_dict
from peft.utils.save_and_load import load_peft_weights
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from dataset_stage3 import get_triplet_dataloader
from model_stage3 import build_stage3_model


def contrastive_loss(query_features, target_features, logit_scale):
    safe_scale = torch.clamp(
        logit_scale.float(),
        min=0.0,
        max=math.log(100.0),
    )
    temperature = torch.exp(safe_scale)
    logits = query_features.float() @ target_features.float().T
    logits = logits * temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    query_to_target = F.cross_entropy(logits, labels)
    target_to_query = F.cross_entropy(logits.T, labels)
    return (query_to_target + target_to_query) / 2.0, temperature.item()


def save_checkpoint(model, output_dir, name, optimizer, scheduler, step):
    checkpoint_dir = os.path.join(output_dir, name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.clip.vision_model.save_pretrained(
        os.path.join(checkpoint_dir, "clip_lora")
    )
    model.llm.save_pretrained(os.path.join(checkpoint_dir, "llm_lora"))
    torch.save(
        {
            "step": step,
            "image_adapter": model.image_adapter.state_dict(),
            "projection": model.projection.state_dict(),
            "logit_scale": model.logit_scale.detach().cpu(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        os.path.join(checkpoint_dir, "extra_modules.pt"),
    )
    print(f"Saved checkpoint -> {checkpoint_dir}")


def load_lora_weights(peft_model, adapter_dir):
    state = load_peft_weights(adapter_dir, device="cpu")
    set_peft_model_state_dict(peft_model, state, adapter_name="default")
    peft_model.set_adapter("default")


def load_checkpoint(model, checkpoint_dir, device):
    clip_lora_dir = os.path.join(checkpoint_dir, "clip_lora")
    llm_lora_dir = os.path.join(checkpoint_dir, "llm_lora")
    extra_path = os.path.join(checkpoint_dir, "extra_modules.pt")

    if not os.path.isdir(clip_lora_dir):
        raise FileNotFoundError(f"Missing CLIP LoRA: {clip_lora_dir}")
    if not os.path.isdir(llm_lora_dir):
        raise FileNotFoundError(f"Missing SFR LoRA: {llm_lora_dir}")
    if not os.path.isfile(extra_path):
        raise FileNotFoundError(f"Missing checkpoint extras: {extra_path}")

    load_lora_weights(model.clip.vision_model, clip_lora_dir)
    load_lora_weights(model.llm, llm_lora_dir)
    extra = torch.load(extra_path, map_location="cpu")
    model.image_adapter.load_state_dict(extra["image_adapter"])
    model.projection.load_state_dict(extra["projection"])
    model.logit_scale.data = extra["logit_scale"].float().to(device)
    model.place_modules(device)
    return extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-large-patch14",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="Salesforce/SFR-Embedding-2_R",
    )
    parser.add_argument("--clip_dim", type=int, default=1024)
    parser.add_argument("--llm_dim", type=int, default=4096)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--clip_lora_rank", type=int, default=16)
    parser.add_argument("--llm_lora_rank", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--llm_micro_batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--train_clip_lora_from_start",
        action="store_true",
        help="Opt out of the safer default and train CLIP LoRA from step 0.",
    )
    parser.add_argument(
        "--unfreeze_clip_at_step",
        type=int,
        default=None,
        help="Global step at which to unfreeze CLIP LoRA after starting frozen.",
    )

    # Separate rates prevent the random bridge from learning at the same
    # conservative rate used for pretrained CLIP/SFR representations.
    parser.add_argument("--bridge_lr", type=float, default=5e-5)
    parser.add_argument("--llm_lora_lr", type=float, default=5e-5)
    parser.add_argument("--clip_lora_lr", type=float, default=1e-5)
    parser.add_argument("--logit_scale_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_dir", type=str, default="./logs_stage3")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_stage3")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument(
        "--llm_precision",
        type=str,
        default="4bit",
        choices=["auto", "4bit", "bf16"],
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["auto", "sdpa", "eager"],
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    args = parser.parse_args()

    if args.batch_size % args.llm_micro_batch != 0:
        raise ValueError("batch_size must be divisible by llm_micro_batch")
    if args.unfreeze_clip_at_step is not None and args.unfreeze_clip_at_step < 0:
        raise ValueError("unfreeze_clip_at_step must be >= 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_model,
        padding_side="left",
    )
    model = build_stage3_model(
        tokenizer=tokenizer,
        clip_model_name=args.clip_model,
        llm_model_name=args.llm_model,
        clip_dim=args.clip_dim,
        llm_dim=args.llm_dim,
        embed_dim=args.embed_dim,
        clip_lora_rank=args.clip_lora_rank,
        llm_lora_rank=args.llm_lora_rank,
        llm_precision=args.llm_precision,
        attn_implementation=args.attn_implementation,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_clip_lora=(not args.train_clip_lora_from_start),
        device=device,
    )
    model.train()

    dataloader = get_triplet_dataloader(
        manifest_path=args.manifest,
        image_root=args.image_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * args.epochs

    clip_lora_params = [
        parameter
        for parameter in model.clip.vision_model.parameters()
        if parameter.requires_grad
    ]
    llm_lora_params = [
        parameter for parameter in model.llm.parameters() if parameter.requires_grad
    ]
    bridge_params = [
        *model.image_adapter.parameters(),
        *model.projection.parameters(),
    ]
    all_trainable_params = [
        *clip_lora_params,
        *llm_lora_params,
        *bridge_params,
        model.logit_scale,
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": clip_lora_params,
                "lr": args.clip_lora_lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": llm_lora_params,
                "lr": args.llm_lora_lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": bridge_params,
                "lr": args.bridge_lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": [model.logit_scale],
                "lr": args.logit_scale_lr,
                "weight_decay": 0.0,
            },
        ],
        betas=(0.9, 0.999),
        eps=1e-6,
    )

    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_step = 0
    if args.resume_from:
        extra = load_checkpoint(model, args.resume_from, device)
        optimizer.load_state_dict(extra["optimizer_state_dict"])
        scheduler.load_state_dict(extra["scheduler_state_dict"])
        start_step = int(extra["step"]) + 1
        print(f"Resuming from global step {start_step}")
        if not args.train_clip_lora_from_start:
            model.set_clip_lora_trainable(False)

    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device}")
    print(f"Triplets: {len(dataloader.dataset):,}")
    print(f"Steps/epoch: {steps_per_epoch:,} | Total steps: {total_steps:,}")
    print(
        f"Trainable parameters: {trainable_count:,} / {total_count:,} "
        f"({100.0 * trainable_count / total_count:.2f}%)"
    )
    print(
        "Optimizer groups: "
        f"CLIP LoRA={sum(p.numel() for p in clip_lora_params):,}, "
        f"SFR LoRA={sum(p.numel() for p in llm_lora_params):,}, "
        f"bridge={sum(p.numel() for p in bridge_params):,}, "
        "logit_scale=1"
    )
    if model.clip_lora_frozen:
        if args.unfreeze_clip_at_step is None:
            print("CLIP LoRA schedule: frozen for entire run")
        else:
            print(
                "CLIP LoRA schedule: "
                f"frozen until global step {args.unfreeze_clip_at_step}"
            )
    else:
        print("CLIP LoRA schedule: trainable from step 0")

    global_step = 0
    training_logs = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(dataloader):
            global_step = epoch * steps_per_epoch + batch_index
            if global_step < start_step:
                continue
            if (
                model.clip_lora_frozen
                and args.unfreeze_clip_at_step is not None
                and global_step >= args.unfreeze_clip_at_step
            ):
                model.set_clip_lora_trainable(True)
                print(
                    f"Unfroze CLIP LoRA at global step {global_step}",
                    flush=True,
                )

            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            ref_imgs = batch["ref_img"].to(device, non_blocking=True)
            target_imgs = batch["target_img"].to(device, non_blocking=True)
            mod_texts = batch["mod_text"]

            query_features, target_features = model.forward_triplet_microbatched(
                ref_imgs=ref_imgs,
                target_imgs=target_imgs,
                mod_texts=mod_texts,
                micro_batch=args.llm_micro_batch,
                device=model.llm_device,
            )
            loss, temperature = contrastive_loss(
                query_features,
                target_features,
                model.logit_scale,
            )

            if not torch.isfinite(loss):
                print(f"Skipping step {global_step}: non-finite loss")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                all_trainable_params,
                max_norm=args.max_grad_norm,
                error_if_nonfinite=False,
            )
            if not torch.isfinite(grad_norm):
                print(f"Skipping step {global_step}: non-finite gradient norm")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                model.logit_scale.clamp_(0.0, math.log(100.0))

            if device.type == "cuda":
                torch.cuda.synchronize()
            step_seconds = time.perf_counter() - start_time
            sampling_counts = Counter(batch["sampling_mode"])

            writer.add_scalar("Loss/Triplet", loss.item(), global_step)
            writer.add_scalar("Metrics/Temperature", temperature, global_step)
            writer.add_scalar("Metrics/GradientNorm", grad_norm.item(), global_step)
            writer.add_scalar("Timing/StepSeconds", step_seconds, global_step)
            for group_index, learning_rate in enumerate(scheduler.get_last_lr()):
                writer.add_scalar(
                    f"LR/Group{group_index}",
                    learning_rate,
                    global_step,
                )

            training_logs.append(
                {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": loss.item(),
                    "temperature": temperature,
                    "grad_norm": grad_norm.item(),
                    "step_time_sec": step_seconds,
                    "sampling_modes": dict(sampling_counts),
                }
            )

            if global_step % 10 == 0:
                print(
                    f"Epoch {epoch} | Step {global_step:5d}/{total_steps} | "
                    f"Loss {loss.item():.4f} | Temp {temperature:.2f} | "
                    f"Grad {grad_norm.item():.3f} | Time {step_seconds:.2f}s | "
                    f"Modes {dict(sampling_counts)}",
                    flush=True,
                )

            if global_step > 0 and global_step % args.save_every == 0:
                save_checkpoint(
                    model,
                    args.output_dir,
                    f"checkpoint_step_{global_step}",
                    optimizer,
                    scheduler,
                    global_step,
                )

    save_checkpoint(
        model,
        args.output_dir,
        "checkpoint_final",
        optimizer,
        scheduler,
        global_step,
    )
    with open(
        os.path.join(args.log_dir, "training_logs.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(training_logs, file, indent=2)
    writer.close()
    print("Stage-3 training complete.")


if __name__ == "__main__":
    main()
