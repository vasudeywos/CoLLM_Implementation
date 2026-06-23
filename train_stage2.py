"""
train_stage2.py — Stage-2 CIR-triplet fine-tuning for CoLLM.

Run merge_stage1_checkpoint.py first. See model_stage2.py and
dataset_stage2.py docstrings for the full design rationale.

Key differences from Stage-1's train.py (paper Sec. 3.3 / 9.2):
  - input is REAL triplets, not synthesized from image-caption pairs:
    no augmentation, no nearest-neighbor search, no Slerp, no modification-
    text templates.
  - single contrastive loss L = L_cl(c_i, z_i), not the 3-way
    (image-only / text-only / composed) average used in Stage 1.
  - vision encoder (CLIP), image_adapter, projection, and logit_scale are
    frozen; only the new rank-16 LLM LoRA is trainable.
  - the paper trains this stage for ONE epoch and reports that further
    epochs overfit (Table 15: CIRR recall-sum peaks at epoch 1, degrades
    every epoch after) — this script defaults to --epochs 1 accordingly.

Usage:
    python train_stage2.py \
        --merged_dir ./checkpoints_stage1_run5/merged_for_stage2 \
        --manifest   /path/to/mtcir_train.jsonl \
        --image_root /path/to/mtcir_images \
        --llm_model  Salesforce/SFR-Embedding-2_R \
        --output_dir ./checkpoints_stage2_run1
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from peft import set_peft_model_state_dict
from peft.utils.save_and_load import load_peft_weights

from dataset_stage2 import get_triplet_dataloader
from model_stage2 import build_stage2_model


def collm_single_contrastive_loss(c, z, logit_scale):
    """Single-triplet version of Stage-1's loss.py: just L_cl(c_i, z_i),
    per paper Sec. 3.3 ('The training objective is L = Lcl(ci, zi)')."""
    safe_logit_scale = torch.clamp(logit_scale.float(), min=0.0, max=math.log(100.0))
    temp = torch.exp(safe_logit_scale)
    c = c.float()
    z = z.float()
    logits = (c @ z.T) * temp
    labels = torch.arange(len(c), device=c.device)
    loss_q2t = F.cross_entropy(logits, labels)
    loss_t2q = F.cross_entropy(logits.T, labels)
    loss = (loss_q2t + loss_t2q) / 2.0
    return loss, temp.item()


def save_checkpoint(model, output_dir, name, optimizer, scheduler, step):
    ckpt_dir = os.path.join(output_dir, name)
    os.makedirs(ckpt_dir, exist_ok=True)
    # Only the LLM has LoRA adapters in Stage-2; CLIP is a plain frozen model.
    model.llm.save_pretrained(os.path.join(ckpt_dir, "llm_lora_stage2"))
    extra = {
        "image_adapter": model.image_adapter.state_dict(),
        "projection": model.projection.state_dict(),
        "logit_scale": model.logit_scale.detach().cpu(),
    }
    extra["step"] = step
    if optimizer is not None:
        extra["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            extra["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(extra, os.path.join(ckpt_dir, "extra_modules.pt"))
    print(f"Saved checkpoint -> {ckpt_dir}")


def load_lora_weights(peft_model, lora_dir):
    lora_state = load_peft_weights(lora_dir, device="cpu")
    set_peft_model_state_dict(peft_model, lora_state, adapter_name="default")
    peft_model.set_adapter("default")


def load_training_checkpoint(model, checkpoint_dir, device):
    llm_lora_dir = os.path.join(checkpoint_dir, "llm_lora_stage2")
    if not os.path.isdir(llm_lora_dir):
        raise FileNotFoundError(f"Missing Stage-2 LLM LoRA checkpoint: {llm_lora_dir}")
    print(f"Loading Stage-2 LLM LoRA from {llm_lora_dir}")
    load_lora_weights(model.llm, llm_lora_dir)

    extra_path = os.path.join(checkpoint_dir, "extra_modules.pt")
    if not os.path.isfile(extra_path):
        raise FileNotFoundError(f"Missing checkpoint extras: {extra_path}")
    print(f"Loading Stage-2 extras from {extra_path}")
    extra = torch.load(extra_path, map_location="cpu")
    model.image_adapter.load_state_dict(extra["image_adapter"])
    model.projection.load_state_dict(extra["projection"])
    model.logit_scale.data = extra["logit_scale"].to(device)
    model.place_trainable_modules(device)
    return extra

def load_lora_for_init(model, checkpoint_dir, device, optimizer=None, scheduler=None):
    ckpt = Path(checkpoint_dir)

    # Load LoRA weights
    llm_lora_dir = ckpt / "llm_lora_stage2"
    if llm_lora_dir.exists():
        print(f"Init: loading LLM LoRA from {llm_lora_dir}")
        from peft.utils.save_and_load import load_peft_weights
        from peft import set_peft_model_state_dict
        lora_state = load_peft_weights(str(llm_lora_dir), device="cpu")
        set_peft_model_state_dict(model.llm, lora_state, adapter_name="default")
        model.llm.set_adapter("default")

    # Load bridge weights
    extra_pt = ckpt / "extra_modules.pt"
    if extra_pt.exists():
        extra = torch.load(extra_pt, map_location="cpu")
        model.image_adapter.load_state_dict(extra["image_adapter"])
        model.projection.load_state_dict(extra["projection"])
        model.logit_scale.data = extra["logit_scale"].to(device)

        # Restore optimizer and scheduler if provided
        if optimizer is not None and "optimizer_state_dict" in extra:
            optimizer.load_state_dict(extra["optimizer_state_dict"])
            print(f"Init: optimizer state restored from {extra_pt}")
        else:
            print(f"Init: optimizer state NOT restored (not found or not requested)")

        if scheduler is not None and "scheduler_state_dict" in extra:
            scheduler.load_state_dict(extra["scheduler_state_dict"])
            # Reset last_epoch to 0 — new chunk is a fresh epoch
            scheduler.last_epoch = 0
            print(f"Init: scheduler state restored, last_epoch reset to 0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_dir", type=str, required=True,
                         help="Output of merge_stage1_checkpoint.py")
    parser.add_argument("--manifest", type=str, required=True,
                         help="JSONL triplet manifest, see dataset_stage2.py")
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--llm_model", type=str, default="Salesforce/SFR-Embedding-2_R")
    parser.add_argument("--llm_dim", type=int, default=4096)
    parser.add_argument("--lora_rank", type=int, default=16,
                         help="Paper Sec 9.2: rank reduced to 16 for fine-tuning.")
    parser.add_argument("--lora_alpha", type=int, default=16,
                         help="Paper Sec 9.2: alpha reduced to 16 for fine-tuning.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--llm_micro_batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5,
                         help="Paper used a constant 1e-4 for Stage-1 at batch "
                              "size 1024 ('other settings remain consistent "
                              "with pre-training', Sec 9.2) — default here "
                              "matches the LR your Stage-1 run actually used; "
                              "set this to whatever you used there.")
    parser.add_argument("--logit_scale_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                         help="0 by default — this fine-tunes a converged "
                              "checkpoint rather than training from scratch.")
    parser.add_argument("--epochs", type=int, default=1,
                         help="Paper Table 15: rapid overfitting after epoch 1.")
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--log_dir", type=str, default="./logs_stage2")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_stage2")
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a Stage-2 checkpoint_step_* directory to resume training from.",
    )
    parser.add_argument("--llm_precision", type=str, default="auto", choices=["auto", "4bit", "bf16"])
    parser.add_argument("--attn_implementation", type=str, default="eager", choices=["auto", "sdpa", "eager"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--init_from", type=str, default=None,
                    help="Load LoRA + adapter/projection weights from a previous "
                         "Stage-2 checkpoint but reset step counter. "
                         "Use this to chain chunk runs.")
    args = parser.parse_args()

    if args.batch_size % args.llm_micro_batch != 0:
        raise ValueError("batch_size must be divisible by llm_micro_batch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, padding_side="left")

    print("Building Stage-2 model from merged checkpoint...")
    model = build_stage2_model(
        tokenizer=tokenizer,
        merged_dir=args.merged_dir,
        llm_dim=args.llm_dim,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        llm_precision=args.llm_precision,
        attn_implementation=args.attn_implementation,
        gradient_checkpointing=args.gradient_checkpointing,
        device=device,
    )

    model.train()
    model.clip.eval()  # frozen vision encoder always stays in eval mode
    resume_extra = None
    start_step = 0
    if args.resume_from is not None:
        resume_extra = load_training_checkpoint(model, args.resume_from, device)
        if "step" not in resume_extra:
            raise KeyError(
                "Resume checkpoint is missing 'step'. Use a checkpoint_step_* "
                "directory, not checkpoint_final."
            )
        start_step = int(resume_extra["step"]) + 1
        print(f"Resuming Stage-2 training from step {start_step}")
        model.train()
        model.clip.eval()

    trainable_params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print("\nTrainable parameter groups:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {name}")

    print(f"Manifest:   {args.manifest}")
    print(f"Image Root: {args.image_root}")
    dataloader = get_triplet_dataloader(
        args.manifest,
        args.image_root,
        batch_size=args.batch_size,
        num_workers=8,
    )
    steps_per_epoch = len(dataloader)
    max_steps = steps_per_epoch * args.epochs
    print(f"Triplets: {len(dataloader.dataset):,} | Steps/epoch: {steps_per_epoch} | Total steps: {max_steps}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=0.01,
        eps=1e-6,
        betas=(0.9, 0.999),
    )

    warmup_steps = max(1, int(max_steps * args.warmup_ratio)) if args.warmup_ratio > 0 else 0

    def lr_lambda(step):
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    if args.init_from is not None and args.resume_from is None:
        load_lora_for_init(
            model,
            args.init_from,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        print(
            f"Initialized from {args.init_from} with optimizer state carried over; "
            "step counter reset to 0"
        )

    if resume_extra is not None:
        if "optimizer_state_dict" not in resume_extra:
            raise KeyError(
                "Resume checkpoint is missing optimizer_state_dict. "
                "Use a Stage-2 checkpoint_step_* checkpoint."
            )
        optimizer.load_state_dict(resume_extra["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_extra:
            scheduler.load_state_dict(resume_extra["scheduler_state_dict"])
        else:
            scheduler.last_epoch = start_step - 1

    if start_step >= max_steps:
        raise ValueError(
            f"Resume step {start_step} is already >= total steps ({max_steps}). "
            "Increase --epochs to continue training."
        )

    step = start_step
    step_logs = []
    start_epoch = start_step // steps_per_epoch
    for epoch in range(start_epoch, args.epochs):
        for batch in dataloader:
            if step < start_step:
                step += 1
                continue
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            ref_imgs = batch["ref_img"].to(device, non_blocking=True)
            target_imgs = batch["target_img"].to(device, non_blocking=True)
            mod_texts = batch["mod_text"]

            c, z = model.forward_triplet_microbatched(
                ref_imgs, target_imgs, mod_texts,
                micro_batch=args.llm_micro_batch, device=model.llm_device,
            )
            loss, temperature_val = collm_single_contrastive_loss(c, z, model.logit_scale)

            if not torch.isfinite(loss):
                print(f"Skipping step {step}: non-finite loss")
                optimizer.zero_grad(set_to_none=True)
                step += 1
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=1.0,
                error_if_nonfinite=False,
            )
            if not torch.isfinite(grad_norm):
                print(f"Skipping step {step}: non-finite grad norm")
                optimizer.zero_grad(set_to_none=True)
                step += 1
                continue

            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                model.logit_scale.clamp_(min=0.0, max=math.log(100.0))
            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            step_time = time.perf_counter() - t0

            writer.add_scalar("Loss/Triplet", loss.item(), step)
            writer.add_scalar("Metrics/Temperature", temperature_val, step)
            writer.add_scalar("LR/Main", scheduler.get_last_lr()[0], step)
            writer.add_scalar("Timing/Step_Seconds", step_time, step)
            step_logs.append({
                "step": step, "epoch": epoch, "loss": loss.item(),
                "temperature": temperature_val, "step_time_sec": step_time,
            })

            if step % 10 == 0:
                print(f"Epoch {epoch} | Step {step:5d}/{max_steps} | "
                      f"Loss {loss.item():.4f} | Temp {temperature_val:.2f} | "
                      f"Time {step_time:.2f}s", flush=True)

            if step > 0 and step % args.save_every == 0:
                save_checkpoint(
                    model,
                    args.output_dir,
                    f"checkpoint_step_{step}",
                    optimizer,
                    scheduler,
                    step,
                )

            step += 1
            if step >= max_steps:
                break

    save_checkpoint(
        model,
        args.output_dir,
        "checkpoint_final",
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
    )
    with open(os.path.join(args.log_dir, "training_logs.json"), "w") as f:
        json.dump(step_logs, f, indent=4)
    writer.close()
    print("Stage-2 training complete!")


if __name__ == "__main__":
    main()
