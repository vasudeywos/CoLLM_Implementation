"""
merge_stage1_checkpoint.py — bake Stage-1 LoRA adapters into clean base
weights, producing a checkpoint that Stage-2 can re-LoRA at a smaller rank.

WHY THIS STEP IS NEEDED
------------------------
The paper's only instruction for Stage-2 (Sec. 9.2) is: "we reduce the
number of trainable parameters by setting the LoRA rank and alpha to 16
[down from 64] ... only the LLM is fine-tuned."

A PEFT LoRA adapter's A/B matrices are shaped by its rank, so you cannot
load a rank-64 adapter's saved weights into a freshly-constructed rank-16
LoraConfig — the shapes don't match. To continue training from the Stage-1
checkpoint under a new, smaller LoRA config, the Stage-1 adaptation has to
be folded ("merged") into the base weights first; a brand-new rank-16
adapter is then attached on top of those merged weights for Stage-2
(see model_stage2.py).

This script does that merge ONCE, offline, for:
  - the CLIP vision tower, which then gets frozen completely for Stage-2
    (no Stage-2 LoRA on it at all — "vision encoder features are already
    aligned" per Sec. 9.2);
  - the LLM, which gets a brand-new rank-16 adapter in model_stage2.py.

The LLM is merged in full precision (bf16/fp16), NOT 4-bit — merging LoRA
deltas into 4-bit-quantized weights is unreliable. The merged LLM is saved
in full precision; model_stage2.py re-quantizes it to 4-bit on load for
Stage-2 training, so you keep the same memory profile during fine-tuning.

image_adapter / projection / logit_scale never touch CLIP's or the LLM's
own weights (they're separate small modules), so there's nothing to merge
for them — they're just copied through unchanged.

Usage:
    python merge_stage1_checkpoint.py \
        --stage1_checkpoint_dir ./checkpoints_stage1_run5/checkpoint_final \
        --llm_model  Salesforce/SFR-Embedding-2_R \
        --clip_model openai/clip-vit-large-patch14 \
        --output_dir ./checkpoints_stage1_run5/merged_for_stage2
"""

import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoModel, CLIPModel
from peft import PeftModel


def resolve_merge_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def merge_clip(stage1_dir: Path, clip_model_name: str, output_dir: Path):
    clip_lora_dir = stage1_dir / "clip_lora"
    print(f"Loading base CLIP: {clip_model_name}")
    clip = CLIPModel.from_pretrained(clip_model_name)

    if not clip_lora_dir.exists():
        print(f"WARNING: no clip_lora dir at {clip_lora_dir} — saving base CLIP unmerged.")
    else:
        print(f"Merging CLIP LoRA from {clip_lora_dir}")
        clip.vision_model = PeftModel.from_pretrained(clip.vision_model, str(clip_lora_dir))
        clip.vision_model = clip.vision_model.merge_and_unload()

    out = output_dir / "clip_merged"
    out.mkdir(parents=True, exist_ok=True)
    clip.save_pretrained(str(out))
    print(f"Saved merged CLIP -> {out}")


def merge_llm(stage1_dir: Path, llm_model_name: str, output_dir: Path, device: torch.device):
    llm_lora_dir = stage1_dir / "llm_lora"
    merge_dtype = resolve_merge_dtype(device)
    print(f"Loading base LLM in {merge_dtype}: {llm_model_name}")
    load_kwargs = {"torch_dtype": merge_dtype}
    if device.type == "cuda":
        # One-time merge, no optimizer states needed — a ~7B model fits
        # comfortably in fp16/bf16 on a 48GB card, so do it on GPU for speed.
        load_kwargs["device_map"] = {"": device.index or 0}
    # NOTE: deliberately NOT 4-bit here — merging LoRA into a 4-bit base is unreliable.
    llm = AutoModel.from_pretrained(llm_model_name, **load_kwargs)

    if not llm_lora_dir.exists():
        print(f"WARNING: no llm_lora dir at {llm_lora_dir} — saving base LLM unmerged.")
    else:
        print(f"Merging LLM LoRA from {llm_lora_dir}")
        llm = PeftModel.from_pretrained(llm, str(llm_lora_dir))
        llm = llm.merge_and_unload()

    out = output_dir / "llm_merged"
    out.mkdir(parents=True, exist_ok=True)
    llm.save_pretrained(str(out))
    print(f"Saved merged LLM -> {out}")


def carry_over_bridge_weights(stage1_dir: Path, output_dir: Path):
    """image_adapter / projection / logit_scale aren't part of CLIP or the
    LLM, so there's nothing to merge — just copy them through so Stage-2
    has a single, self-contained checkpoint directory."""
    src = stage1_dir / "extra_modules.pt"
    dst = output_dir / "extra_modules.pt"
    if not src.exists():
        raise FileNotFoundError(f"Missing {src} — Stage-1 checkpoint dir looks incomplete.")
    shutil.copyfile(src, dst)
    print(f"Copied bridge weights -> {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_checkpoint_dir", type=str, required=True)
    parser.add_argument("--llm_model", type=str, default="Salesforce/SFR-Embedding-2_R")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1_dir = Path(args.stage1_checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merge_clip(stage1_dir, args.clip_model, output_dir)
    merge_llm(stage1_dir, args.llm_model, output_dir, device)
    carry_over_bridge_weights(stage1_dir, output_dir)

    print(f"\nDone. Stage-2 can now build from -> {output_dir}")
    print("(model_stage2.build_stage2_model() expects exactly this layout:")
    print("  output_dir/clip_merged/, output_dir/llm_merged/, output_dir/extra_modules.pt)")


if __name__ == "__main__":
    main()