"""
eval_fiq.py — Zero-shot Recall@10 on Fashion-IQ for CoLLM Stage-1

Adapted recall logic from WebCoVR (src/test/blip2/fashioniq.py).
Gallery encoding is done live with model.encode_target() (no pre-computed .pth files needed).

Usage:
    python eval_fiq.py \
        --checkpoint_dir ./checkpoints_stage1_run5/checkpoint_final \
        --fiq_image_dir   /path/to/fashionIQ_dataset/images \
        --fiq_caption_dir /path/to/fashionIQ_dataset/captions \
        --fiq_split_dir   /path/to/fashionIQ_dataset/image_splits \
        --llm_model      Salesforce/SFR-Embedding-2_R \
        --llm_dim        4096 \
        --llm_precision  4bit \
        --attn_implementation eager \
        --batch_size     64

Fashion-IQ data structure expected (matches the official repo layout):
    {fiq_caption_dir}/cap.{category}.val.json   — list of {candidate, target, captions:[str,str]}
    {fiq_split_dir}/split.{category}.val.json   — flat list of image ids that form the gallery
    {fiq_image_dir}/{image_id}.png (or .jpg)    — all images in one flat folder

Image files: {fiq_image_dir}/{image_id}.png  (or .jpg — script tries both)
"""

import argparse
import json
import re
import time
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer

from model import CoLLMStage1, resolve_attn_implementation, resolve_llm_quantization
from peft import PeftModel  # noqa: F401 (kept for reference; loading uses load_adapter instead)
from model_stage2 import build_stage2_model

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

FIQ_CATEGORIES = ["dress", "shirt", "toptee"]

# ─────────────────────────────────────────────────────────────────────────────
# Image transform — 224px to match your CLIP training setup
# ─────────────────────────────────────────────────────────────────────────────

eval_transform = transforms.Compose([
    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

# ─────────────────────────────────────────────────────────────────────────────
# Caption preprocessing — adopted from WebCoVR's src/data/utils.py
# ─────────────────────────────────────────────────────────────────────────────

def pre_caption(caption: str, max_words: int = 30) -> str:
    caption = re.sub(r"([.!\"()*#:;~])", " ", caption.lower())
    caption = re.sub(r"\s{2,}", " ", caption).rstrip("\n").strip()
    words = caption.split()
    if len(words) > max_words:
        caption = " ".join(words[:max_words])
    return caption

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def find_image(img_dir: Path, image_id: str) -> Path:
    """Try .png then .jpg — FashionIQ ships as .png but some mirrors use .jpg."""
    for ext in [".png", ".jpg", ".jpeg"]:
        p = img_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Image not found for id={image_id} in {img_dir}")


class FashionIQQueryDataset(Dataset):
    """
    Returns one item per query pair:
        ref_img   : [3, 224, 224] tensor
        caption   : joined + cleaned modification text  (two captions merged)
        pair_id   : int index
        target_id : str  (ground-truth target image id)
        ref_id    : str  (reference image id — used to mask ref==target in sim matrix)
    """
    def __init__(self, annotation_path: str, img_dir: str):
        self.annotation = json.load(open(annotation_path))
        self.img_dir = Path(img_dir)

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        ann = self.annotation[idx]
        ref_id = ann["candidate"]
        tar_id = ann["target"]
        cap1, cap2 = ann["captions"]
        caption = pre_caption(f"{cap1} and {cap2}", max_words=30)

        ref_img = Image.open(find_image(self.img_dir, ref_id)).convert("RGB")
        ref_img = eval_transform(ref_img)

        return {
            "ref_img":   ref_img,
            "caption":   caption,
            "pair_id":   idx,
            "target_id": tar_id,
            "ref_id":    ref_id,
        }


class FashionIQGalleryDataset(Dataset):
    """All unique gallery images for one category."""
    def __init__(self, gallery_ids: list, img_dir: str):
        self.gallery_ids = gallery_ids
        self.img_dir = Path(img_dir)

    def __len__(self):
        return len(self.gallery_ids)

    def __getitem__(self, idx):
        img_id = self.gallery_ids[idx]
        img = Image.open(find_image(self.img_dir, img_id)).convert("RGB")
        return eval_transform(img), img_id

# ─────────────────────────────────────────────────────────────────────────────
# Recall helpers — directly adopted from WebCoVR's fashioniq.py
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(sim: torch.Tensor, query_lbls: list, target_lbls: np.ndarray, k: int) -> float:
    """
    sim        : [N_queries, N_gallery] cosine similarity matrix (higher = more similar)
    query_lbls : list of ground-truth target image ids, length N_queries
    target_lbls: np.array of gallery image ids,        length N_gallery
    k          : cutoff
    Returns Recall@k as a percentage (0–100).
    """
    # Sort descending by similarity (most similar first)
    sorted_indices = torch.argsort(sim, dim=-1, descending=True).cpu()
    sorted_names   = target_lbls[sorted_indices.numpy()]          # [N_q, N_gallery]

    query_lbls_arr = np.array(query_lbls)                         # [N_q]
    # labels[i,j] = True if sorted_names[i,j] == ground truth for query i
    labels = torch.tensor(
        sorted_names == np.repeat(query_lbls_arr, len(target_lbls)).reshape(len(query_lbls), -1)
    )
    # Sanity: every query has exactly one correct match
    assert torch.equal(
        torch.sum(labels, dim=-1).int(),
        torch.ones(len(query_lbls)).int()
    ), "Each query must have exactly one ground-truth target in the gallery."

    return round((torch.sum(labels[:, :k]) / len(labels)).item() * 100, 2)


def get_recalls(sim: torch.Tensor, query_lbls: list, target_lbls: np.ndarray,
                ks=(1, 5, 10, 50)) -> dict:
    return {f"R{k}": recall_at_k(sim, query_lbls, target_lbls, k) for k in ks}

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(args, device):
    if args.stage == 2:
        return load_model_stage2(args, device)

    # ── original Stage-1 load (unchanged) ────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, padding_side="left")
    model = CoLLMStage1(
        tokenizer=tokenizer,
        llm_model_name=args.llm_model,
        llm_dim=args.llm_dim,
        lora_rank=args.lora_rank,
        llm_precision=args.llm_precision,
        attn_implementation=args.attn_implementation,
        gradient_checkpointing=False,
        device=device,
    )
    ckpt_dir = Path(args.checkpoint_dir)
    clip_lora_dir = ckpt_dir / "clip_lora"
    if clip_lora_dir.exists():
        print(f"Loading CLIP LoRA from {clip_lora_dir}")
        model.clip.vision_model.load_adapter(
            str(clip_lora_dir), adapter_name="trained", is_trainable=False
        )
        model.clip.vision_model.set_adapter("trained")
    else:
        print(f"WARNING: No clip_lora dir found at {clip_lora_dir}")
    llm_lora_dir = ckpt_dir / "llm_lora"
    if llm_lora_dir.exists():
        print(f"Loading LLM LoRA from {llm_lora_dir}")
        model.llm.load_adapter(
            str(llm_lora_dir), adapter_name="trained", is_trainable=False
        )
        model.llm.set_adapter("trained")
    else:
        print(f"WARNING: No llm_lora dir found at {llm_lora_dir}")
    extra_pt = ckpt_dir / "extra_modules.pt"
    if extra_pt.exists():
        print(f"Loading adapter/projection from {extra_pt}")
        extra = torch.load(extra_pt, map_location="cpu", weights_only=True)
        model.image_adapter.load_state_dict(extra["image_adapter"])
        model.projection.load_state_dict(extra["projection"])
        model.logit_scale.data = extra["logit_scale"].to(device)
    else:
        print(f"WARNING: No extra_modules.pt found at {extra_pt}")
    model.eval()
    return model

# ─────────────────────────────────────────────────────────────────────────────
# Per-category evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_category(model, category: str, args, device) -> dict:
    caption_dir = Path(args.fiq_caption_dir)
    split_dir   = Path(args.fiq_split_dir)
    img_dir     = args.fiq_image_dir

    anno_path    = caption_dir / f"cap.{category}.val.json"
    gallery_path = split_dir / f"split.{category}.val.json"

    assert anno_path.exists(),    f"Missing annotation: {anno_path}"
    assert gallery_path.exists(), f"Missing gallery split: {gallery_path}"

    gallery_ids = json.load(open(gallery_path))  # list of all image ids in this category

    print(f"\n[{category.upper()}] Queries: {len(json.load(open(anno_path)))}  |  Gallery: {len(gallery_ids)}")

    # ── 1. Encode gallery images ─────────────────────────────────────────────
    print(f"[{category.upper()}] Encoding gallery ({len(gallery_ids)} images)...")
    gallery_dataset = FashionIQGalleryDataset(gallery_ids, img_dir)
    gallery_loader  = DataLoader(gallery_dataset, batch_size=args.batch_size,
                                  num_workers=4, pin_memory=True, shuffle=False)

    all_gallery_feats = []
    all_gallery_ids   = []
    for imgs, ids in gallery_loader:
        imgs = imgs.to(device)
        feats = model.encode_target(imgs)   # [B, 768] — CLIP visual_projection + L2 norm
        all_gallery_feats.append(feats.cpu())
        all_gallery_ids.extend(list(ids))

    gallery_feats   = torch.cat(all_gallery_feats, dim=0)        # [N_gallery, 768]
    gallery_feats   = F.normalize(gallery_feats, dim=-1)
    gallery_ids_arr = np.array(all_gallery_ids)

    # ── 2. Encode queries ────────────────────────────────────────────────────
    print(f"[{category.upper()}] Encoding queries...")
    query_dataset = FashionIQQueryDataset(str(anno_path), img_dir)
    query_loader  = DataLoader(query_dataset, batch_size=args.batch_size,
                                num_workers=4, pin_memory=True, shuffle=False,
                                collate_fn=lambda x: {
                                    "ref_img":   torch.stack([s["ref_img"]   for s in x]),
                                    "caption":   [s["caption"]   for s in x],
                                    "target_id": [s["target_id"] for s in x],
                                    "ref_id":    [s["ref_id"]    for s in x],
                                })

    all_query_feats = []
    all_target_ids  = []
    all_ref_ids     = []

    for batch in query_loader:
        ref_imgs   = batch["ref_img"].to(device)
        captions   = batch["caption"]
        target_ids = batch["target_id"]
        ref_ids    = batch["ref_id"]

        # Step A: get CLIP reference embedding (pooler_output, before visual_projection)
        h_ref = model.encode_reference(ref_imgs)          # [B, 1024]

        # Step B: image adapter → visual token
        adapter_dtype  = next(model.image_adapter.parameters()).dtype
        visual_token   = model.image_adapter(h_ref.to(adapter_dtype))  # [B, 1, 4096]

        # Step C: LLM encode — composed query c = proj(LLM([g(h*); w*]))
        # At inference we use the full composed query (image + text), not c_v or c_w
        llm_device = model.llm_device
        visual_token = visual_token.to(llm_device)
        query_feats  = model.encode_query(
            visual_token=visual_token,
            texts=captions,
            device=llm_device,
        )                                                  # [B, 768] normalized

        all_query_feats.append(query_feats.cpu())
        all_target_ids.extend(target_ids)
        all_ref_ids.extend(ref_ids)

    query_feats = torch.cat(all_query_feats, dim=0)        # [N_queries, 768]
    query_feats = F.normalize(query_feats, dim=-1)

    # ── 3. Compute similarity matrix ─────────────────────────────────────────
    sim_q2t = query_feats @ gallery_feats.t()              # [N_queries, N_gallery]

    # Mask out reference image from retrieval (ref_id == gallery_id → set to -10)
    # Adopted directly from WebCoVR's fashioniq.py
    gallery_id_list = all_gallery_ids
    for i, ref_id in enumerate(all_ref_ids):
        for j, gal_id in enumerate(gallery_id_list):
            if ref_id == gal_id:
                sim_q2t[i][j] = -10.0
                break   # each ref appears once in gallery

    # ── 4. Compute recalls ───────────────────────────────────────────────────
    recalls = get_recalls(sim_q2t, all_target_ids, gallery_ids_arr, ks=[1, 5, 10, 50])
    print(f"[{category.upper()}] Results: {recalls}")
    return recalls

# ─────────────────────────────────────────────────────────────────────────────
# Mean across categories — adopted from WebCoVR's mean_results()
# ─────────────────────────────────────────────────────────────────────────────

def print_mean_results(all_recalls: dict):
    """
    all_recalls: {"dress": {R1:.., R10:.., R50:..}, "shirt": {...}, "toptee": {...}}
    Prints a table identical to the WebCoVR format.
    """
    ks = ["R10", "R50"]

    rows, headers = [], []
    row = []
    for cat in FIQ_CATEGORIES:
        for k in ks:
            v = all_recalls[cat].get(k, 0.0)
            row.append(f"{v:.2f}")
            headers.append(f"{cat}\n{k}")

    # Average across categories per metric
    for k in ks:
        avg = round(np.mean([all_recalls[cat].get(k, 0.0) for cat in FIQ_CATEGORIES]), 2)
        row.append(f"{avg:.2f}")
        headers.append(f"Average\n{k}")

    rows.append(row)
    print("\n" + "="*60)
    print("Fashion-IQ Zero-Shot Results")
    print("="*60)
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    print("  &  ".join(row))   # LaTeX-friendly one-liner

    # Also print R10 average explicitly
    r10_avg = round(np.mean([all_recalls[cat].get("R10", 0.0) for cat in FIQ_CATEGORIES]), 2)
    r50_avg = round(np.mean([all_recalls[cat].get("R50", 0.0) for cat in FIQ_CATEGORIES]), 2)
    print(f"\nFIQ Average R10: {r10_avg:.2f}   |   FIQ Average R50: {r50_avg:.2f}")


def load_model_stage2(args, device):
    """Load a Stage-2 checkpoint for eval.
    Stage-2 checkpoint dir contains:
        llm_lora_stage2/   — rank-16 LLM LoRA
        extra_modules.pt   — image_adapter, projection, logit_scale
    Requires --merged_dir pointing to the merged Stage-1 base weights.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, padding_side="left")

    model = build_stage2_model(
        tokenizer=tokenizer,
        merged_dir=args.merged_dir,
        llm_dim=args.llm_dim,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_rank,   # alpha == rank per paper Sec 9.2
        llm_precision=args.llm_precision,
        attn_implementation=args.attn_implementation,
        gradient_checkpointing=False,
        device=device,
    )

    ckpt_dir = Path(args.checkpoint_dir)

    # Load Stage-2 LLM LoRA
    llm_lora_dir = ckpt_dir / "llm_lora_stage2"
    if llm_lora_dir.exists():
        print(f"Loading Stage-2 LLM LoRA from {llm_lora_dir}")
        model.llm.load_adapter(
            str(llm_lora_dir), adapter_name="trained", is_trainable=False
        )
        model.llm.set_adapter("trained")
    else:
        print(f"WARNING: No llm_lora_stage2 dir at {llm_lora_dir} — using merged base weights only")

    # Load image_adapter / projection / logit_scale
    extra_pt = ckpt_dir / "extra_modules.pt"
    if extra_pt.exists():
        print(f"Loading adapter/projection from {extra_pt}")
        extra = torch.load(extra_pt, map_location="cpu", weights_only=True)
        model.image_adapter.load_state_dict(extra["image_adapter"])
        model.projection.load_state_dict(extra["projection"])
        model.logit_scale.data = extra["logit_scale"].to(device)
    else:
        print(f"WARNING: No extra_modules.pt at {extra_pt}")

    model.eval()
    return model

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Zero-shot FashionIQ evaluation for CoLLM Stage-1")

    # Checkpoint
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to checkpoint dir, e.g. ./checkpoints_stage1_run5/checkpoint_final")

    # FashionIQ data
    parser.add_argument("--fiq_image_dir", type=str, required=True,
                        help="Directory containing all FashionIQ images (*.png or *.jpg)")
    parser.add_argument("--fiq_caption_dir", type=str, required=True,
                        help="Directory containing cap.{cat}.val.json files")
    parser.add_argument("--fiq_split_dir", type=str, required=True,
                        help="Directory containing split.{cat}.val.json files")

    # Model config — must match training config
    parser.add_argument("--llm_model",          type=str,  default="Salesforce/SFR-Embedding-2_R")
    parser.add_argument("--llm_dim",            type=int,  default=4096)
    parser.add_argument("--lora_rank",          type=int,  default=64)
    parser.add_argument("--llm_precision",      type=str,  default="4bit",
                        choices=["auto", "4bit", "bf16"])
    parser.add_argument("--attn_implementation",type=str,  default="eager",
                        choices=["auto", "sdpa", "eager"])
    parser.add_argument("--batch_size",         type=int,  default=64)
    parser.add_argument("--categories",         type=str,  nargs="+",
                        default=FIQ_CATEGORIES,
                        help="Which categories to evaluate (default: all three)")
    parser.add_argument("--output_json",        type=str,  default="fiq_zeroshot_results.json",
                        help="Where to save the recall numbers as JSON")
    # Stage selector
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="1 = Stage-1 checkpoint, 2 = Stage-2 checkpoint")

    # Required for Stage-2 only — merged base weights dir
    parser.add_argument("--merged_dir", type=str, default=None,
                        help="Stage-2 only: path to merged_for_stage2 dir "
                            "(output of merge_stage1.py)")

    args = parser.parse_args()
    if args.stage == 2 and args.merged_dir is None:
        parser.error("--merged_dir is required when --stage 2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    t0 = time.time()
    print("Loading model...")
    model = load_model(args, device)
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # ── Evaluate each category ────────────────────────────────────────────────
    all_recalls = {}
    total_start = time.time()

    for category in args.categories:
        cat_start = time.time()
        recalls   = evaluate_category(model, category, args, device)
        all_recalls[category] = recalls
        elapsed   = str(datetime.timedelta(seconds=int(time.time() - cat_start)))
        print(f"[{category.upper()}] Done in {elapsed}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print_mean_results(all_recalls)

    total_elapsed = str(datetime.timedelta(seconds=int(time.time() - total_start)))
    print(f"\nTotal evaluation time: {total_elapsed}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    import json as _json
    with open(args.output_json, "w") as f:
        _json.dump(all_recalls, f, indent=2)
    print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()