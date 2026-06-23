import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


# Deterministic CLIP preprocessing for BOTH reference and target images.
# No random crop / no random flip in Stage-2.
clip_transform = transforms.Compose([
    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


def find_image(image_root: Path, image_rel_path: str) -> Path:
    p = image_root / image_rel_path
    if p.exists():
        return p
    raise FileNotFoundError(f"Could not find image:\n{p}")


class CIRTripletDataset(Dataset):
    def __init__(self, manifest_path: str, image_root: str):
        self.image_root = Path(image_root)
        self.rows = []

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)

                ref_image = row["image"]
                target_image = row["target_image"]
                modifications = row.get("modifications", [])
                if not isinstance(modifications, list):
                    raise TypeError(
                        f"Row {len(self.rows)} has non-list 'modifications': "
                        f"{type(modifications).__name__}"
                    )

                # Each list item describes one part of the full transformation.
                # Keep one triplet per image pair and combine all parts into the
                # complete modification instruction used for training.
                mod_text = " ".join(
                    text.strip()
                    for text in modifications
                    if isinstance(text, str) and text.strip()
                )
                self.rows.append({
                    "image": ref_image,
                    "target_image": target_image,
                    "mod_text": mod_text,
                })

        print(f"Loaded {len(self.rows):,} merged triplets from {manifest_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        ref_path = find_image(self.image_root, row["image"])
        target_path = find_image(self.image_root, row["target_image"])

        ref_img = clip_transform(Image.open(ref_path).convert("RGB"))
        target_img = clip_transform(Image.open(target_path).convert("RGB"))

        return {
            "ref_img": ref_img,
            "target_img": target_img,
            "mod_text": row["mod_text"],
        }


def collate_triplets(batch):
    return {
        "ref_img": torch.stack([b["ref_img"] for b in batch]),
        "target_img": torch.stack([b["target_img"] for b in batch]),
        "mod_text": [b["mod_text"] for b in batch],
    }


def get_triplet_dataloader(
    manifest_path,
    image_root,
    batch_size=32,
    num_workers=4,
    shuffle=True,
):
    dataset = CIRTripletDataset(
        manifest_path=manifest_path,
        image_root=image_root,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_triplets,
    )
