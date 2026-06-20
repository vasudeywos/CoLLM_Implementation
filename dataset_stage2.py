import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

# Reference image — augmented (same as Stage-1 aug_images / h'_i path)
ref_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

# Target image — deterministic (same as Stage-1 target_transform / z_i path)
target_transform = transforms.Compose([
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
                self.rows.append(json.loads(line))
        print(f"Loaded {len(self.rows):,} triplets from {manifest_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        ref_path    = find_image(self.image_root, row["image"])
        target_path = find_image(self.image_root, row["target_image"])

        ref_img    = ref_transform(Image.open(ref_path).convert("RGB"))
        target_img = target_transform(Image.open(target_path).convert("RGB"))

        modifications = row["modifications"]
        mod_text = random.choice(modifications) if modifications else ""

        return {
            "ref_img":    ref_img,
            "target_img": target_img,
            "mod_text":   mod_text,
        }


def collate_triplets(batch):
    return {
        "ref_img":    torch.stack([b["ref_img"]    for b in batch]),
        "target_img": torch.stack([b["target_img"] for b in batch]),
        "mod_text":   [b["mod_text"] for b in batch],
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