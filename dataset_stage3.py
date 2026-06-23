"""
Triplet dataset for Stage-3 direct training from base CLIP and SFR models.

Each MTCIR row contains a reference image, target image, and a list of short
modification texts. A modification query is sampled on every __getitem__ call:

  * 50%: concatenate a random proper subset (2..N-1 texts when possible)
  * 30%: concatenate all texts
  * 20%: use one randomly selected text

For rows with too few valid texts, the requested mode falls back to the
closest valid alternative. This keeps every row usable without producing an
empty query.
"""

import json
import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


clip_transform = transforms.Compose(
    [
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ]
)


def find_image(image_root: Path, image_rel_path: str) -> Path:
    path = image_root / image_rel_path
    if path.exists():
        return path
    raise FileNotFoundError(f"Could not find image:\n{path}")


def clean_modifications(modifications) -> list[str]:
    if not isinstance(modifications, list):
        raise TypeError(
            f"'modifications' must be a list, got {type(modifications).__name__}"
        )
    return [
        text.strip()
        for text in modifications
        if isinstance(text, str) and text.strip()
    ]


def sample_modification_text(modifications: list[str]) -> tuple[str, str]:
    """Return (query_text, sampling_mode) using the 50/30/20 policy."""
    count = len(modifications)
    if count == 0:
        raise ValueError("MTCIR row has no non-empty modification text")
    if count == 1:
        return modifications[0], "single"

    draw = random.random()

    if draw < 0.50:
        # A proper subset is possible only when N >= 3. With N == 2, using
        # one item is the nearest non-full alternative.
        if count >= 3:
            subset_size = random.randint(2, count - 1)
            selected_indices = sorted(random.sample(range(count), subset_size))
            selected = [modifications[index] for index in selected_indices]
            return " ".join(selected), "random_subset"
        return random.choice(modifications), "single_fallback"

    if draw < 0.80:
        return " ".join(modifications), "full_concat"

    return random.choice(modifications), "single"


class CIRTripletStage3Dataset(Dataset):
    def __init__(self, manifest_path: str, image_root: str):
        self.image_root = Path(image_root)
        self.rows = []

        with open(manifest_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)
                modifications = clean_modifications(row.get("modifications", []))
                if not modifications:
                    raise ValueError(
                        f"Manifest line {line_number} has no valid modifications"
                    )

                self.rows.append(
                    {
                        "image": row["image"],
                        "target_image": row["target_image"],
                        "modifications": modifications,
                    }
                )

        print(f"Loaded {len(self.rows):,} Stage-3 triplets from {manifest_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        ref_path = find_image(self.image_root, row["image"])
        target_path = find_image(self.image_root, row["target_image"])

        ref_img = clip_transform(Image.open(ref_path).convert("RGB"))
        target_img = clip_transform(Image.open(target_path).convert("RGB"))
        mod_text, sampling_mode = sample_modification_text(row["modifications"])

        return {
            "ref_img": ref_img,
            "target_img": target_img,
            "mod_text": mod_text,
            "sampling_mode": sampling_mode,
        }


def collate_triplets(batch):
    return {
        "ref_img": torch.stack([sample["ref_img"] for sample in batch]),
        "target_img": torch.stack([sample["target_img"] for sample in batch]),
        "mod_text": [sample["mod_text"] for sample in batch],
        "sampling_mode": [sample["sampling_mode"] for sample in batch],
    }


def get_triplet_dataloader(
    manifest_path,
    image_root,
    batch_size=32,
    num_workers=4,
    shuffle=True,
):
    dataset = CIRTripletStage3Dataset(
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
