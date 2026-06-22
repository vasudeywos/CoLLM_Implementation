import torch
import webdataset as wds
import torchvision.transforms as transforms
from torch.utils.data import default_collate

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

# Augmentation for h'_i (Paper uses augmentations for Slerp diversity)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

# Deterministic transform for target z_i
target_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

def preprocess(sample):
    image, caption = sample
    aug_image = train_transform(image)
    target_image = target_transform(image)
    
    # Handle caption bytes/str
    if isinstance(caption, bytes):
        caption = caption.decode("utf-8")
        
    return {
        "aug_images": aug_image,
        "target_images": target_image,
        "captions": caption
    }

def get_cc3m_dataloader(tar_path_pattern, batch_size=64, num_workers=4):
    dataset = (
        wds.WebDataset(tar_path_pattern,
                        resampled=True,
                        shardshuffle=100)
        .shuffle(1000)
        .decode("pil")
        .to_tuple("jpg", "txt")
        .map(preprocess)
        .batched(batch_size, partial=False, collation_fn=default_collate)
    )
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=num_workers)
    return loader