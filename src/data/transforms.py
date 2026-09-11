"""SignalScope Data Augmentation & Transformation Pipeline.

Implements the generalization augmentations required by the approved architecture:
- Random Gaussian Blur (destroys high-frequency pixel fingerprints)
- Random JPEG Compression Simulation (forces model to learn structural rather than compression artifacts)
- Random Resized Crop and Horizontal Flip
- Standard ImageNet Normalization

Validation/Test transforms are strictly deterministic (Resize + CenterCrop).
"""

import io
import random
from typing import Tuple
from PIL import Image, ImageFilter
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomJPEGCompression:
    """Simulate JPEG compression artifacts dynamically in memory.
    
    Parameters
    ----------
    quality_range : tuple of (int, int), default=(40, 95)
        Range of JPEG quality factors to sample from.
    p : float, default=0.5
        Probability of applying JPEG compression.
    """
    def __init__(self, quality_range: Tuple[int, int] = (40, 95), p: float = 0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            quality = random.randint(self.quality_range[0], self.quality_range[1])
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            return Image.open(buffer).convert("RGB")
        return img


class RandomGaussianBlur:
    """Apply Gaussian blur with variable radius to suppress high-frequency noise.
    
    Parameters
    ----------
    radius_range : tuple of (float, float), default=(0.1, 2.0)
        Range of Gaussian blur radius.
    p : float, default=0.5
        Probability of applying blur.
    """
    def __init__(self, radius_range: Tuple[float, float] = (0.1, 2.0), p: float = 0.5):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            radius = random.uniform(self.radius_range[0], self.radius_range[1])
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


def get_train_transforms(image_size: int = 224) -> T.Compose:
    """Construct training transforms with generalization augmentations."""
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        RandomGaussianBlur(radius_range=(0.1, 1.5), p=0.4),
        RandomJPEGCompression(quality_range=(45, 90), p=0.5),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = 224) -> T.Compose:
    """Construct deterministic transforms for validation and test evaluation."""
    # Scale slightly larger then center crop to target dimension
    resize_size = int(image_size * (256.0 / 224.0))
    return T.Compose([
        T.Resize(resize_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
