"""SignalScope PyTorch Dataset and Development Splitting Engine.

Implements:
1. SignalScopeDataset: PyTorch Dataset for loading images, binary labels, and metadata.
2. create_development_splits(): Partitions dataset into:
   - train: ~70%
   - internal_val_seen: ~15%
   - internal_val_unseen: ~15% (Strictly a development proxy)
   Methodology is generator-aware if generator labels exist; otherwise, uses
   a defensible stratified leak-free split with explicit documentation of limitations.
3. Configurable unseen generator selection without hardcoded dataset assumptions.
4. Strict isolation: The official held-out test set is completely excluded.
"""

from __future__ import annotations

import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data.inspect_data import (
    VALID_IMAGE_EXTENSIONS,
    detect_class_from_path,
    detect_generator_from_path,
    inspect_dataset,
)


class SignalScopeDataset(Dataset):
    """PyTorch Dataset for SignalScope Real vs. AI image classification.

    Parameters
    ----------
    samples : list of tuple (path: str, label: int, generator: Optional[str])
        List of image records, where label=0 (Real) and label=1 (Synthetic/AI).
    transform : callable, optional
        PyTorch torchvision transform pipeline.
    """
    def __init__(
        self,
        samples: List[Tuple[str, int, Optional[str]]],
        transform: Optional[Callable] = None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        path, label, generator = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        meta = {
            "path": path,
            "generator": generator if generator else "unknown",
            "is_ai": bool(label == 1),
        }
        return image, label, meta


def create_development_splits(
    data_dir: str,
    train_ratio: float = 0.70,
    val_seen_ratio: float = 0.15,
    val_unseen_ratio: float = 0.15,
    seed: int = 42,
    unseen_generators: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create reproducible, leak-free development splits.

    Parameters
    ----------
    data_dir : str
        Path to the development dataset directory.
        (MUST NOT point to the official held-out test set).
    train_ratio : float, default=0.70
    val_seen_ratio : float, default=0.15
    val_unseen_ratio : float, default=0.15
    seed : int, default=42
    unseen_generators : list of str, optional
        Explicit generator names to reserve exclusively for internal_val_unseen.
        If None, automatically reserves one generator family when metadata permits.

    Returns
    -------
    dict
        Dictionary containing:
        - "train": list of samples
        - "internal_val_seen": list of samples
        - "internal_val_unseen": list of samples (development proxy)
        - "split_strategy": "generator_aware" or "stratified_proxy"
        - "split_summary": detailed counts, generator distributions, and ratios
    """
    inspection = inspect_dataset(data_dir)
    if inspection.get("status") == "dataset_not_found":
        raise FileNotFoundError(f"Cannot split: Dataset directory not found at {data_dir}")

    root_path = Path(data_dir)
    real_samples: List[Tuple[str, int, Optional[str]]] = []
    fake_samples: List[Tuple[str, int, Optional[str]]] = []

    for p in root_path.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue

        cls = detect_class_from_path(p)
        gen = detect_generator_from_path(p, root_path)

        if cls == "real":
            real_samples.append((str(p), 0, "real"))
        elif cls == "synthetic":
            fake_samples.append((str(p), 1, gen))

    if not real_samples or not fake_samples:
        raise ValueError(
            f"Dataset must contain both real and fake images. Found {len(real_samples)} real and {len(fake_samples)} fake."
        )

    # Set seed for reproducible splitting
    rng = random.Random(seed)
    rng.shuffle(real_samples)
    rng.shuffle(fake_samples)

    # Check whether reliable generator metadata exists for generator-aware splitting
    generators = {g for _, _, g in fake_samples if g is not None}
    has_reliable_generator_tags = len(generators) >= 2

    train: List[Tuple[str, int, Optional[str]]] = []
    internal_val_seen: List[Tuple[str, int, Optional[str]]] = []
    internal_val_unseen: List[Tuple[str, int, Optional[str]]] = []
    split_strategy = "stratified_proxy"

    # Split Real samples proportionally across all three splits
    n_real = len(real_samples)
    n_real_train = int(n_real * train_ratio)
    n_real_val_seen = int(n_real * val_seen_ratio)

    real_train = real_samples[:n_real_train]
    real_val_seen = real_samples[n_real_train : n_real_train + n_real_val_seen]
    real_val_unseen = real_samples[n_real_train + n_real_val_seen :]

    if has_reliable_generator_tags or (unseen_generators and any(g in generators for g in unseen_generators)):
        split_strategy = "generator_aware"

        if unseen_generators:
            target_unseen_gens = set(unseen_generators)
        else:
            gen_list = sorted(list(generators))
            target_unseen_gens = {gen_list[-1]}

        seen_fakes = [s for s in fake_samples if s[2] not in target_unseen_gens]
        unseen_fakes = [s for s in fake_samples if s[2] in target_unseen_gens]

        n_seen = len(seen_fakes)
        n_seen_train = int(n_seen * (train_ratio / (train_ratio + val_seen_ratio))) if (train_ratio + val_seen_ratio) > 0 else 0

        train = real_train + seen_fakes[:n_seen_train]
        internal_val_seen = real_val_seen + seen_fakes[n_seen_train:]
        internal_val_unseen = real_val_unseen + unseen_fakes
    else:
        # Defensible stratified leak-free split
        # internal_val_unseen is explicitly documented as a stratified development proxy
        n_fake = len(fake_samples)
        n_fake_train = int(n_fake * train_ratio)
        n_fake_val_seen = int(n_fake * val_seen_ratio)

        fake_train = fake_samples[:n_fake_train]
        fake_val_seen = fake_samples[n_fake_train : n_fake_train + n_fake_val_seen]
        fake_val_unseen = fake_samples[n_fake_train + n_fake_val_seen :]

        train = real_train + fake_train
        internal_val_seen = real_val_seen + fake_val_seen
        internal_val_unseen = real_val_unseen + fake_val_unseen

    # Shuffle each partition so classes and generators are mixed
    rng.shuffle(train)
    rng.shuffle(internal_val_seen)
    rng.shuffle(internal_val_unseen)

    return {
        "train": train,
        "internal_val_seen": internal_val_seen,
        "internal_val_unseen": internal_val_unseen,
        "split_strategy": split_strategy,
        "is_unseen_a_proxy": True,
        "split_summary": {
            "train_count": len(train),
            "train_real": sum(1 for s in train if s[1] == 0),
            "train_fake": sum(1 for s in train if s[1] == 1),
            "train_generators": dict(Counter(s[2] for s in train if s[1] == 1)),
            "val_seen_count": len(internal_val_seen),
            "val_seen_real": sum(1 for s in internal_val_seen if s[1] == 0),
            "val_seen_fake": sum(1 for s in internal_val_seen if s[1] == 1),
            "val_seen_generators": dict(Counter(s[2] for s in internal_val_seen if s[1] == 1)),
            "val_unseen_count": len(internal_val_unseen),
            "val_unseen_real": sum(1 for s in internal_val_unseen if s[1] == 0),
            "val_unseen_fake": sum(1 for s in internal_val_unseen if s[1] == 1),
            "val_unseen_generators": dict(Counter(s[2] for s in internal_val_unseen if s[1] == 1)),
        },
    }
