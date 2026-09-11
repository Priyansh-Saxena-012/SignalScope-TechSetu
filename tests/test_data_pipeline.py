"""Unit and Integration Tests for Stage 2 Data Pipeline.

Tests:
1. Generalization transforms (Gaussian blur, JPEG compression simulation, crop, normalize).
2. Data inspection logic (format detection, class balance, metadata scanning, generator distribution).
3. Leak-free development splitting (generator-aware split, explicit unseen split, stratified proxy split).
4. Robustness to diverse dataset folder structures (CIFAKE, GenImage-style nature/ai hierarchies).
"""

from pathlib import Path
from PIL import Image
import pytest
import torch

from src.data.dataset import SignalScopeDataset, create_development_splits
from src.data.inspect_data import inspect_dataset
from src.data.transforms import (
    RandomGaussianBlur,
    RandomJPEGCompression,
    get_eval_transforms,
    get_train_transforms,
)


@pytest.fixture
def mock_dataset_dir(tmp_path):
    """Create a temporary mock dataset with real and synthetic folders."""
    dataset_root = tmp_path / "mock_dataset"
    real_dir = dataset_root / "real"
    fake_dir = dataset_root / "fake"
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)

    for i in range(20):
        img = Image.new("RGB", (32, 32), color=(i * 10, 50, 100))
        img.save(real_dir / f"real_{i:03d}.jpg")

    for i in range(20):
        img = Image.new("RGB", (32, 32), color=(100, i * 10, 50))
        img.save(fake_dir / f"fake_{i:03d}.png")

    return str(dataset_root)


@pytest.fixture
def mock_generator_dataset_dir(tmp_path):
    """Create a temporary mock dataset with explicit generator tags."""
    dataset_root = tmp_path / "mock_gen_dataset"
    real_dir = dataset_root / "real"
    sd_dir = dataset_root / "fake" / "stable_diffusion"
    mj_dir = dataset_root / "fake" / "midjourney"

    real_dir.mkdir(parents=True)
    sd_dir.mkdir(parents=True)
    mj_dir.mkdir(parents=True)

    for i in range(20):
        img = Image.new("RGB", (32, 32), color=(i * 10, 50, 100))
        img.save(real_dir / f"real_{i:03d}.jpg")

    for i in range(10):
        img = Image.new("RGB", (32, 32), color=(50, i * 10, 150))
        img.save(sd_dir / f"sd_{i:03d}.png")

    for i in range(10):
        img = Image.new("RGB", (32, 32), color=(150, 50, i * 10))
        img.save(mj_dir / f"mj_{i:03d}.png")

    return str(dataset_root)


@pytest.fixture
def mock_genimage_style_dir(tmp_path):
    """Create a temporary mock dataset mimicking GenImage structure (nature vs ai)."""
    dataset_root = tmp_path / "mock_genimage"
    
    # Model 1: wukong
    wk_nature = dataset_root / "wukong" / "train" / "nature"
    wk_ai = dataset_root / "wukong" / "train" / "ai"
    wk_nature.mkdir(parents=True)
    wk_ai.mkdir(parents=True)

    # Model 2: adm
    adm_nature = dataset_root / "adm" / "val" / "nature"
    adm_ai = dataset_root / "adm" / "val" / "ai"
    adm_nature.mkdir(parents=True)
    adm_ai.mkdir(parents=True)

    for i in range(10):
        img = Image.new("RGB", (32, 32), color=(10, 20, 30))
        img.save(wk_nature / f"nat_wk_{i:02d}.JPEG")
        img.save(wk_ai / f"ai_wk_{i:02d}.JPEG")
        img.save(adm_nature / f"nat_adm_{i:02d}.JPEG")
        img.save(adm_ai / f"ai_adm_{i:02d}.JPEG")

    return str(dataset_root)


# --- 1. Transforms Tests ---

def test_random_jpeg_compression():
    """Verify in-memory JPEG compression simulation."""
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    compressor = RandomJPEGCompression(quality_range=(50, 80), p=1.0)
    out_img = compressor(img)

    assert isinstance(out_img, Image.Image)
    assert out_img.size == (64, 64)
    assert out_img.mode == "RGB"


def test_random_gaussian_blur():
    """Verify Gaussian blur simulation."""
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    blurrer = RandomGaussianBlur(radius_range=(0.5, 1.5), p=1.0)
    out_img = blurrer(img)

    assert isinstance(out_img, Image.Image)
    assert out_img.size == (64, 64)


def test_train_transforms_output_shape():
    """Verify train transforms produce normalized (3, 224, 224) tensors."""
    img = Image.new("RGB", (32, 32), color=(100, 100, 100))
    transforms = get_train_transforms(image_size=224)
    tensor = transforms(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_eval_transforms_output_shape():
    """Verify eval transforms produce deterministic (3, 224, 224) tensors."""
    img = Image.new("RGB", (64, 64), color=(50, 50, 50))
    transforms = get_eval_transforms(image_size=224)
    tensor = transforms(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


# --- 2. Inspection Tests ---

def test_inspect_missing_dataset():
    """Verify inspection handles missing directory honestly without crashing."""
    report = inspect_dataset("non_existent_directory_path")
    assert report["status"] == "dataset_not_found"
    assert report["exists"] is False
    assert report["total_images"] == 0


def test_inspect_mock_dataset(mock_dataset_dir):
    """Verify inspection correctly discovers formats, classes, and balance."""
    report = inspect_dataset(mock_dataset_dir)
    assert report["status"] == "ok"
    assert report["exists"] is True
    assert report["total_images"] == 40
    assert report["class_labels"]["real"] == 20
    assert report["class_labels"]["synthetic"] == 20
    assert report["class_balance_ratio"] == 1.0
    assert ".jpg" in report["image_formats"]
    assert ".png" in report["image_formats"]
    assert report["generator_metadata"]["has_generator_labels"] is False


def test_inspect_generator_tags_and_distribution(mock_generator_dataset_dir):
    """Verify inspection detects generator tags and generator distribution counts."""
    report = inspect_dataset(mock_generator_dataset_dir)
    assert report["generator_metadata"]["has_generator_labels"] is True
    assert "stable_diffusion" in report["generator_metadata"]["detected_generators"]
    assert "midjourney" in report["generator_metadata"]["detected_generators"]
    assert report["generator_distribution"]["stable_diffusion"] == 10
    assert report["generator_distribution"]["midjourney"] == 10


def test_inspect_genimage_style_dataset(mock_genimage_style_dir):
    """Verify inspection correctly classifies GenImage nature vs. ai layout."""
    report = inspect_dataset(mock_genimage_style_dir)
    assert report["status"] == "ok"
    assert report["total_images"] == 40
    assert report["class_labels"]["real"] == 20
    assert report["class_labels"]["synthetic"] == 20
    assert "wukong" in report["generator_distribution"]
    assert "adm" in report["generator_distribution"]
    assert report["generator_distribution"]["wukong"] == 10
    assert report["generator_distribution"]["adm"] == 10


# --- 3. Splitting & Dataset Tests ---

def test_create_stratified_proxy_split(mock_dataset_dir):
    """Verify stratified proxy split when no generator metadata is available."""
    splits = create_development_splits(
        mock_dataset_dir,
        train_ratio=0.70,
        val_seen_ratio=0.15,
        val_unseen_ratio=0.15,
        seed=42,
    )

    assert splits["split_strategy"] == "stratified_proxy"
    assert splits["is_unseen_a_proxy"] is True

    total = len(splits["train"]) + len(splits["internal_val_seen"]) + len(splits["internal_val_unseen"])
    assert total == 40

    train_paths = {s[0] for s in splits["train"]}
    val_seen_paths = {s[0] for s in splits["internal_val_seen"]}
    val_unseen_paths = {s[0] for s in splits["internal_val_unseen"]}

    assert len(train_paths.intersection(val_seen_paths)) == 0
    assert len(train_paths.intersection(val_unseen_paths)) == 0
    assert len(val_seen_paths.intersection(val_unseen_paths)) == 0


def test_create_generator_aware_split(mock_generator_dataset_dir):
    """Verify generator-aware splitting when generator labels exist."""
    splits = create_development_splits(
        mock_generator_dataset_dir,
        train_ratio=0.70,
        val_seen_ratio=0.15,
        val_unseen_ratio=0.15,
        seed=42,
    )

    assert splits["split_strategy"] == "generator_aware"

    unseen_fakes = [s for s in splits["internal_val_unseen"] if s[1] == 1]
    assert len(unseen_fakes) > 0

    train_fakes = [s for s in splits["train"] if s[1] == 1]
    unseen_gens = {s[2] for s in unseen_fakes}
    train_gens = {s[2] for s in train_fakes}

    assert len(train_gens.intersection(unseen_gens)) == 0


def test_create_explicit_unseen_generator_split(mock_genimage_style_dir):
    """Verify passing explicit unseen generator allocates that generator exclusively to unseen."""
    splits = create_development_splits(
        mock_genimage_style_dir,
        unseen_generators=["wukong"],
        seed=42,
    )

    assert splits["split_strategy"] == "generator_aware"

    unseen_fakes = [s for s in splits["internal_val_unseen"] if s[1] == 1]
    train_fakes = [s for s in splits["train"] if s[1] == 1]

    for s in unseen_fakes:
        assert s[2] == "wukong"

    for s in train_fakes:
        assert s[2] != "wukong"


def test_signal_scope_dataset_getitem(mock_dataset_dir):
    """Verify PyTorch SignalScopeDataset yields valid batch items."""
    splits = create_development_splits(mock_dataset_dir)
    ds = SignalScopeDataset(splits["train"], transform=get_eval_transforms(224))

    assert len(ds) > 0
    img, label, meta = ds[0]

    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert label in (0, 1)
    assert "path" in meta
    assert "is_ai" in meta
