r"""Unit tests for Tiny-GenImage Parquet Dataset Loader.

Tests:
1. Dataset construction, lazy indexing, and sample retrieval using mock Parquet files.
2. Image decoding to PIL and transformation pipeline to PyTorch Tensors.
3. Label handling: binary labels (0 for Real, 1 for Synthetic).
4. Generator metadata preservation: human-readable name, numeric ID, and is_ai flag.
5. Independent training and validation dataset construction and separation.
6. Test-set isolation: explicit rejection if pointed at or inside held-out test benchmark.
7. Integration with real Tiny-GenImage Parquet files (28,000 train, 7,000 val) when available.
8. Error handling: out-of-range indexing and non-existent sources.
"""

import io
import os
from pathlib import Path
import sys
import duckdb
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from src.data.parquet_dataset import (
    DEFAULT_TINY_GENIMAGE_TRAIN_SOURCE,
    DEFAULT_TINY_GENIMAGE_VAL_SOURCE,
    TINY_GENIMAGE_GENERATOR_MAP,
    TINY_GENIMAGE_LABEL_MAP,
    TinyGenImageParquetDataset,
    create_tiny_genimage_datasets,
    resolve_parquet_files,
)
from src.data.path_safety import TestSetContaminationError


@pytest.fixture
def mock_parquet_dataset(tmp_path):
    """Create two mock Parquet files (train and val) with valid image BLOBs."""
    data_dir = tmp_path / "mock_parquet_data"
    data_dir.mkdir(parents=True)

    def _create_mock_file(file_path: Path, num_samples: int, is_train: bool):
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t (image STRUCT(bytes BLOB, path VARCHAR), label BIGINT, generator BIGINT)")

        for i in range(num_samples):
            # Alternating Real (0) and Synthetic (1) with diverse generators
            is_ai = i % 2 == 1
            label = 1 if is_ai else 0
            generator = (i % 4) + 1 if is_ai else 0

            # Generate valid in-memory image
            color = (200, i * 10 % 255, 50) if is_ai else (50, 100, i * 10 % 255)
            img = Image.new("RGB", (32, 32), color=color)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            img_bytes = buf.getvalue()
            img_name = f"{'ai' if is_ai else 'real'}_{i:03d}.jpg"

            conn.execute("INSERT INTO t VALUES ({bytes: ?, path: ?}, ?, ?)", [img_bytes, img_name, label, generator])

        f_sql = str(file_path).replace("\\", "/")
        conn.execute(f"COPY t TO '{f_sql}' (FORMAT PARQUET)")
        conn.close()

    train_file = data_dir / "train-00000.parquet"
    val_file = data_dir / "val-00000.parquet"

    _create_mock_file(train_file, num_samples=10, is_train=True)
    _create_mock_file(val_file, num_samples=6, is_train=False)

    return {
        "train_file": str(train_file),
        "val_file": str(val_file),
        "data_dir": str(data_dir),
    }


# --- 1. Mock Dataset Construction & Retrieval Tests ---

def test_mock_dataset_construction_and_len(mock_parquet_dataset):
    """Verify dataset resolves files and reports accurate row count."""
    train_ds = TinyGenImageParquetDataset(mock_parquet_dataset["train_file"])
    val_ds = TinyGenImageParquetDataset(mock_parquet_dataset["val_file"])

    assert len(train_ds) == 10
    assert len(val_ds) == 6
    assert len(train_ds.files) == 1
    assert len(val_ds.files) == 1


def test_mock_dataset_getitem_types_and_metadata(mock_parquet_dataset):
    """Verify sample retrieval decodes PIL Image, correct label, and preserved metadata."""
    ds = TinyGenImageParquetDataset(mock_parquet_dataset["train_file"])

    # Sample 0 (Real)
    img0, lbl0, meta0 = ds[0]
    assert isinstance(img0, Image.Image)
    assert img0.size == (32, 32)
    assert lbl0 == 0
    assert meta0["label_name"] == "real"
    assert meta0["is_ai"] is False
    assert meta0["generator"] == "real"
    assert meta0["generator_id"] == 0
    assert "real_000.jpg" in meta0["path"]

    # Sample 1 (Synthetic)
    img1, lbl1, meta1 = ds[1]
    assert isinstance(img1, Image.Image)
    assert lbl1 == 1
    assert meta0["label_name"] in TINY_GENIMAGE_LABEL_MAP.values()
    assert meta1["label_name"] == "fake"
    assert meta1["is_ai"] is True
    assert meta1["generator"] in TINY_GENIMAGE_GENERATOR_MAP.values()
    assert meta1["generator_id"] > 0
    assert "ai_001.jpg" in meta1["path"]


def test_mock_dataset_with_transforms_and_dataloader(mock_parquet_dataset):
    """Verify dataset works with torchvision transforms and multi-worker DataLoader batching."""
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = TinyGenImageParquetDataset(mock_parquet_dataset["train_file"], transform=transform)

    img_tensor, label, meta = ds[0]
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 64, 64)
    assert isinstance(label, int)

    # Test with multiprocessing DataLoader (num_workers=2) and shuffle=True
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)
    batch_images, batch_labels, batch_meta = next(iter(loader))

    assert batch_images.shape == (4, 3, 64, 64)
    assert batch_labels.shape == (4,)
    assert len(batch_meta["path"]) == 4
    assert len(batch_meta["generator"]) == 4
    assert len(batch_meta["label_name"]) == 4


def test_mock_dataset_summary(mock_parquet_dataset):
    """Verify get_summary computes accurate distributions."""
    ds = TinyGenImageParquetDataset(mock_parquet_dataset["train_file"])
    summary = ds.get_summary()

    assert summary["total_samples"] == 10
    assert summary["real_samples"] == 5
    assert summary["synthetic_samples"] == 5
    assert "real" in summary["generators_breakdown"]
    assert summary["generators_breakdown"]["real"] == 5


# --- 2. Isolation & Error Handling Tests ---

def test_test_set_isolation_rejection(tmp_path):
    """Verify dataset construction rejects paths pointing to held-out test set."""
    mock_held_out = tmp_path / "held_out_test"
    mock_held_out.mkdir(parents=True)
    bad_file = mock_held_out / "test.parquet"

    with pytest.raises(TestSetContaminationError):
        TinyGenImageParquetDataset(str(bad_file), held_out_test_dir=str(mock_held_out))


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="Requires Windows filesystem path semantics (drive letters and backslashes)",
)
def test_test_set_isolation_rejection_default_windows_path():
    """Verify official default Windows test path rejection."""
    with pytest.raises(TestSetContaminationError):
        TinyGenImageParquetDataset(r"C:\Datasets\SignalScope\test\data.parquet")


def test_index_out_of_bounds(mock_parquet_dataset):
    """Verify IndexError is raised for out-of-range indices."""
    ds = TinyGenImageParquetDataset(mock_parquet_dataset["train_file"])
    with pytest.raises(IndexError):
        _ = ds[-1]
    with pytest.raises(IndexError):
        _ = ds[len(ds)]


def test_nonexistent_source():
    """Verify FileNotFoundError when source does not match any Parquet files."""
    with pytest.raises(FileNotFoundError):
        TinyGenImageParquetDataset("C:/nonexistent_path/does_not_exist_*.parquet")


# --- 3. Real Tiny-GenImage Integration Tests (When Data Available) ---

@pytest.mark.skipif(
    not os.path.exists(r"C:\Datasets\SignalScope_Train\data"),
    reason="Tiny-GenImage dataset not mounted at C:\\Datasets\\SignalScope_Train\\data",
)
def test_real_tiny_genimage_train_dataset():
    """Verify full training Parquet dataset: 14 files, 28,000 samples, 1:1 balance."""
    train_ds = TinyGenImageParquetDataset(DEFAULT_TINY_GENIMAGE_TRAIN_SOURCE)

    assert len(train_ds) == 28000
    assert len(train_ds.files) == 14

    # Verify first and last samples
    img0, lbl0, meta0 = train_ds[0]
    assert isinstance(img0, Image.Image)
    assert lbl0 in (0, 1)
    assert "generator" in meta0
    assert "path" in meta0

    img_last, lbl_last, meta_last = train_ds[27999]
    assert isinstance(img_last, Image.Image)
    assert lbl_last in (0, 1)

    # Summary check
    summary = train_ds.get_summary()
    assert summary["total_samples"] == 28000
    assert summary["real_samples"] == 14000
    assert summary["synthetic_samples"] == 14000


@pytest.mark.skipif(
    not os.path.exists(r"C:\Datasets\SignalScope_Train\data"),
    reason="Tiny-GenImage dataset not mounted at C:\\Datasets\\SignalScope_Train\\data",
)
def test_real_tiny_genimage_val_dataset():
    """Verify full validation Parquet dataset: 4 files, 7,000 samples, 1:1 balance."""
    val_ds = TinyGenImageParquetDataset(DEFAULT_TINY_GENIMAGE_VAL_SOURCE)

    assert len(val_ds) == 7000
    assert len(val_ds.files) == 4

    summary = val_ds.get_summary()
    assert summary["total_samples"] == 7000
    assert summary["real_samples"] == 3500
    assert summary["synthetic_samples"] == 3500


@pytest.mark.skipif(
    not os.path.exists(r"C:\Datasets\SignalScope_Train\data"),
    reason="Tiny-GenImage dataset not mounted at C:\\Datasets\\SignalScope_Train\\data",
)
def test_real_tiny_genimage_factory_separation():
    """Verify create_tiny_genimage_datasets returns independent train and validation instances."""
    train_ds, val_ds = create_tiny_genimage_datasets()

    assert len(train_ds) == 28000
    assert len(val_ds) == 7000
    assert train_ds is not val_ds

    # Check that file lists do not overlap
    train_file_set = set(train_ds.files)
    val_file_set = set(val_ds.files)
    assert len(train_file_set.intersection(val_file_set)) == 0
