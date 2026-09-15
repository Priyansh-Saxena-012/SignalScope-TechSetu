"""Tests for model/analyze.py — Baseline Analysis Script.

Tests:
1. Checkpoint loading round-trip: save → load → verify model output and metadata.
2. Generator metadata collection: verify evaluate_checkpoint collects all three arrays.
3. End-to-end JSON result generation: mock dataset + mock checkpoint → full analysis → JSON.

All tests use synthetic mock data only. No real datasets or checkpoints are required.
"""

import io
import json
from pathlib import Path

import duckdb
import numpy as np
from PIL import Image
import pytest
import torch

from model.analyze import (
    evaluate_checkpoint,
    load_checkpoint,
    run_baseline_analysis,
)
from model.backbone import build_classifier
from src.utils.config import get_default_config


def _create_test_parquet(path: Path, num_samples: int = 4) -> str:
    """Helper to generate a minimal synthetic Parquet file for analysis tests."""
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE t (image STRUCT(bytes BLOB, path VARCHAR), label BIGINT, generator BIGINT)"
    )
    for i in range(num_samples):
        is_ai = i % 2 == 1
        img = Image.new("RGB", (32, 32), color=(100, 150, 200) if is_ai else (50, 50, 50))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        conn.execute(
            "INSERT INTO t VALUES ({bytes: ?, path: ?}, ?, ?)",
            [buf.getvalue(), f"sample_{i}.jpg", 1 if is_ai else 0, 1 if is_ai else 0],
        )
    f_sql = str(path).replace("\\", "/")
    conn.execute(f"COPY t TO '{f_sql}' (FORMAT PARQUET)")
    conn.close()
    return str(path)


def _save_mock_checkpoint(path: Path, backbone: str = "convnext_tiny") -> str:
    """Create and save a mock checkpoint for testing."""
    model = build_classifier(backbone_name=backbone, pretrained=False, drop_rate=0.2)
    checkpoint_data = {
        "epoch": 3,
        "backbone": backbone,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scheduler_state_dict": None,
        "val_roc_auc": 0.85,
        "config": {
            "model": {
                "backbone": backbone,
                "dropout": 0.2,
                "image_size": 64,
            },
        },
    }
    torch.save(checkpoint_data, path)
    return str(path)


def test_load_checkpoint_roundtrip(tmp_path):
    """Verify checkpoint save → load round-trip: model produces output and metadata is correct."""
    ckpt_path = tmp_path / "mock_checkpoint.pth"
    _save_mock_checkpoint(ckpt_path, backbone="convnext_tiny")

    device = torch.device("cpu")
    model, meta = load_checkpoint(str(ckpt_path), device)

    # Model should be in eval mode
    assert not model.training

    # Model should produce output
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2,)
    assert logits.dtype == torch.float32

    # Metadata should contain checkpoint info
    assert meta["epoch"] == 3
    assert meta["val_roc_auc_at_save"] == 0.85
    assert meta["backbone"] == "convnext_tiny"
    assert "checkpoint_path" in meta


def test_load_checkpoint_missing_file(tmp_path):
    """Verify load_checkpoint raises FileNotFoundError for missing checkpoint."""
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint(str(tmp_path / "nonexistent.pth"), torch.device("cpu"))


def test_evaluate_checkpoint_collects_generator_tags(tmp_path):
    """Verify evaluate_checkpoint collects labels, scores, and generator tags."""
    val_pq = _create_test_parquet(tmp_path / "mock_val.parquet", num_samples=6)

    from src.data.parquet_dataset import TinyGenImageParquetDataset
    from src.data.transforms import get_eval_transforms
    from torch.utils.data import DataLoader

    val_ds = TinyGenImageParquetDataset(
        source=val_pq,
        transform=get_eval_transforms(image_size=64),
    )
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    model = build_classifier("convnext_tiny", pretrained=False)
    model.eval()
    device = torch.device("cpu")

    y_true, y_scores, gen_tags = evaluate_checkpoint(model, val_loader, device, use_amp=False)

    # All three arrays should have the same length as the dataset
    assert len(y_true) == 6
    assert len(y_scores) == 6
    assert len(gen_tags) == 6

    # Labels should be integers
    assert all(isinstance(l, int) for l in y_true)
    assert set(y_true) <= {0, 1}

    # Scores should be probabilities in [0, 1]
    assert all(0.0 <= s <= 1.0 for s in y_scores)

    # Generator tags should be strings from the known generator map
    assert all(isinstance(g, str) for g in gen_tags)
    assert all(g != "" for g in gen_tags)


def test_run_baseline_analysis_produces_json(tmp_path):
    """End-to-end: mock checkpoint + mock Parquet → run_baseline_analysis → JSON output."""
    # Create mock data
    val_pq = _create_test_parquet(tmp_path / "mock_val.parquet", num_samples=8)
    ckpt_path = _save_mock_checkpoint(tmp_path / "checkpoint_best.pth", backbone="convnext_tiny")
    output_json = tmp_path / "analysis_results.json"

    # Create a minimal config
    cfg = get_default_config()
    cfg["data"]["val_data_path"] = val_pq
    cfg["data"]["held_out_test_dir"] = None
    cfg["model"]["backbone"] = "convnext_tiny"
    cfg["model"]["image_size"] = 64
    cfg["training"]["batch_size"] = 4
    cfg["training"]["device"] = "cpu"

    import yaml
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Run analysis
    results = run_baseline_analysis(
        checkpoint_path=ckpt_path,
        config_path=str(config_path),
        output_path=str(output_json),
        device_override="cpu",
    )

    # JSON file should exist
    assert output_json.exists()

    # Load and verify JSON structure
    with open(output_json, "r", encoding="utf-8") as f:
        saved = json.load(f)

    # Required top-level keys
    assert "analysis_timestamp" in saved
    assert "checkpoint" in saved
    assert "dataset" in saved
    assert "overall" in saved
    assert "per_generator" in saved

    # Checkpoint metadata
    assert saved["checkpoint"]["epoch"] == 3
    assert saved["checkpoint"]["backbone"] == "convnext_tiny"

    # Dataset metadata
    assert saved["dataset"]["split"] == "validation"
    assert saved["dataset"]["total_samples"] == 8

    # Overall metrics should contain all required fields
    overall = saved["overall"]
    assert "roc_auc" in overall
    assert "accuracy" in overall
    assert "macro_f1" in overall
    assert "precision" in overall
    assert "recall_tpr" in overall
    assert "false_positive_rate" in overall
    assert "confusion_matrix" in overall
    assert "total_samples" in overall
    assert overall["total_samples"] == 8

    # Confusion matrix structure
    cm = overall["confusion_matrix"]
    assert "tn" in cm and "fp" in cm and "fn" in cm and "tp" in cm
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == 8

    # Per-generator should be a dict (may be empty for mock data depending on labels)
    assert isinstance(saved["per_generator"], dict)

    # Return value should match saved JSON
    assert results["dataset"]["total_samples"] == 8
    assert results["overall"]["total_samples"] == 8
