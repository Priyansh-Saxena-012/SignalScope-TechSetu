"""Unit tests for dataset-independent Model, Evaluation, and Configuration modules.

Tests:
1. Swappable vision backbone interface (ViT-Base/16 and ConvNeXt-Tiny).
2. Core classification metrics (ROC-AUC, Macro-F1, FPR, Confusion Matrix).
3. Generator-sliced evaluation and unseen-split aggregation.
4. Configuration loading, validation, and serialization.
5. Trainer scaffolding initialization and device resolution.

CRITICAL: All tests use synthetic arrays and random in-memory tensors only.
Zero real dataset assumptions are made.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.model.backbone import SUPPORTED_BACKBONES, SignalScopeClassifier, build_classifier
from src.model.train import SignalScopeTrainer, resolve_device, set_seed
from src.evaluation.evaluate import (
    compute_metrics,
    evaluate_with_generator_breakdown,
    format_evaluation_report,
)
from src.utils.config import get_default_config, load_config, save_config, validate_config


# --- 1. Model Architecture Tests ---

def test_vit_base_forward_pass():
    """Verify ViT-Base/16 forward pass with dummy tensor outputs shape (B,)."""
    model = build_classifier("vit_base_patch16_224", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2,)
    assert logits.dtype == torch.float32


def test_convnext_tiny_forward_pass():
    """Verify ConvNeXt-Tiny forward pass outputs shape (B,)."""
    model = build_classifier("convnext_tiny", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2,)


def test_swappable_backbone_consistency():
    """Verify ViT and ConvNeXt expose identical predict_probabilities interface."""
    x = torch.randn(3, 3, 224, 224)

    for backbone in ["vit_base_patch16_224", "convnext_tiny"]:
        model = build_classifier(backbone, pretrained=False)
        probs = model.predict_probabilities(x)

        assert probs.shape == (3,)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

        summary = model.get_parameter_summary()
        assert summary["total_parameters"] > 0
        assert summary["estimated_fp16_mb"] > 0


def test_unsupported_backbone_raises():
    """Verify unsupported backbone name raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported backbone"):
        build_classifier("non_existent_vision_backbone_xyz")


# --- 2. Evaluation Utilities Tests ---

def test_compute_metrics_accuracy_and_auc():
    """Verify standard metric calculations on synthetic known distributions."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    metrics = compute_metrics(y_true, y_scores, threshold=0.5)

    assert metrics["roc_auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["confusion_matrix"]["tn"] == 4
    assert metrics["confusion_matrix"]["tp"] == 4
    assert metrics["confusion_matrix"]["fp"] == 0
    assert metrics["confusion_matrix"]["fn"] == 0


def test_compute_metrics_fpr_and_threshold():
    """Verify false positive rate calculation at different decision thresholds."""
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.6, 0.2, 0.8, 0.4])  # Sample 0 is FP at threshold=0.5

    metrics = compute_metrics(y_true, y_scores, threshold=0.5)
    # TN=1, FP=1 -> FPR = 1 / (1 + 1) = 0.5
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["confusion_matrix"]["fp"] == 1
    assert metrics["confusion_matrix"]["tn"] == 1


def test_evaluate_with_generator_breakdown():
    """Verify evaluation slices metrics by generator and aggregates unseen split."""
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_scores = [0.1, 0.2, 0.3, 0.4, 0.8, 0.9, 0.7, 0.6]
    gen_tags = ["real", "real", "real", "real", "midjourney", "midjourney", "wukong", "wukong"]

    res = evaluate_with_generator_breakdown(
        y_true, y_scores, gen_tags, unseen_generators=["midjourney"], threshold=0.5
    )

    assert "overall" in res
    assert "per_generator" in res
    assert "midjourney" in res["per_generator"]
    assert "wukong" in res["per_generator"]

    # Unseen generator slice should contain midjourney
    assert res["unseen_generator_split"] is not None
    assert res["unseen_generator_split"]["roc_auc"] is not None

    report_str = format_evaluation_report(res)
    assert "SIGNALSCOPE EVALUATION REPORT" in report_str
    assert "Unseen Generators" in report_str


# --- 3. Configuration Management Tests ---

def test_load_and_validate_config():
    """Verify train_config.yaml loads and passes schema validation."""
    cfg = load_config("configs/train_config.yaml")

    assert validate_config(cfg) is True
    assert cfg["model"]["backbone"] == "vit_base_patch16_224"
    assert cfg["model"]["fallback_backbone"] == "convnext_tiny"
    assert "training" in cfg
    assert "data" in cfg


def test_save_and_reload_config(tmp_path):
    """Verify saving and re-loading configuration preserves fields."""
    cfg = get_default_config()
    cfg["experiment"]["name"] = "test_custom_run"

    out_file = tmp_path / "custom_config.yaml"
    save_config(cfg, str(out_file))

    loaded = load_config(str(out_file))
    assert loaded["experiment"]["name"] == "test_custom_run"


# --- 4. Trainer Scaffolding Tests ---

def test_device_and_seed_resolution():
    """Verify device selection falls back cleanly to CPU and seed is respected."""
    set_seed(123)
    dev, info = resolve_device("cpu")

    assert dev.type == "cpu"
    assert info["device"] == "cpu"


def test_trainer_scaffold_initialization(tmp_path):
    """Verify trainer initializes optimizer, model, and directories without running training."""
    cfg = get_default_config()
    cfg["experiment"]["output_dir"] = str(tmp_path / "test_run")
    cfg["model"]["pretrained"] = False
    cfg["model"]["backbone"] = "convnext_tiny"  # Lightweight for test
    cfg["training"]["device"] = "cpu"

    trainer = SignalScopeTrainer(cfg)

    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert (tmp_path / "test_run" / "checkpoints").exists()
    assert (tmp_path / "test_run" / "config.yaml").exists()
