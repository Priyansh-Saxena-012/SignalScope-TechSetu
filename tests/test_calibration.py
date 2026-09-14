"""Tests for SignalScope Post-Hoc Temperature Scaling Calibration (Stage 15).

Validates:
1. Calibration artifact loading from temperature.json
2. Loaded temperature is strictly positive (T > 0)
3. Calibrated probability satisfies calibrated_p = sigmoid(raw_logit / T)
4. Decision threshold invariance: threshold remains strictly 0.50
5. Missing calibration artifact is handled explicitly and safely
6. Production checkpoint file is not modified
7. Calibration does not alter model parameters or raw logits
8. Calibration metrics computation (NLL, Brier, ECE)
9. Strict test-set isolation enforcement
"""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import pytest
import numpy as np
import torch

from model.backbone import build_classifier
from model.calibrate import (
    DEFAULT_TEMPERATURE_PATH,
    apply_temperature_scaling,
    compute_all_calibration_metrics,
    compute_brier_score,
    compute_ece,
    compute_nll,
    fit_temperature,
    load_temperature,
    save_calibration_artifact,
)
from model.predict import clear_model_cache, predict, validate_prediction_schema
from src.data.path_safety import TestSetContaminationError, validate_path_safety


@pytest.fixture
def temp_image(tmp_path):
    """Create a temporary image file for testing."""
    from PIL import Image
    img_path = tmp_path / "test_sample.png"
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def mock_checkpoint(tmp_path):
    """Create a temporary mock checkpoint for integrity tests."""
    ckpt_path = tmp_path / "mock_ckpt.pth"
    model = build_classifier("vit_base_patch16_224", pretrained=False, drop_rate=0.2)
    checkpoint_data = {
        "epoch": 5,
        "backbone": "vit_base_patch16_224",
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint_data, str(ckpt_path))
    return ckpt_path


def test_temperature_json_exists_and_valid():
    """Verify that model/weights/temperature.json exists and conforms to schema."""
    assert DEFAULT_TEMPERATURE_PATH.exists(), f"Missing calibration artifact: {DEFAULT_TEMPERATURE_PATH}"

    with open(DEFAULT_TEMPERATURE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "calibration" in data
    assert "temperature" in data["calibration"]
    assert "decision_threshold" in data["calibration"]
    assert data["calibration"]["decision_threshold"] == 0.50

    temp = float(data["calibration"]["temperature"])
    assert temp > 0.0, f"Temperature must be strictly positive, got {temp}"
    assert 0.1 <= temp <= 10.0, f"Temperature {temp} outside reasonable range [0.1, 10.0]"

    # Verify dataset metadata clearly identifies development validation set
    assert "dataset" in data
    assert "Tiny-GenImage Development Validation Partition" in data["dataset"]["name"]
    assert data["dataset"]["is_held_out_benchmark"] is False
    assert data["dataset"]["total_samples"] == 7000

    # Verify model checkpoint metadata
    assert "model" in data
    assert data["model"]["checkpoint_modified"] is False
    assert data["model"]["parameters_modified"] is False


def test_load_temperature_function():
    """Verify load_temperature correctly parses artifact and reports calibration status."""
    temp, is_calibrated = load_temperature(DEFAULT_TEMPERATURE_PATH)
    assert is_calibrated is True
    assert temp > 0.0

    # Test missing file fallback
    non_existent = Path("non_existent_temperature_path.json")
    fallback_temp, fallback_calibrated = load_temperature(non_existent, allow_uncalibrated_fallback=True)
    assert fallback_temp == 1.0
    assert fallback_calibrated is False

    # Test missing file raises error when fallback disallowed
    with pytest.raises(FileNotFoundError):
        load_temperature(non_existent, allow_uncalibrated_fallback=False)


def test_temperature_scaling_mathematics():
    """Verify calibrated probability equals sigmoid(raw_logit / T) and NOT sigmoid(p / T)."""
    logits = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float64)
    T = 2.0

    scaled_probs = apply_temperature_scaling(logits, T)

    # Manual ground truth: 1 / (1 + exp(-z / T))
    expected = 1.0 / (1.0 + np.exp(-logits / T))
    np.testing.assert_allclose(scaled_probs, expected, rtol=1e-6)

    # For z = 0, probability must remain exactly 0.50 regardless of T
    zero_logit = np.array([0.0])
    for test_t in [0.5, 1.0, 1.959, 5.0]:
        assert abs(apply_temperature_scaling(zero_logit, test_t)[0] - 0.5) < 1e-9


def test_threshold_remains_strictly_half():
    """Verify that operating decision threshold is fixed at exactly 0.50."""
    T = 1.959
    # Negative logit -> p < 0.50 -> Real
    p_neg = apply_temperature_scaling(np.array([-0.01]), T)[0]
    assert p_neg < 0.50

    # Positive logit -> p > 0.50 -> AI
    p_pos = apply_temperature_scaling(np.array([0.01]), T)[0]
    assert p_pos > 0.50

    # Exact 0.0 logit -> p == 0.50 -> threshold boundary
    p_zero = apply_temperature_scaling(np.array([0.0]), T)[0]
    assert abs(p_zero - 0.50) < 1e-9


def test_predict_integration_with_calibration(temp_image, mock_checkpoint, tmp_path):
    """Verify predict() uses temperature scaling and returns standardized schema."""
    clear_model_cache()

    # Create a custom temperature artifact for testing
    test_temp_file = tmp_path / "custom_temp.json"
    with open(test_temp_file, "w") as f:
        json.dump({"calibration": {"temperature": 2.5, "decision_threshold": 0.5}}, f)

    res = predict(
        temp_image,
        allow_stub=False,
        weights_path=str(mock_checkpoint),
        temperature_path=test_temp_file,
    )

    assert validate_prediction_schema(res) is True
    assert res["status"] == "success"
    assert res["is_calibrated"] is True
    assert res["temperature"] == 2.5

    # Verify probability matches sigmoid(raw_logit / 2.5)
    expected_prob = 1.0 / (1.0 + math.exp(-res["raw_logit"] / 2.5))
    assert abs(res["ai_probability"] - expected_prob) < 1e-3

    clear_model_cache()


def test_calibration_does_not_modify_model_parameters(temp_image, mock_checkpoint):
    """Verify that calibration does not alter raw logits or model parameters."""
    clear_model_cache()

    # Run uncalibrated
    res1 = predict(
        temp_image,
        allow_stub=False,
        weights_path=str(mock_checkpoint),
        temperature_path="non_existent_file.json",
        allow_uncalibrated_fallback=True,
    )
    raw_logit_1 = res1["raw_logit"]

    # Run calibrated
    res2 = predict(
        temp_image,
        allow_stub=False,
        weights_path=str(mock_checkpoint),
        temperature_path=DEFAULT_TEMPERATURE_PATH,
    )
    raw_logit_2 = res2["raw_logit"]

    # Raw logit must be completely invariant
    assert raw_logit_1 == raw_logit_2
    clear_model_cache()


def test_production_checkpoint_file_unmodified():
    """Verify that the production checkpoint SHA-256 matches the frozen value."""
    ckpt_path = Path("model/weights/checkpoint_best.pth")
    assert ckpt_path.exists()

    with open(ckpt_path, "rb") as f:
        current_hash = hashlib.sha256(f.read()).hexdigest()

    with open(DEFAULT_TEMPERATURE_PATH, "r") as f:
        temp_data = json.load(f)

    recorded_hash = temp_data["model"]["checkpoint_sha256"]
    assert current_hash == recorded_hash, f"Checkpoint hash mismatch! {current_hash} vs {recorded_hash}"


def test_calibration_metrics_computation():
    """Verify calculation of NLL, Brier score, and ECE with synthetic ground truth."""
    y_true = np.array([0, 0, 1, 1])
    # Perfect probabilities
    p_perfect = np.array([0.01, 0.01, 0.99, 0.99])
    metrics_perfect = compute_all_calibration_metrics(y_true, p_perfect)
    assert metrics_perfect["nll"] < 0.05
    assert metrics_perfect["brier_score"] < 0.01
    assert metrics_perfect["ece"] < 0.05

    # Completely uncalibrated probabilities
    p_poor = np.array([0.9, 0.9, 0.1, 0.1])
    metrics_poor = compute_all_calibration_metrics(y_true, p_poor)
    assert metrics_poor["nll"] > 1.5
    assert metrics_poor["brier_score"] > 0.5


def test_calibration_strictly_isolates_held_out_test_set():
    """Verify that path safety validator rejects any attempt to calibrate on held-out test data."""
    with pytest.raises(TestSetContaminationError):
        validate_path_safety(r"C:\Datasets\SignalScope\test")

    with pytest.raises(TestSetContaminationError):
        validate_path_safety(r"C:\Datasets\SignalScope\test\adm_imagenet\ai")
