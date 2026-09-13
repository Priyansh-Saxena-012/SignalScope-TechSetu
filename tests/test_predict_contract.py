"""Tests for SignalScope Prediction Interface Contract.

Validates that model/predict.py strictly satisfies the required JSON schema,
data types, error handling, and honest indication of untrained status in Stage 1,
as well as live prediction, model caching, and UI integration when weights are present.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from PIL import Image
import pytest
import torch

from model.backbone import build_classifier
from model.predict import (
    REQUIRED_SCHEMA_KEYS,
    ModelNotAvailableError,
    clear_model_cache,
    get_model,
    predict,
    validate_prediction_schema,
)


@pytest.fixture
def temp_image(tmp_path):
    """Create a temporary image file for contract verification."""
    img_path = tmp_path / "sample_test.png"
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def mock_vit_checkpoint_path(tmp_path):
    """Create a minimal ViT-Base/16 mock checkpoint file for testing live prediction."""
    ckpt_path = tmp_path / "mock_vit_checkpoint.pth"
    model = build_classifier("vit_base_patch16_224", pretrained=False, drop_rate=0.2)
    checkpoint_data = {
        "epoch": 1,
        "backbone": "vit_base_patch16_224",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scheduler_state_dict": None,
        "val_roc_auc": 0.91,
        "config": {
            "model": {"backbone": "vit_base_patch16_224", "dropout": 0.2},
            "training": {"mixed_precision": False},
        },
    }
    torch.save(checkpoint_data, str(ckpt_path))
    return str(ckpt_path)


def test_schema_definition():
    """Verify that required schema contains the exact keys and types specified in PS."""
    assert "label" in REQUIRED_SCHEMA_KEYS
    assert "confidence" in REQUIRED_SCHEMA_KEYS
    assert "is_ai" in REQUIRED_SCHEMA_KEYS
    assert REQUIRED_SCHEMA_KEYS["label"] is str
    assert REQUIRED_SCHEMA_KEYS["confidence"] is float
    assert REQUIRED_SCHEMA_KEYS["is_ai"] is bool


def test_predict_returns_valid_schema(temp_image):
    """Verify that predict() output matches the required schema and data types."""
    clear_model_cache()
    result = predict(temp_image, allow_stub=True)

    assert isinstance(result, dict)
    assert "label" in result
    assert "confidence" in result
    assert "is_ai" in result

    assert isinstance(result["label"], str)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["is_ai"], bool)

    assert validate_prediction_schema(result) is True


def test_stub_honestly_indicates_model_unavailable(temp_image, monkeypatch):
    """Verify that the Stage 1 stub does not fake predictions when weights are unavailable."""
    clear_model_cache()
    monkeypatch.setattr("model.predict.DEFAULT_CHECKPOINT_PATHS", [])
    monkeypatch.delenv("SIGNALSCOPE_CHECKPOINT", raising=False)
    result = predict(temp_image, allow_stub=True)

    # Must indicate that actual model is not trained yet
    assert result["label"] == "MODEL_NOT_TRAINED"
    assert result.get("status") == "model_not_yet_trained"
    assert "not yet available" in result.get("message", "").lower()

    # When stub mode is disabled, it must raise ModelNotAvailableError
    with pytest.raises(ModelNotAvailableError):
        predict(temp_image, allow_stub=False)


def test_predict_raises_on_missing_file():
    """Verify that predict() raises FileNotFoundError on non-existent input."""
    with pytest.raises(FileNotFoundError):
        predict("non_existent_image_file_path.jpg", allow_stub=True)


def test_cli_predict_contract(temp_image, tmp_path, mock_vit_checkpoint_path):
    """Verify that model/predict.py can be invoked via CLI and outputs valid JSON."""
    # 1. Verify stub JSON output when weights are unavailable
    nonexistent_weights = str(tmp_path / "nonexistent_weights.pth")
    cmd_stub = [
        sys.executable,
        os.path.join("model", "predict.py"),
        "--image",
        temp_image,
        "--weights",
        nonexistent_weights,
    ]
    proc = subprocess.run(cmd_stub, capture_output=True, text=True)

    assert proc.returncode == 0, f"CLI invocation failed with error: {proc.stderr}"

    output_data = json.loads(proc.stdout)
    assert isinstance(output_data, dict)
    assert validate_prediction_schema(output_data) is True
    assert output_data["label"] == "MODEL_NOT_TRAINED"

    # 2. Verify live JSON output when trained checkpoint is provided
    cmd_live = [
        sys.executable,
        os.path.join("model", "predict.py"),
        "--image",
        temp_image,
        "--weights",
        mock_vit_checkpoint_path,
    ]
    proc_live = subprocess.run(cmd_live, capture_output=True, text=True)
    assert proc_live.returncode == 0, f"CLI live invocation failed with error: {proc_live.stderr}"

    live_data = json.loads(proc_live.stdout)
    assert isinstance(live_data, dict)
    assert validate_prediction_schema(live_data) is True
    assert live_data["status"] == "success"
    assert live_data["label"] in ("AI-generated", "Real")


def test_predict_with_weights_contract(temp_image, mock_vit_checkpoint_path):
    """Verify that predict() with real weights performs inference according to contract."""
    clear_model_cache()
    result = predict(temp_image, allow_stub=False, weights_path=mock_vit_checkpoint_path)

    assert validate_prediction_schema(result) is True
    assert result["status"] == "success"
    assert result["label"] in ("AI-generated", "Real")
    assert 0.0 <= result["ai_probability"] <= 1.0
    assert 0.5 <= result["confidence"] <= 1.0

    if result["is_ai"]:
        assert result["label"] == "AI-generated"
        assert result["confidence"] == result["ai_probability"]
    else:
        assert result["label"] == "Real"
        assert result["confidence"] == round(1.0 - result["ai_probability"], 4)

    # Integrity rules: no fabricated forensic evidence
    assert result["generator_family"] == "Undetermined"
    assert result["explanation"]["cues"] == []
    assert "baseline classifier prediction" in result["explanation"]["summary"].lower()
    clear_model_cache()


def test_model_caching(temp_image, mock_vit_checkpoint_path):
    """Verify that get_model caches the model instance across invocations."""
    clear_model_cache()
    model1, _, _ = get_model(weights_path=mock_vit_checkpoint_path)
    model2, _, _ = get_model(weights_path=mock_vit_checkpoint_path)
    assert model1 is not None
    assert model1 is model2

    # Predict call reuses cached model
    res = predict(temp_image, weights_path=mock_vit_checkpoint_path)
    assert res["status"] == "success"
    model3, _, _ = get_model(weights_path=mock_vit_checkpoint_path)
    assert model1 is model3

    clear_model_cache()


def test_ui_predict_mock_and_live(temp_image, mock_vit_checkpoint_path, monkeypatch):
    """Verify UI prediction bridge works in both demo fallback and live checkpoint modes."""
    import src.ui.app as ui_app

    test_img = Image.open(temp_image)

    # 1. When weights are absent: UI should run in demo mode
    clear_model_cache()
    monkeypatch.setattr("model.predict.DEFAULT_CHECKPOINT_PATHS", [])
    monkeypatch.delenv("SIGNALSCOPE_CHECKPOINT", raising=False)
    demo_result = ui_app.predict(test_img)
    assert demo_result["is_demo_mode"] is True
    assert demo_result["verdict"] in (
        "Likely AI-generated",
        "Likely Real",
        "Inconclusive / Low Confidence",
    )

    # 2. When checkpoint is set: UI should run in live mode
    monkeypatch.setenv("SIGNALSCOPE_CHECKPOINT", mock_vit_checkpoint_path)
    clear_model_cache()
    live_result = ui_app.predict(test_img)
    assert live_result["is_demo_mode"] is False
    assert live_result["generator_family"] == "Undetermined"
    assert live_result["explanation"]["cues"] == []
    assert live_result["verdict"] in (
        "Likely AI-generated",
        "Likely Real",
        "Inconclusive / Low Confidence",
    )
    clear_model_cache()
