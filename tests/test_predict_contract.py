"""Tests for SignalScope Prediction Interface Contract.

Validates that src/model/predict.py strictly satisfies the required JSON schema,
data types, error handling, and honest indication of untrained status in Stage 1.
"""

import json
import os
import subprocess
import sys
from PIL import Image
import pytest

from src.model.predict import (
    REQUIRED_SCHEMA_KEYS,
    ModelNotAvailableError,
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
    result = predict(temp_image, allow_stub=True)

    assert isinstance(result, dict)
    assert "label" in result
    assert "confidence" in result
    assert "is_ai" in result

    assert isinstance(result["label"], str)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["is_ai"], bool)

    assert validate_prediction_schema(result) is True


def test_stub_honestly_indicates_model_unavailable(temp_image):
    """Verify that the Stage 1 stub does not fake predictions."""
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


def test_cli_predict_contract(temp_image):
    """Verify that src/model/predict.py can be invoked via CLI and outputs valid JSON."""
    cmd = [
        sys.executable,
        os.path.join("src", "model", "predict.py"),
        "--image",
        temp_image,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 0, f"CLI invocation failed with error: {proc.stderr}"

    output_data = json.loads(proc.stdout)
    assert isinstance(output_data, dict)
    assert validate_prediction_schema(output_data) is True
    assert output_data["label"] == "MODEL_NOT_TRAINED"
