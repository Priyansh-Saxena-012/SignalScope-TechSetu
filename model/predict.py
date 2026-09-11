"""SignalScope Prediction Interface & Contract.

This module provides the standardized prediction interface required by the
SignalScope problem statement (Section 4.1 and Section 7.1).

Stage 1 Status: STUB IMPLEMENTATION.
Trained model weights are not yet available. Predictions are disabled to
prevent fake or misleading results until model training is completed in Stage 3.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

# Standardized output schema definition
REQUIRED_SCHEMA_KEYS = {"label": str, "confidence": float, "is_ai": bool}


class ModelNotAvailableError(RuntimeError):
    """Raised when inference is requested but trained model weights are not present."""
    pass


def predict(image_path: str, allow_stub: bool = True) -> Dict[str, Any]:
    """Run inference on an input image.

    Parameters
    ----------
    image_path : str
        Path to the target image file.
    allow_stub : bool, default=True
        If True, returns the standardized schema stub indicating model unavailability.
        If False, raises ModelNotAvailableError.

    Returns
    -------
    dict
        Dictionary conforming to the required prediction contract schema:
        {"label": str, "confidence": float, "is_ai": bool, ...}
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Stage 1 Verification Stub:
    # Trained weights are not yet available. We do NOT fake predictions.
    if allow_stub:
        return {
            "label": "MODEL_NOT_TRAINED",
            "confidence": 0.0,
            "is_ai": False,
            "status": "model_not_yet_trained",
            "message": "Stage 1 Stub: Trained model weights are not yet available. Predictions will be active after Stage 3 training."
        }
    else:
        raise ModelNotAvailableError(
            "Trained model weights are not yet available in model/weights/. "
            "Model training must be completed in Stage 3 before live inference."
        )


def validate_prediction_schema(output: Dict[str, Any]) -> bool:
    """Validate that an output dictionary strictly adheres to the required prediction schema."""
    for key, expected_type in REQUIRED_SCHEMA_KEYS.items():
        if key not in output:
            return False
        if not isinstance(output[key], expected_type):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SignalScope Real vs. AI-Generated Image Prediction Interface"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image file to classify."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Output result formatted as JSON (default: True)."
    )

    args = parser.parse_args()

    try:
        result = predict(args.image, allow_stub=True)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
