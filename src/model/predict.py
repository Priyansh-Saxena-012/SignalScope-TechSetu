"""SignalScope Prediction Interface & Contract.

This module provides the standardized prediction interface required by the
SignalScope problem statement (Section 4.1 and Section 7.1).

Stage 3 Status: LIVE INFERENCE.
Loads the trained checkpoint exported to ``src/model/weights/`` (see
``model_card.json`` alongside it for backbone/threshold/eval metrics) and runs
real predictions. Falls back to the honest Stage-1 stub (or raises
``ModelNotAvailableError``) if no exported weights are found.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image

# Ensure project root is in sys.path when executed directly (python src/model/predict.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms import get_eval_transforms
from src.model.backbone import SignalScopeClassifier, build_classifier

# Standardized output schema definition
REQUIRED_SCHEMA_KEYS = {"label": str, "confidence": float, "is_ai": bool}

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DEFAULT_WEIGHTS_PATH = WEIGHTS_DIR / "model_weights_fp16.pth"
MODEL_CARD_PATH = WEIGHTS_DIR / "model_card.json"


class ModelNotAvailableError(RuntimeError):
    """Raised when inference is requested but trained model weights are not present."""
    pass


@functools.lru_cache(maxsize=1)
def _load_model() -> Tuple[SignalScopeClassifier, int, Dict[str, Any]]:
    """Load and cache the trained classifier, image size, and model card metadata."""
    card: Dict[str, Any] = {}
    if MODEL_CARD_PATH.exists():
        card = json.loads(MODEL_CARD_PATH.read_text(encoding="utf-8"))

    backbone_name = card.get("backbone", "vit_base_patch16_224")
    image_size = card.get("image_size", 224)

    model = build_classifier(backbone_name=backbone_name, pretrained=False)
    state_dict = torch.load(DEFAULT_WEIGHTS_PATH, map_location="cpu")
    # Exported weights are fp16 for compact distribution; cast back to fp32 for inference.
    state_dict = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, image_size, card


def predict(image_path: str, allow_stub: bool = True) -> Dict[str, Any]:
    """Run inference on an input image.

    Parameters
    ----------
    image_path : str
        Path to the target image file.
    allow_stub : bool, default=True
        Only consulted when no trained weights are exported yet: if True, returns
        the standardized schema stub indicating model unavailability; if False,
        raises ModelNotAvailableError. Ignored once trained weights are present.

    Returns
    -------
    dict
        Dictionary conforming to the required prediction contract schema:
        {"label": str, "confidence": float, "is_ai": bool, ...}
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if not DEFAULT_WEIGHTS_PATH.exists():
        # Stage 1 Verification Stub: trained weights are not yet available.
        # We do NOT fake predictions.
        if allow_stub:
            return {
                "label": "MODEL_NOT_TRAINED",
                "confidence": 0.0,
                "is_ai": False,
                "status": "model_not_yet_trained",
                "message": "Stage 1 Stub: Trained model weights are not yet available. Predictions will be active after Stage 3 training."
            }
        raise ModelNotAvailableError(
            "Trained model weights are not yet available in src/model/weights/. "
            "Model training must be completed in Stage 3 before live inference."
        )

    model, image_size, card = _load_model()
    threshold = float(card.get("threshold", 0.5))

    image = Image.open(image_path).convert("RGB")
    tensor = get_eval_transforms(image_size)(image).unsqueeze(0)

    with torch.no_grad():
        probability_ai = float(model.predict_probabilities(tensor).item())

    is_ai = probability_ai >= threshold
    return {
        "label": "AI-GENERATED" if is_ai else "REAL",
        "confidence": probability_ai,
        "is_ai": bool(is_ai),
        "status": "ok",
        "model": {
            "backbone": card.get("backbone", "vit_base_patch16_224"),
            "val_roc_auc": card.get("val_roc_auc"),
            "test_roc_auc": card.get("test_roc_auc"),
        },
    }


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
