"""SignalScope Prediction Interface & Contract.

This module provides the standardized prediction interface required by the
SignalScope problem statement (Section 4.1 and Section 7.1).

Supports:
- Live inference using trained ViT-Base/16 checkpoint weights
- In-memory model caching across calls (crucial for Streamlit responsiveness)
- Automatic device selection (CUDA if available, otherwise CPU)
- Standardized schema output conforming to REQUIRED_SCHEMA_KEYS
- Honest fallback stub when weights are not present on disk
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple, Union

# Ensure project root is in sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image
import torch

from model.backbone import build_classifier, SignalScopeClassifier
from model.calibrate import DEFAULT_TEMPERATURE_PATH, load_temperature
from src.data.transforms import get_eval_transforms
from src.provenance.analyzer import analyze_provenance

# Standardized output schema definition
REQUIRED_SCHEMA_KEYS = {"label": str, "confidence": float, "is_ai": bool}

# Default checkpoint search locations relative to repo root
DEFAULT_CHECKPOINT_PATHS = [
    Path("model/weights/checkpoint_best.pth"),
    Path("model/weights/model_weights_fp16.pth"),
    Path("experiments/baseline_vit/checkpoints/checkpoint_best.pth"),
]

# Global cache for the loaded model to prevent reloading on every UI interaction
_CACHED_MODEL: Optional[SignalScopeClassifier] = None
_CACHED_WEIGHTS_PATH: Optional[str] = None
_CACHED_DEVICE: Optional[torch.device] = None


class ModelNotAvailableError(RuntimeError):
    """Raised when inference is requested but trained model weights are not present."""
    pass


def resolve_weights_path(weights_path: Optional[str] = None) -> Optional[Path]:
    """Resolve the path to the trained model checkpoint.

    Checks:
    1. Explicitly provided `weights_path`
    2. SIGNALSCOPE_CHECKPOINT environment variable
    3. Standard candidate paths relative to repository root
    """
    if weights_path is not None:
        p = Path(weights_path)
        return p if p.exists() else None

    env_path = os.environ.get("SIGNALSCOPE_CHECKPOINT")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    repo_root = Path(__file__).resolve().parent.parent
    for rel_path in DEFAULT_CHECKPOINT_PATHS:
        candidate = repo_root / rel_path
        if candidate.exists():
            return candidate

    return None


def get_model(
    weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Optional[SignalScopeClassifier], torch.device, Optional[Path]]:
    """Retrieve or load the cached SignalScopeClassifier.

    Parameters
    ----------
    weights_path : str, optional
        Explicit path to checkpoint file.
    device : torch.device, optional
        Target device. Defaults to CUDA if available, else CPU.

    Returns
    -------
    model : SignalScopeClassifier or None
        Loaded classifier in eval mode, or None if weights not found.
    device : torch.device
        Active execution device.
    resolved_path : Path or None
        Path to loaded weights.
    """
    global _CACHED_MODEL, _CACHED_WEIGHTS_PATH, _CACHED_DEVICE

    target_device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    resolved = resolve_weights_path(weights_path)
    if resolved is None:
        return None, target_device, None

    resolved_str = str(resolved.resolve())
    if (
        _CACHED_MODEL is not None
        and _CACHED_WEIGHTS_PATH == resolved_str
        and _CACHED_DEVICE == target_device
    ):
        return _CACHED_MODEL, target_device, resolved

    # Load checkpoint
    checkpoint = torch.load(resolved, map_location=target_device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        ckpt_config = checkpoint.get("config", {})
        model_cfg = ckpt_config.get("model", {}) if isinstance(ckpt_config, dict) else {}
        backbone_name = checkpoint.get("backbone", model_cfg.get("backbone", "vit_base_patch16_224"))
        drop_rate = model_cfg.get("dropout", 0.2)
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        backbone_name = "vit_base_patch16_224"
        drop_rate = 0.2
    else:
        raise ValueError(f"Invalid checkpoint format at {resolved}")

    model = build_classifier(
        backbone_name=backbone_name,
        pretrained=False,
        drop_rate=drop_rate,
    )
    model.load_state_dict(state_dict)
    model.to(target_device)
    model.eval()

    _CACHED_MODEL = model
    _CACHED_WEIGHTS_PATH = resolved_str
    _CACHED_DEVICE = target_device

    return model, target_device, resolved


def clear_model_cache() -> None:
    """Clear cached model from memory (useful for tests)."""
    global _CACHED_MODEL, _CACHED_WEIGHTS_PATH, _CACHED_DEVICE
    _CACHED_MODEL = None
    _CACHED_WEIGHTS_PATH = None
    _CACHED_DEVICE = None


def predict(
    image_path: str,
    allow_stub: bool = True,
    weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    temperature_path: Optional[Union[str, Path]] = None,
    allow_uncalibrated_fallback: bool = True,
) -> Dict[str, Any]:
    """Run inference on an input image.

    Parameters
    ----------
    image_path : str
        Path to the target image file.
    allow_stub : bool, default=True
        If True, returns the standardized schema stub indicating model unavailability
        when trained weights are not found.
        If False, raises ModelNotAvailableError when weights are not found.
    weights_path : str, optional
        Path to model weights/checkpoint file.
    device : torch.device, optional
        Device to run inference on.
    temperature_path : str or Path, optional
        Explicit path to temperature.json calibration artifact.
        Defaults to model/weights/temperature.json.
    allow_uncalibrated_fallback : bool, default=True
        If True, falls back safely to uncalibrated raw sigmoid (T=1.0) when calibration
        artifact is not found.
        If False, raises FileNotFoundError when calibration artifact is missing.

    Returns
    -------
    dict
        Dictionary conforming to the required prediction contract schema:
        {"label": str, "confidence": float, "is_ai": bool, ...}
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    model, target_device, loaded_weights_path = get_model(weights_path=weights_path, device=device)

    if model is None:
        if allow_stub:
            return {
                "label": "MODEL_NOT_TRAINED",
                "confidence": 0.0,
                "is_ai": False,
                "status": "model_not_yet_trained",
                "message": "Stage 1 Stub: Trained model weights are not yet available. Predictions will be active after Stage 3 training.",
            }
        else:
            raise ModelNotAvailableError(
                "Trained model weights are not yet available in model/weights/. "
                "Model training must be completed before live inference."
            )

    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Failed to load image from {image_path}: {exc}") from exc

    eval_transform = get_eval_transforms(image_size=224)
    tensor = eval_transform(pil_img).unsqueeze(0).to(target_device)

    with torch.no_grad():
        logits = model(tensor)
        raw_logit_val = float(logits.item())

    # Load post-hoc calibration temperature (T > 0)
    temperature, is_calibrated = load_temperature(
        artifact_path=temperature_path,
        allow_uncalibrated_fallback=allow_uncalibrated_fallback,
    )

    # Formal temperature scaling: calibrated_p = sigmoid(raw_logit / T)
    scaled_logit = raw_logit_val / float(temperature)
    # Numerically safe sigmoid calculation
    if scaled_logit >= 0:
        prob = 1.0 / (1.0 + math.exp(-scaled_logit))
    else:
        prob = math.exp(scaled_logit) / (1.0 + math.exp(scaled_logit))

    ai_probability = float(prob)
    # Strict invariance: decision threshold remains 0.50
    is_ai = bool(ai_probability >= 0.5)
    label = "AI-generated" if is_ai else "Real"
    confidence = ai_probability if is_ai else (1.0 - ai_probability)

    # Auxiliary provenance and metadata analysis (never alters model prediction or probability)
    provenance_result = analyze_provenance(image_path)
    metadata_payload = {
        "c2pa_present": provenance_result.get("c2pa_present", False),
        "exif_intact": provenance_result.get("exif_intact", False),
        "provenance": provenance_result.get("provenance", {}),
    }

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "is_ai": is_ai,
        "ai_probability": round(ai_probability, 4),
        "raw_logit": round(raw_logit_val, 4),
        "temperature": round(float(temperature), 4),
        "is_calibrated": is_calibrated,
        "status": "success",
        "generator_family": "Undetermined",
        "explanation": {
            "summary": "ViT-Base/16 baseline classifier prediction with post-hoc temperature calibration. Detailed forensic localization and generator attribution are not yet implemented.",
            "cues": [],
        },
        "metadata": metadata_payload,
        "provenance": provenance_result.get("provenance", {}),
        "device": str(target_device),
        "weights_path": str(loaded_weights_path),
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
        help="Path to the image file to classify.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional path to model checkpoint/weights file.",
    )
    parser.add_argument(
        "--temperature",
        type=str,
        default=None,
        help="Optional path to temperature.json calibration artifact.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Output result formatted as JSON (default: True).",
    )

    args = parser.parse_args()

    try:
        result = predict(
            args.image,
            allow_stub=True,
            weights_path=args.weights,
            temperature_path=args.temperature,
        )
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
