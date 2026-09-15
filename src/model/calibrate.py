"""SignalScope Post-Hoc Temperature Scaling Calibration Engine.

Provides mathematically rigorous post-hoc temperature scaling calibration
for binary classification:
- Temperature scaling parameter optimization via NLL minimization
- Calibration error metrics: Negative Log-Likelihood (NLL), Brier score,
  and Expected Calibration Error (ECE) with equal-width binning
- Artifact serialization to model/weights/temperature.json
- Model evaluation and calibration fitting on the 7,000-image development
  validation dataset
- Strict data-safety enforcement: Rejects any access to the official held-out test set
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy.optimize import minimize_scalar

from src.data.path_safety import TestSetContaminationError, validate_path_safety

try:
    import torch
    import torch.nn as nn
    from PIL import Image
    from model.backbone import build_classifier, SignalScopeClassifier
    from src.data.transforms import get_eval_transforms
except (ImportError, OSError):
    torch = None
    nn = None
    Image = None
    build_classifier = None
    SignalScopeClassifier = None
    get_eval_transforms = None


DEFAULT_TEMPERATURE_PATH = REPO_ROOT / "model" / "weights" / "temperature.json"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "model" / "weights" / "checkpoint_best.pth"
DEFAULT_DEV_VAL_PARQUET_GLOB = r"C:\Datasets\SignalScope_Train\data\validation-*.parquet"


# ---------------------------------------------------------------------------
# Calibration Metrics Computation
# ---------------------------------------------------------------------------

def compute_nll(y_true: Union[np.ndarray, Sequence[int]], probs: Union[np.ndarray, Sequence[float]], eps: float = 1e-12) -> float:
    """Compute binary Negative Log-Likelihood (cross-entropy).

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Binary ground-truth labels (0 or 1).
    probs : array-like of shape (N,)
        Predicted probabilities for class 1.
    eps : float, default=1e-12
        Epsilon clip to prevent log(0).

    Returns
    -------
    float
        Average Negative Log-Likelihood.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    nll = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(nll)


def compute_brier_score(y_true: Union[np.ndarray, Sequence[int]], probs: Union[np.ndarray, Sequence[float]]) -> float:
    """Compute Brier score (mean squared error between probabilities and binary labels).

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Binary ground-truth labels (0 or 1).
    probs : array-like of shape (N,)
        Predicted probabilities for class 1.

    Returns
    -------
    float
        Brier score in [0.0, 1.0]. Lower is better.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    return float(np.mean((p - y) ** 2))


def compute_ece(
    y_true: Union[np.ndarray, Sequence[int]],
    probs: Union[np.ndarray, Sequence[float]],
    n_bins: int = 15,
) -> Dict[str, Any]:
    """Compute Expected Calibration Error (ECE) using equal-width probability binning.

    For binary classification, confidence is defined as the predicted probability
    for the predicted class: conf = max(p, 1 - p), and accuracy is whether the
    argmax prediction matches y_true. ECE is the weighted average absolute difference:
        ECE = sum_{b=1}^B (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Binary ground-truth labels (0 or 1).
    probs : array-like of shape (N,)
        Predicted probabilities for class 1 (AI).
    n_bins : int, default=15
        Number of equal-width bins in [0.5, 1.0].

    Returns
    -------
    dict
        Dictionary containing:
        - "ece": float, weighted ECE
        - "max_calibration_error": float (MCE)
        - "n_bins": int
        - "bins": list of bin diagnostic dictionaries
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probs, dtype=float)
    n_samples = len(y)
    if n_samples == 0:
        return {"ece": 0.0, "max_calibration_error": 0.0, "n_bins": n_bins, "bins": []}

    # Binary classification confidence and correctness
    pred_labels = (p >= 0.5).astype(int)
    confidences = np.maximum(p, 1.0 - p)
    accuracies = (pred_labels == y).astype(float)

    bin_boundaries = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_details = []

    for i in range(n_bins):
        low = bin_boundaries[i]
        high = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences >= low) & (confidences < high)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_err = abs(bin_acc - bin_conf)
            ece += (bin_count / n_samples) * bin_err
            mce = max(mce, bin_err)

            bin_details.append({
                "bin_index": i,
                "range": [round(float(low), 4), round(float(high), 4)],
                "count": bin_count,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
                "abs_error": round(bin_err, 4),
            })

    return {
        "ece": float(ece),
        "max_calibration_error": float(mce),
        "n_bins": n_bins,
        "bins": bin_details,
    }


def compute_all_calibration_metrics(
    y_true: Union[np.ndarray, Sequence[int]],
    probs: Union[np.ndarray, Sequence[float]],
    n_bins: int = 15,
) -> Dict[str, Any]:
    """Compute NLL, Brier score, and ECE in a consolidated dictionary."""
    nll = compute_nll(y_true, probs)
    brier = compute_brier_score(y_true, probs)
    ece_dict = compute_ece(y_true, probs, n_bins=n_bins)

    return {
        "nll": round(nll, 6),
        "brier_score": round(brier, 6),
        "ece": round(ece_dict["ece"], 6),
        "mce": round(ece_dict["max_calibration_error"], 6),
        "n_bins": n_bins,
    }


# ---------------------------------------------------------------------------
# Temperature Fitting Objective & Solver
# ---------------------------------------------------------------------------

def apply_temperature_scaling(logits: Union[np.ndarray, Sequence[float]], temperature: float) -> np.ndarray:
    """Apply temperature scaling to raw scalar logits and return calibrated probabilities.

    Parameters
    ----------
    logits : array-like of shape (N,)
        Raw scalar logits z.
    temperature : float
        Scalar temperature parameter T > 0.

    Returns
    -------
    np.ndarray
        Calibrated probabilities p = sigmoid(z / T).
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be strictly positive (> 0), got {temperature}")
    z = np.asarray(logits, dtype=np.float64)
    scaled_z = z / float(temperature)
    scaled_z = np.clip(scaled_z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-scaled_z))


def fit_temperature(
    logits: Union[np.ndarray, Sequence[float]],
    y_true: Union[np.ndarray, Sequence[int]],
    t_min: float = 0.01,
    t_max: float = 10.0,
) -> float:
    """Find optimal temperature T > 0 that minimizes Negative Log-Likelihood (NLL) on validation logits.

    Parameters
    ----------
    logits : array-like of shape (N,)
        Raw scalar logits from the frozen classifier on the validation set.
    y_true : array-like of shape (N,)
        Binary ground-truth labels (0=Real, 1=AI).
    t_min : float, default=0.01
        Lower bound for scalar optimization.
    t_max : float, default=10.0
        Upper bound for scalar optimization.

    Returns
    -------
    float
        Optimal scalar temperature T.
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)

    def nll_objective(temp: float) -> float:
        scaled_z = np.clip(z / temp, -60.0, 60.0)
        p = 1.0 / (1.0 + np.exp(-scaled_z))
        p = np.clip(p, 1e-15, 1.0 - 1e-15)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    res = minimize_scalar(nll_objective, bounds=(t_min, t_max), method="bounded")
    if not res.success:
        raise RuntimeError(f"Temperature optimization failed to converge: {res.message}")

    optimal_t = float(res.x)
    return optimal_t


# ---------------------------------------------------------------------------
# Calibration Artifact I/O
# ---------------------------------------------------------------------------

def save_calibration_artifact(
    temperature: float,
    metrics_before: Dict[str, Any],
    metrics_after: Dict[str, Any],
    checkpoint_path: Path,
    output_path: Path = DEFAULT_TEMPERATURE_PATH,
    sample_count: int = 7000,
) -> Path:
    """Save calibrated temperature and verification metadata to JSON artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_hash = None
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            ckpt_hash = hashlib.sha256(f.read()).hexdigest()

    payload = {
        "calibration": {
            "temperature": round(float(temperature), 6),
            "fitting_method": "temperature_scaling_nll_minimization",
            "objective": "negative_log_likelihood",
            "decision_threshold": 0.50,
            "formula": "calibrated_probability = sigmoid(raw_logit / temperature)",
        },
        "dataset": {
            "name": "Tiny-GenImage Development Validation Partition",
            "type": "development_validation_split",
            "is_held_out_benchmark": False,
            "total_samples": sample_count,
            "class_distribution": {
                "real": sample_count // 2,
                "ai_generated": sample_count // 2,
            },
        },
        "model": {
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_sha256": ckpt_hash,
            "architecture": "vit_base_patch16_224",
            "parameters_modified": False,
            "checkpoint_modified": False,
        },
        "metrics": {
            "before_calibration": metrics_before,
            "after_calibration": metrics_after,
            "improvements": {
                "nll_delta": round(metrics_after["nll"] - metrics_before["nll"], 6),
                "brier_delta": round(metrics_after["brier_score"] - metrics_before["brier_score"], 6),
                "ece_delta": round(metrics_after["ece"] - metrics_before["ece"], 6),
            },
        },
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version.split()[0],
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return output_path


def load_temperature(
    artifact_path: Optional[Union[str, Path]] = None,
    allow_uncalibrated_fallback: bool = True,
) -> Tuple[float, bool]:
    """Load the calibrated temperature parameter from artifact file.

    Parameters
    ----------
    artifact_path : str or Path, optional
        Path to temperature.json. Defaults to model/weights/temperature.json.
    allow_uncalibrated_fallback : bool, default=True
        If True, returns (1.0, False) when artifact is missing, indicating uncalibrated
        identity temperature scaling.
        If False, raises FileNotFoundError.

    Returns
    -------
    temperature : float
        Positive scalar temperature T.
    is_calibrated : bool
        True if temperature was loaded from valid calibration artifact;
        False if uncalibrated fallback T=1.0 is active.
    """
    path = Path(artifact_path) if artifact_path is not None else DEFAULT_TEMPERATURE_PATH

    if not path.exists():
        if allow_uncalibrated_fallback:
            return 1.0, False
        raise FileNotFoundError(
            f"Calibration artifact not found at {path}. "
            "Run model/calibrate.py on the development validation set to produce temperature.json."
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        temp = float(data.get("calibration", {}).get("temperature", data.get("temperature", 1.0)))
        if temp <= 0:
            raise ValueError(f"Invalid non-positive temperature in artifact {path}: {temp}")
        return temp, True
    except Exception as exc:
        if allow_uncalibrated_fallback:
            return 1.0, False
        raise RuntimeError(f"Failed to parse calibration artifact at {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Validation Set Logit Extraction & Calibration Runner
# ---------------------------------------------------------------------------

def run_calibration(
    parquet_glob: str = DEFAULT_DEV_VAL_PARQUET_GLOB,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    output_artifact_path: Path = DEFAULT_TEMPERATURE_PATH,
    batch_size: int = 32,
    device_str: str = "auto",
) -> Dict[str, Any]:
    """Execute full calibration fitting pipeline on the 7,000-image development validation set."""
    validate_path_safety(parquet_glob, context_desc="calibration validation source")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        ckpt_hash_before = hashlib.sha256(f.read()).hexdigest()

    if torch is None:
        raise RuntimeError("PyTorch is required for model inference during calibration.")

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"Loading frozen model from {checkpoint_path.name} on {device}...")
    from model.predict import get_model
    model, _, _ = get_model(weights_path=str(checkpoint_path), device=device)
    if model is None:
        raise RuntimeError(f"Failed to load model from {checkpoint_path}")

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    import duckdb
    print(f"Reading development validation records from {parquet_glob}...")
    conn = duckdb.connect()
    f_sql = parquet_glob.replace("\\", "/")
    query = f"""
        SELECT (image).bytes, (image).path, label, generator
        FROM read_parquet('{f_sql}')
        ORDER BY (image).path
    """
    rows = conn.execute(query).fetchall()
    total_samples = len(rows)
    print(f"Loaded {total_samples} validation records.")

    eval_transform = get_eval_transforms(image_size=224)
    raw_logits: List[float] = []
    labels: List[int] = []

    print("Running frozen ViT-Base/16 inference across validation cohort...")
    t_start = time.time()
    n_batches = math.ceil(total_samples / batch_size)

    for b_idx in range(n_batches):
        batch_rows = rows[b_idx * batch_size : (b_idx + 1) * batch_size]
        batch_tensors = []
        for img_bytes, p_str, lbl, gen in batch_rows:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            batch_tensors.append(eval_transform(pil_img))
            labels.append(int(lbl))

        batch_tensor = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            out_logits = model(batch_tensor)
            if out_logits.ndim == 0:
                raw_logits.append(float(out_logits.item()))
            else:
                raw_logits.extend(out_logits.cpu().tolist())

        if (b_idx + 1) % 25 == 0 or (b_idx + 1) == n_batches:
            elapsed = time.time() - t_start
            processed = len(raw_logits)
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"  Batch {b_idx + 1}/{n_batches} | {processed}/{total_samples} samples | {rate:.1f} img/s")

    raw_logits_np = np.asarray(raw_logits, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=int)

    uncalibrated_probs = 1.0 / (1.0 + np.exp(-raw_logits_np))
    metrics_before = compute_all_calibration_metrics(labels_np, uncalibrated_probs)

    print("Fitting scalar temperature T via NLL minimization...")
    optimal_t = fit_temperature(raw_logits_np, labels_np)
    print(f"Fitted Temperature T: {optimal_t:.4f}")

    calibrated_probs = apply_temperature_scaling(raw_logits_np, optimal_t)
    metrics_after = compute_all_calibration_metrics(labels_np, calibrated_probs)

    with open(checkpoint_path, "rb") as f:
        ckpt_hash_after = hashlib.sha256(f.read()).hexdigest()
    if ckpt_hash_before != ckpt_hash_after:
        raise RuntimeError("[FATAL INTEGRITY ERROR] Model checkpoint was modified during calibration!")

    saved_path = save_calibration_artifact(
        temperature=optimal_t,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        checkpoint_path=checkpoint_path,
        output_path=output_artifact_path,
        sample_count=total_samples,
    )
    print(f"Saved calibration artifact to {saved_path}")

    return {
        "temperature": optimal_t,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "sample_count": total_samples,
        "artifact_path": str(saved_path),
        "checkpoint_sha256": ckpt_hash_after,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SignalScope Post-Hoc Temperature Scaling Calibration"
    )
    parser.add_argument(
        "--validation-parquet",
        type=str,
        default=DEFAULT_DEV_VAL_PARQUET_GLOB,
        help="Glob to validation parquet files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to frozen production checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_TEMPERATURE_PATH),
        help="Path to save temperature.json.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device ('auto', 'cuda', 'cpu').",
    )

    args = parser.parse_args()
    results = run_calibration(
        parquet_glob=args.validation_parquet,
        checkpoint_path=Path(args.checkpoint),
        output_artifact_path=Path(args.output),
        batch_size=args.batch_size,
        device_str=args.device,
    )

    print("\n" + "=" * 65)
    print("CALIBRATION COMPLETE")
    print("=" * 65)
    print(f"Optimal Temperature T : {results['temperature']:.4f}")
    print(f"Validation Samples    : {results['sample_count']}")
    print("-" * 65)
    print(f"NLL (Before -> After) : {results['metrics_before']['nll']:.4f} -> {results['metrics_after']['nll']:.4f}")
    print(f"Brier (Before -> After): {results['metrics_before']['brier_score']:.4f} -> {results['metrics_after']['brier_score']:.4f}")
    print(f"ECE (Before -> After) : {results['metrics_before']['ece']:.4f} -> {results['metrics_after']['ece']:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
