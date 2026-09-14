"""SignalScope Robustness Evaluation CLI Runner (Stage 12).

Executes the controlled robustness evaluation across the 15 degradation conditions
plus clean baseline on the frozen ViT-Base/16 checkpoint.

Enforces:
- Strict test-set isolation: Rejects any attempt to access C:\\Datasets\\SignalScope\\test.
- Read-only data access: Reads validation Parquet files without modification.
- Deterministic 350-sample stratified cohort (175 Real, 25 each from 7 generators).
- Fixed threshold 0.50 for all binary decisions.
- Machine-readable artifact serialization (JSON, CSV) and SIH-ready markdown report.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image

from src.evaluation.robustness import (
    DEFAULT_VAL_PARQUET_PATH,
    RobustnessCondition,
    SampleRecord,
    TestSetContaminationError,
    validate_path_safety,
    compute_robustness_metrics,
    get_git_commit_hash,
    get_standard_robustness_conditions,
    sample_deterministic_dev_set,
    save_robustness_artifacts,
)

try:
    import torch
    import torch.nn as nn
    from model.backbone import build_classifier, SignalScopeClassifier
    from src.data.transforms import get_eval_transforms
except (ImportError, OSError):
    torch = None
    nn = None
    build_classifier = None
    SignalScopeClassifier = None
    get_eval_transforms = None


def resolve_device(device_str: str = "auto") -> Any:
    """Resolve target execution device."""
    if torch is None:
        return "cpu"
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_classifier(
    checkpoint_path: str,
    device: Any,
) -> Any:
    """Load the frozen SignalScopeClassifier from checkpoint."""
    if torch is None:
        raise RuntimeError("PyTorch is not available in the current environment.")

    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ckpt_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model_cfg = ckpt_config.get("model", {}) if isinstance(ckpt_config, dict) else {}
    backbone_name = checkpoint.get("backbone", model_cfg.get("backbone", "vit_base_patch16_224"))
    drop_rate = model_cfg.get("dropout", 0.2)

    model = build_classifier(
        backbone_name=backbone_name,
        pretrained=False,
        drop_rate=drop_rate,
    )
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_predict_callable(
    model: Any,
    device: Any,
    image_size: int = 224,
) -> Callable[[Image.Image], float]:
    """Create a high-performance single-image inference callable."""
    eval_transform = get_eval_transforms(image_size=image_size)

    def _predict(img: Image.Image) -> float:
        tensor = eval_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            prob = float(torch.sigmoid(logits).item())
        return prob

    return _predict


def run_robustness_experiment(
    checkpoint_path: str = "model/weights/checkpoint_best.pth",
    val_source: str = DEFAULT_VAL_PARQUET_PATH,
    output_dir: str = "experiments/robustness",
    report_path: str = "report/robustness_evaluation.md",
    seed: int = 42,
    device_name: str = "auto",
    mock_inference: bool = False,
) -> Dict[str, Any]:
    """Execute the full Stage 12 robustness evaluation.

    Parameters
    ----------
    checkpoint_path : str
        Path to the frozen model checkpoint.
    val_source : str
        Path/glob to Tiny-GenImage validation Parquet files.
    output_dir : str
        Directory to save robustness_results.json and robustness_summary.csv.
    report_path : str
        Path to save robustness_evaluation.md.
    seed : int
        Deterministic random seed (42).
    device_name : str
        Execution device ('auto', 'cuda', 'cpu').
    mock_inference : bool
        If True, simulates inference for testing when GPU/torch is unavailable.

    Returns
    -------
    dict
        Full structured results dictionary.
    """
    # 1. Enforce strict test-set isolation
    validate_path_safety(val_source, context_desc="validation dataset source")

    # 2. Sample deterministic 350-image cohort
    samples = sample_deterministic_dev_set(
        parquet_source=val_source,
        n_real=175,
        n_ai_per_generator=25,
        seed=seed,
    )

    y_true = np.array([s.label for s in samples], dtype=int)
    generator_tags = [s.generator for s in samples]

    # 3. Setup prediction callable
    if mock_inference:
        # Deterministic mock inference for fast pipeline verification
        def _predict_fn(img: Image.Image) -> float:
            import hashlib
            buf = io.BytesIO()
            img.resize((32, 32)).save(buf, format="PNG")
            h = int(hashlib.sha256(buf.getvalue()).hexdigest()[:8], 16)
            return (h % 1000) / 1000.0
    else:
        target_device = resolve_device(device_name)
        model = load_classifier(checkpoint_path, device=target_device)
        _predict_fn = build_predict_callable(model, device=target_device, image_size=224)

    import time
    t_start = time.perf_counter()
    print(f"Loaded {len(samples)} samples (175 Real, 175 AI across 7 generators).")
    print("Evaluating condition [1/16]: clean (baseline)...")

    # 4. Evaluate Clean Baseline first
    conditions = get_standard_robustness_conditions()
    clean_condition = next(c for c in conditions if c.name == "clean")
    degraded_conditions = [c for c in conditions if c.name != "clean"]

    clean_scores = np.array([_predict_fn(clean_condition.transform_fn(s.image)) for s in samples], dtype=float)

    # Compute baseline metrics
    clean_metrics = compute_robustness_metrics(
        clean_scores=clean_scores,
        degraded_scores=clean_scores,
        y_true=y_true,
        generator_tags=generator_tags,
        threshold=0.50,
    )

    summary_by_condition: Dict[str, Any] = {
        "clean": {
            "name": "clean",
            "category": "baseline",
            "parameter_name": "none",
            "parameter_value": None,
            "description": clean_condition.description,
            **clean_metrics,
        }
    }

    # Store per-sample trajectories
    sample_trajectories: List[Dict[str, Any]] = []
    for i, s in enumerate(samples):
        sample_trajectories.append({
            "sample_id": s.sample_id,
            "path": s.path,
            "label": s.label,
            "generator": s.generator,
            "clean_probability": round(float(clean_scores[i]), 4),
            "clean_prediction": int(clean_scores[i] >= 0.50),
            "degraded_probabilities": {},
        })

    # 5. Evaluate Degraded Conditions
    for idx, cond in enumerate(degraded_conditions):
        print(f"Evaluating condition [{idx+2}/16]: {cond.name}...")
        deg_scores = np.array([_predict_fn(cond.transform_fn(s.image)) for s in samples], dtype=float)

        for i in range(len(samples)):
            sample_trajectories[i]["degraded_probabilities"][cond.name] = round(float(deg_scores[i]), 4)

        metrics = compute_robustness_metrics(
            clean_scores=clean_scores,
            degraded_scores=deg_scores,
            y_true=y_true,
            generator_tags=generator_tags,
            threshold=0.50,
        )

        summary_by_condition[cond.name] = {
            "name": cond.name,
            "category": cond.category,
            "parameter_name": cond.parameter_name,
            "parameter_value": cond.parameter_value,
            "description": cond.description,
            **metrics,
        }

    # 6. Assemble complete results payload
    experiment_results = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_backbone": "vit_base_patch16_224",
            "checkpoint": checkpoint_path,
            "git_commit": get_git_commit_hash(),
            "dataset_source": val_source,
            "held_out_test_dir_isolated": True,
            "total_samples": len(samples),
            "sample_composition": {
                "real": 175,
                "ai_total": 175,
                "generators": {
                    gen: int(np.sum(np.array(generator_tags) == gen))
                    for gen in sorted(list(set(g for g in generator_tags if g != "real")))
                },
            },
            "decision_threshold": 0.50,
            "mock_mode": mock_inference,
            "elapsed_seconds": round(time.perf_counter() - t_start, 2),
            "total_evaluations": len(samples) * len(conditions),
        },
        "conditions": [c.name for c in conditions],
        "summary_by_condition": summary_by_condition,
        "sample_trajectories": sample_trajectories,
    }

    # 7. Save Artifacts
    json_path, csv_path, md_path = save_robustness_artifacts(
        experiment_results,
        output_dir=output_dir,
        report_path=report_path,
    )

    return experiment_results


def main() -> None:
    parser = argparse.ArgumentParser(description="SignalScope Robustness Evaluation CLI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model/weights/checkpoint_best.pth",
        help="Path to frozen ViT model checkpoint",
    )
    parser.add_argument(
        "--val-source",
        type=str,
        default=DEFAULT_VAL_PARQUET_PATH,
        help="Source path/glob for Tiny-GenImage validation Parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/robustness",
        help="Directory to write JSON and CSV summary artifacts",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="report/robustness_evaluation.md",
        help="File path for SIH-ready markdown report",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed random seed for deterministic sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device ('auto', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Execute in mock inference mode for dry-run verification",
    )

    args = parser.parse_args()

    try:
        results = run_robustness_experiment(
            checkpoint_path=args.checkpoint,
            val_source=args.val_source,
            output_dir=args.output_dir,
            report_path=args.report_path,
            seed=args.seed,
            device_name=args.device,
            mock_inference=args.mock,
        )
        print(f"Successfully completed robustness evaluation across {len(results['conditions'])} conditions.")
        print(f"Artifacts saved to {args.output_dir} and {args.report_path}")
    except Exception as exc:
        print(f"Error executing robustness evaluation: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
