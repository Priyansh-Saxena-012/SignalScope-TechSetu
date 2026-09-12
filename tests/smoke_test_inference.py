"""SignalScope 10-Image Pipeline Smoke Test.

Verifies end-to-end functionality:
Image Loading -> Preprocessing/Transforms -> Model Forward Pass -> Probability Generation -> Metrics Evaluation.

CRITICAL CONSTRAINTS:
1. Sourced from 10 explicit images in the held-out test dataset (5 synthetic, 5 real).
2. NO crawling of the held-out test directory.
3. STRICTLY an inference/pipeline sanity check.
4. MUST NOT be used for training, validation, model selection, or tuning.
5. Handles the absence of trained SignalScope weights appropriately.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from model.backbone import build_classifier
from src.data.dataset import SignalScopeDataset
from src.data.transforms import get_eval_transforms
from src.evaluation.evaluate import (
    compute_metrics,
    evaluate_with_generator_breakdown,
    format_evaluation_report,
)

# 10 Explicit held-out benchmark image paths (5 AI/synthetic, 5 real/nature)
DEFAULT_SMOKE_TEST_IMAGES = [
    # 5 Synthetic / AI Images (label=1) across 5 distinct generator families
    {
        "path": r"C:\Datasets\SignalScope\test\midjourney_imagenet\ai\0_midjourney_100.png",
        "label": 1,
        "generator": "midjourney",
    },
    {
        "path": r"C:\Datasets\SignalScope\test\adm_imagenet\ai\0_adm_153.PNG",
        "label": 1,
        "generator": "adm",
    },
    {
        "path": r"C:\Datasets\SignalScope\test\biggan_imagenet\ai\000_biggan_00020.png",
        "label": 1,
        "generator": "biggan",
    },
    {
        "path": r"C:\Datasets\SignalScope\test\sdv4_imagenet\ai\000_sdv4_00020.png",
        "label": 1,
        "generator": "sdv4",
    },
    {
        "path": r"C:\Datasets\SignalScope\test\wukong_imagenet\ai\0_wukong_image128.png",
        "label": 1,
        "generator": "wukong",
    },
    # 5 Real / Nature Images (label=0) paired from corresponding sets
    {
        "path": r"C:\Datasets\SignalScope\test\midjourney_imagenet\nature\ILSVRC2012_val_00000008.JPEG",
        "label": 0,
        "generator": "real",
    },
    {
        "path": r"C:\Datasets\SignalScope\test\adm_imagenet\nature\ILSVRC2012_val_00000005.JPEG",
        "label": 0,
        "generator": "real",
    },
    {
        "path": r"C:\Datasets\SignalScope\test\biggan_imagenet\nature\ILSVRC2012_val_00000009.JPEG",
        "label": 0,
        "generator": "real",
    },
    {
        "path": r"C:\Datasets\SignalScope\test\sdv4_imagenet\nature\ILSVRC2012_val_00000037.JPEG",
        "label": 0,
        "generator": "real",
    },
    {
        "path": r"C:\Datasets\SignalScope\test\wukong_imagenet\nature\ILSVRC2012_val_00000002.JPEG",
        "label": 0,
        "generator": "real",
    },
]


def run_smoke_test(
    image_entries: Optional[List[Dict[str, Any]]] = None,
    backbone: str = "vit_base_patch16_224",
    pretrained: bool = False,
    threshold: float = 0.5,
    batch_size: int = 2,
) -> Dict[str, Any]:
    """Execute end-to-end 10-image pipeline smoke test.

    Parameters
    ----------
    image_entries : list of dict, optional
        List of image records containing 'path', 'label', and 'generator'.
    backbone : str, default="vit_base_patch16_224"
    pretrained : bool, default=False
        Whether to instantiate with pretrained weights.
    threshold : float, default=0.5
    batch_size : int, default=2

    Returns
    -------
    dict
        Smoke test results dictionary including predictions and evaluation metrics.
    """
    entries = image_entries or DEFAULT_SMOKE_TEST_IMAGES

    # 1. Verify existence of all 10 images before doing anything
    missing = [entry["path"] for entry in entries if not os.path.exists(entry["path"])]
    if missing:
        raise FileNotFoundError(
            f"[SMOKE TEST ERROR] The following expected smoke test image(s) do not exist:\n"
            + "\n".join(f"  - {p}" for p in missing)
        )

    # 2. Prepare sample tuples for SignalScopeDataset
    samples = [(entry["path"], int(entry["label"]), entry.get("generator", "unknown")) for entry in entries]

    # 3. Instantiate Dataset with evaluation transforms
    transform = get_eval_transforms(image_size=224)
    dataset = SignalScopeDataset(samples, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 4. Instantiate Model
    model = build_classifier(backbone_name=backbone, pretrained=pretrained)
    model.eval()

    # 5. Run Inference
    all_paths: List[str] = []
    all_generators: List[str] = []
    y_true: List[int] = []
    y_scores: List[float] = []

    with torch.no_grad():
        for images, labels, meta in loader:
            probs = model.predict_probabilities(images)
            y_scores.extend(probs.cpu().numpy().tolist())
            y_true.extend(labels.cpu().numpy().tolist())
            all_paths.extend(meta["path"])
            all_generators.extend(meta["generator"])

    # 6. Format individual predictions
    predictions = []
    for p, gen, yt, score in zip(all_paths, all_generators, y_true, y_scores):
        pred_label = 1 if score >= threshold else 0
        predictions.append({
            "path": p,
            "filename": os.path.basename(p),
            "generator": gen,
            "true_label": yt,
            "true_class": "AI" if yt == 1 else "Real",
            "predicted_score": round(float(score), 4),
            "predicted_label": pred_label,
            "predicted_class": "AI" if pred_label == 1 else "Real",
            "match": bool(yt == pred_label),
        })

    # 7. Compute Evaluation Metrics
    eval_results = evaluate_with_generator_breakdown(
        y_true=y_true,
        y_scores=y_scores,
        generator_tags=all_generators,
        threshold=threshold,
    )

    return {
        "predictions": predictions,
        "eval_results": eval_results,
        "total_images": len(predictions),
        "backbone": backbone,
        "pretrained": pretrained,
        "is_model_trained": False,
    }


def print_smoke_test_report(results: Dict[str, Any]) -> None:
    """Print structured smoke test report to stdout."""
    print("=" * 80)
    print("SIGNALSCOPE 10-IMAGE PIPELINE SMOKE TEST REPORT")
    print("=" * 80)
    print(f"Backbone Architecture : {results['backbone']}")
    print(f"Pretrained Weights    : {results['pretrained']}")
    print(f"Status Note           : UNTRAINED WEIGHTS (Architecture sanity check only)")
    print("-" * 80)
    print(f"{'#':<3} {'Filename':<30} {'Generator':<12} {'True':<6} {'Score':<8} {'Pred':<6} {'Result'}")
    print("-" * 80)

    for idx, item in enumerate(results["predictions"], 1):
        match_str = "MATCH" if item["match"] else "DIFF"
        print(
            f"{idx:<3} {item['filename']:<30} {item['generator']:<12} "
            f"{item['true_class']:<6} {item['predicted_score']:<8.4f} "
            f"{item['predicted_class']:<6} {match_str}"
        )

    print("-" * 80)
    print("\n" + format_evaluation_report(results["eval_results"]))
    print("\n[CRITICAL NOTICE]")
    print("The model currently uses untrained backbone initialization.")
    print("Predictions and metrics above reflect a pipeline integrity check, NOT detector accuracy.")
    print("Do NOT use these metrics for model tuning, selection, or performance evaluation.")
    print("=" * 80)


def test_smoke_inference_pipeline():
    """PyTest test function to verify that smoke test runs cleanly if images exist."""
    # Check if images exist on disk (only runs full test if test set is present)
    all_exist = all(os.path.exists(e["path"]) for e in DEFAULT_SMOKE_TEST_IMAGES)
    if not all_exist:
        import pytest
        pytest.skip("Held-out test images are not present on disk in this test environment.")

    results = run_smoke_test()
    assert results["total_images"] == 10
    assert len(results["predictions"]) == 10
    assert results["eval_results"]["overall"]["total_samples"] == 10
    assert "roc_auc" in results["eval_results"]["overall"]


def main() -> None:
    parser = argparse.ArgumentParser(description="SignalScope 10-Image Pipeline Smoke Test")
    parser.add_argument(
        "--backbone",
        type=str,
        default="vit_base_patch16_224",
        help="Backbone architecture (default: vit_base_patch16_224)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary decision threshold (default: 0.5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Inference batch size (default: 2)",
    )
    args = parser.parse_args()

    try:
        results = run_smoke_test(
            backbone=args.backbone,
            threshold=args.threshold,
            batch_size=args.batch_size,
        )
        print_smoke_test_report(results)
    except FileNotFoundError as fnf:
        print(f"\n[ERROR] {fnf}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[UNEXPECTED ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
