"""SignalScope Baseline Analysis Script.

Standalone entry point for evaluating a saved checkpoint on the Tiny-GenImage
development validation set with full metric computation and generator-wise
breakdown.

This script:
- Loads a trained SignalScopeClassifier from a saved checkpoint
- Runs inference on the configured validation DataLoader
- Collects predictions, ground-truth labels, and generator metadata
- Computes overall and per-generator evaluation metrics
- Outputs results as human-readable text and machine-readable JSON

CRITICAL: This script evaluates only the development validation set.
The official 100k held-out benchmark must NEVER be used here.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.backbone import build_classifier, SignalScopeClassifier
from src.data.parquet_dataset import (
    DEFAULT_TINY_GENIMAGE_VAL_SOURCE,
    TinyGenImageParquetDataset,
)
from src.data.transforms import get_eval_transforms
from src.evaluation.evaluate import (
    evaluate_with_generator_breakdown,
    format_evaluation_report,
)
from src.utils.config import load_config


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[SignalScopeClassifier, Dict[str, Any]]:
    """Load a SignalScopeClassifier from a saved training checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint `.pth` file.
    device : torch.device
        Target device to map weights onto.

    Returns
    -------
    model : SignalScopeClassifier
        Model with loaded weights in eval mode.
    checkpoint_meta : dict
        Checkpoint metadata (epoch, val_roc_auc, backbone, config).
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model configuration from checkpoint
    ckpt_config = checkpoint.get("config", {})
    model_cfg = ckpt_config.get("model", {})
    backbone_name = checkpoint.get("backbone", model_cfg.get("backbone", "vit_base_patch16_224"))
    drop_rate = model_cfg.get("dropout", 0.2)

    # Build model architecture and load weights
    model = build_classifier(
        backbone_name=backbone_name,
        pretrained=False,  # Weights come from checkpoint, not from pretrained hub
        drop_rate=drop_rate,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    checkpoint_meta = {
        "checkpoint_path": str(ckpt_path),
        "epoch": checkpoint.get("epoch"),
        "val_roc_auc_at_save": checkpoint.get("val_roc_auc"),
        "backbone": backbone_name,
    }
    return model, checkpoint_meta


def build_val_loader(
    config: Dict[str, Any],
    device: torch.device,
) -> DataLoader:
    """Construct a validation DataLoader from experiment configuration.

    Parameters
    ----------
    config : dict
        Experiment configuration dictionary.
    device : torch.device
        Target device (used for pin_memory decision).

    Returns
    -------
    DataLoader
        Validation DataLoader with eval transforms applied.
    """
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    image_size = model_cfg.get("image_size", 224)
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 0)
    held_out_dir = data_cfg.get("held_out_test_dir")

    val_source = data_cfg.get("val_data_path") or DEFAULT_TINY_GENIMAGE_VAL_SOURCE
    val_transform = get_eval_transforms(image_size=image_size)

    val_ds = TinyGenImageParquetDataset(
        source=val_source,
        transform=val_transform,
        held_out_test_dir=held_out_dir,
    )

    pin_memory = (device.type == "cuda")
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return val_loader


def evaluate_checkpoint(
    model: SignalScopeClassifier,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> Tuple[List[int], List[float], List[str]]:
    """Run inference on a DataLoader, collecting labels, scores, and generator tags.

    Parameters
    ----------
    model : SignalScopeClassifier
        Model in eval mode.
    val_loader : DataLoader
        Validation DataLoader yielding (images, labels, meta) tuples.
    device : torch.device
    use_amp : bool
        Whether to use automatic mixed precision for inference.

    Returns
    -------
    y_true : list of int
        Ground truth binary labels.
    y_scores : list of float
        Predicted probabilities (sigmoid of logits).
    generator_tags : list of str
        Human-readable generator name for each sample.
    """
    model.eval()
    all_labels: List[int] = []
    all_scores: List[float] = []
    all_generators: List[str] = []

    total_batches = len(val_loader)

    with torch.no_grad():
        for batch_idx, (images, labels, meta) in enumerate(val_loader):
            images = images.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)

            probs = torch.sigmoid(logits).cpu().numpy()

            all_scores.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_generators.extend(meta["generator"])

            # Progress logging
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
                print(f"  Inference [{batch_idx + 1}/{total_batches}]")

    return all_labels, all_scores, all_generators


def run_baseline_analysis(
    checkpoint_path: str,
    config_path: str,
    output_path: Optional[str] = None,
    device_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a complete baseline analysis pipeline.

    Parameters
    ----------
    checkpoint_path : str
        Path to the saved checkpoint `.pth` file.
    config_path : str
        Path to the experiment config YAML.
    output_path : str, optional
        Path to save JSON results. If None, results are printed only.
    device_override : str, optional
        Force device ('cpu', 'cuda'). If None, auto-detected.

    Returns
    -------
    dict
        Complete analysis results including overall and per-generator metrics.
    """
    # 1. Resolve device
    if device_override:
        device = torch.device(device_override)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    use_amp = (device.type == "cuda")

    print("=" * 65)
    print("SIGNALSCOPE BASELINE ANALYSIS")
    print("=" * 65)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Config     : {config_path}")
    print(f"Device     : {device}")
    print(f"AMP        : {use_amp}")
    print("=" * 65)

    # 2. Load config and checkpoint
    config = load_config(config_path)
    model, checkpoint_meta = load_checkpoint(checkpoint_path, device)

    print(f"Backbone   : {checkpoint_meta['backbone']}")
    print(f"Epoch      : {checkpoint_meta['epoch']}")
    print(f"Saved AUC  : {checkpoint_meta['val_roc_auc_at_save']}")
    print("-" * 65)

    # 3. Build validation loader
    val_loader = build_val_loader(config, device)
    val_samples = len(val_loader.dataset)
    print(f"Val Samples: {val_samples}")
    print(f"Val Source  : {config.get('data', {}).get('val_data_path', 'default')}")
    print("-" * 65)

    # 4. Run inference
    print("Running inference...")
    y_true, y_scores, generator_tags = evaluate_checkpoint(
        model, val_loader, device, use_amp=use_amp,
    )
    print(f"Collected  : {len(y_true)} predictions")
    print("-" * 65)

    # 5. Compute metrics with generator breakdown
    results = evaluate_with_generator_breakdown(
        y_true=y_true,
        y_scores=y_scores,
        generator_tags=generator_tags,
        threshold=0.5,
    )

    # 6. Print human-readable report
    print(format_evaluation_report(results))

    # 7. Assemble full JSON output
    analysis_output = {
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_meta,
        "dataset": {
            "split": "validation",
            "source": config.get("data", {}).get("val_data_path"),
            "total_samples": val_samples,
        },
        "device": str(device),
        "overall": results["overall"],
        "per_generator": results["per_generator"],
        "unseen_generator_split": results.get("unseen_generator_split"),
    }

    # 8. Save JSON if output path specified
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(analysis_output, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

    return analysis_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SignalScope Baseline Analysis — Evaluate checkpoint on development validation set"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved checkpoint .pth file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to experiment configuration YAML",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON analysis results (e.g., experiments/baseline_analysis.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override execution device ('cpu', 'cuda', 'auto')",
    )
    args = parser.parse_args()

    run_baseline_analysis(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_path=args.output,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
