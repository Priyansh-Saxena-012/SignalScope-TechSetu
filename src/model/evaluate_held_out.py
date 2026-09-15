"""SignalScope Official Held-Out Benchmark Runner (Stage 10).

Executes the official evaluation on the held-out benchmark dataset:
- STRICT TEST-SET ISOLATION: Read-only access to held-out test data.
  Never writes, caches, modifies, or annotates inside the test set.
- REPRODUCIBLE DETERMINISTIC DISCOVERY: Alphabetical file sorting for stable sample_id.
- FROZEN MODEL INFERENCE: ViT-Base/16 checkpoint with exact eval transforms.
- COMPREHENSIVE METRICS: Overall and per-generator ROC-AUC (continuous probabilities),
  Macro-F1, Accuracy, Precision, Recall, FPR, Confusion Matrix.
- STANDARDIZED ARTIFACTS:
  1. artifacts/evaluation/predictions_heldout_100k.parquet
  2. artifacts/evaluation/benchmark_heldout_100k.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Optional heavy imports (loaded when executing or when available)
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except (ImportError, OSError):
    torch = None
    nn = None
    DataLoader = None
    Dataset = object

def normalize_path(p: Union[str, Path]) -> Path:
    """Normalize path for robust cross-platform comparison (case, slashes, realpath)."""
    abs_p = os.path.normcase(os.path.realpath(os.path.abspath(str(p))))
    return Path(abs_p)

# Canonical mapping from directory names in test/ to canonical generator names
OFFICIAL_GENERATOR_DIR_MAP: Dict[str, str] = {
    "adm_imagenet": "adm",
    "biggan_imagenet": "biggan",
    "glide_imagenet": "glide",
    "midjourney_imagenet": "midjourney",
    "sdv4_imagenet": "sd14",
    "sdv5_imagenet": "sd15",
    "vqdm_imagenet": "vqdm",
    "wukong_imagenet": "wukong",
}

# Label semantics: 0 = Real (nature), 1 = AI-generated (ai)
LABEL_REAL: int = 0
LABEL_AI: int = 1

EXPECTED_TOTAL_IMAGES: int = 100_000
EXPECTED_AI_IMAGES: int = 50_000
EXPECTED_REAL_IMAGES: int = 50_000


class HeldOutDataset(Dataset):
    """Deterministic, read-only PyTorch Dataset for the official held-out benchmark.

    Parameters
    ----------
    samples : list of dict
        List of sample records:
        {"sample_id": int, "rel_path": str, "full_path": Path, "generator": str, "true_label": int}
    transform : callable, optional
        Preprocessing transforms to apply to each PIL image.
    """

    def __init__(self, samples: List[Dict[str, Any]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, int, str, str, int]:
        if torch is None or Image is None:
            raise RuntimeError("PyTorch and PIL must be installed to index HeldOutDataset.")

        item = self.samples[idx]
        img_path = item["full_path"]

        # Read strictly read-only using PIL
        with open(img_path, "rb") as f:
            with Image.open(f) as img:
                img = img.convert("RGB")
                if self.transform is not None:
                    tensor = self.transform(img)
                else:
                    tensor = torch.zeros((3, 224, 224))

        return (
            tensor,
            item["true_label"],
            item["generator"],
            item["rel_path"],
            item["sample_id"],
        )


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file in 64KB blocks."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_git_commit_hash(repo_dir: Path) -> str:
    """Retrieve current HEAD git commit hash."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            stderr=subprocess.DEVNULL,
        ).decode("ascii").strip()
        return out
    except Exception:
        return "unknown"


def validate_test_directory_structure(
    test_root: Path,
    expected_ai_count: int = EXPECTED_AI_IMAGES,
    expected_real_count: int = EXPECTED_REAL_IMAGES,
) -> List[Dict[str, Any]]:
    """Scan and validate the official test dataset directory structure.

    Enforces:
    1. test_root exists and is a directory.
    2. All 8 expected generator directories exist.
    3. Each generator contains 'ai' and 'nature' subdirectories.
    4. Deterministic sorting: sorted alphabetically by relative path.
    5. Exact image counts match expected counts.

    Returns
    -------
    list of dict
        Ordered sample records.
    """
    if not test_root.exists() or not test_root.is_dir():
        raise FileNotFoundError(f"Held-out test dataset directory not found: {test_root}")

    found_dirs = {p.name for p in test_root.iterdir() if p.is_dir()}
    expected_dirs = set(OFFICIAL_GENERATOR_DIR_MAP.keys())
    missing_dirs = expected_dirs - found_dirs
    if missing_dirs:
        raise ValueError(
            f"Missing expected generator subdirectories in {test_root}: {sorted(list(missing_dirs))}"
        )

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    raw_samples: List[Dict[str, Any]] = []

    total_ai = 0
    total_real = 0

    for dir_name in sorted(OFFICIAL_GENERATOR_DIR_MAP.keys()):
        gen_name = OFFICIAL_GENERATOR_DIR_MAP[dir_name]
        gen_dir = test_root / dir_name

        ai_dir = gen_dir / "ai"
        nature_dir = gen_dir / "nature"

        if not ai_dir.exists() or not ai_dir.is_dir():
            raise ValueError(f"Missing 'ai' directory in {gen_dir}")
        if not nature_dir.exists() or not nature_dir.is_dir():
            raise ValueError(f"Missing 'nature' directory in {gen_dir}")

        # Scan AI images
        for entry in sorted(ai_dir.iterdir(), key=lambda p: p.name):
            if entry.is_file() and entry.suffix.lower() in valid_extensions:
                rel_path = f"{dir_name}/ai/{entry.name}"
                raw_samples.append({
                    "rel_path": rel_path,
                    "full_path": entry,
                    "generator": gen_name,
                    "true_label": LABEL_AI,
                })
                total_ai += 1

        # Scan Real/Nature images
        for entry in sorted(nature_dir.iterdir(), key=lambda p: p.name):
            if entry.is_file() and entry.suffix.lower() in valid_extensions:
                rel_path = f"{dir_name}/nature/{entry.name}"
                raw_samples.append({
                    "rel_path": rel_path,
                    "full_path": entry,
                    "generator": gen_name,
                    "true_label": LABEL_REAL,
                })
                total_real += 1

    # Verify counts
    if total_ai != expected_ai_count or total_real != expected_real_count:
        raise ValueError(
            f"Dataset count mismatch in {test_root}: "
            f"Found AI={total_ai} (expected {expected_ai_count}), "
            f"Real={total_real} (expected {expected_real_count}), "
            f"Total={len(raw_samples)} (expected {expected_ai_count + expected_real_count})"
        )

    # Sort deterministically by relative path and assign sample_id
    raw_samples.sort(key=lambda x: x["rel_path"])
    for idx, item in enumerate(raw_samples):
        item["sample_id"] = idx

    return raw_samples


def validate_safety_and_paths(test_root: Path, output_dir: Path) -> None:
    """Ensure output directory is strictly outside the held-out test dataset directory."""
    norm_test = normalize_path(test_root)
    norm_out = normalize_path(output_dir)

    if norm_test == norm_out or norm_test in norm_out.parents:
        raise ValueError(
            f"[FATAL TEST ISOLATION ERROR] Output directory {output_dir} is inside "
            f"or identical to held-out test directory {test_root}. Output files must "
            f"never be written to the held-out test set."
        )


def load_frozen_vit_model(
    checkpoint_path: Path,
    device: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """Load the frozen ViT-Base/16 model from checkpoint.

    Sets model to eval mode and returns model + checkpoint metadata.
    """
    if torch is None:
        raise RuntimeError("PyTorch must be installed to load the model.")

    from model.backbone import build_classifier

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        backbone_name = checkpoint.get("backbone", "vit_base_patch16_224")
        ckpt_config = checkpoint.get("config", {})
        model_cfg = ckpt_config.get("model", {}) if isinstance(ckpt_config, dict) else {}
        drop_rate = model_cfg.get("dropout", 0.2)
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        backbone_name = "vit_base_patch16_224"
        drop_rate = 0.2
    else:
        raise ValueError(f"Invalid checkpoint format at {checkpoint_path}")

    model = build_classifier(
        backbone_name=backbone_name,
        pretrained=False,
        drop_rate=drop_rate,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    return model, checkpoint


def calculate_metrics_dict(
    y_true: List[int],
    y_scores: List[float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate standard binary classification metrics using pure python/sklearn."""
    try:
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        import numpy as np

        y_t = np.asarray(y_true, dtype=int)
        y_s = np.asarray(y_scores, dtype=float)

        # ROC-AUC requires both classes
        if len(np.unique(y_t)) > 1:
            auc_val = float(roc_auc_score(y_t, y_s))
        else:
            auc_val = None

        y_pred = (y_s >= threshold).astype(int)

        acc = float(accuracy_score(y_t, y_pred))
        macro_f1 = float(f1_score(y_t, y_pred, average="macro", zero_division=0))
        prec = float(precision_score(y_t, y_pred, zero_division=0))
        rec = float(recall_score(y_t, y_pred, zero_division=0))

        cm = confusion_matrix(y_t, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        return {
            "roc_auc": round(auc_val, 4) if auc_val is not None else None,
            "macro_f1": round(macro_f1, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "fpr": round(fpr, 4),
            "total_samples": len(y_t),
            "confusion_matrix": {
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "matrix": [[tn, fp], [fn, tp]],
            },
        }
    except ImportError:
        # Fallback if scikit-learn is not present
        return _calculate_metrics_fallback(y_true, y_scores, threshold)


def _calculate_metrics_fallback(
    y_true: List[int],
    y_scores: List[float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Pure-Python fallback for confusion matrix, accuracy, precision, recall, and trapezoidal AUC."""
    n = len(y_true)
    tn = fp = fn = tp = 0

    for yt, ys in zip(y_true, y_scores):
        yp = 1 if ys >= threshold else 0
        if yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1
        elif yt == 1 and yp == 1:
            tp += 1

    acc = (tp + tn) / n if n > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    f1_pos = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    prec_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_neg = (2 * prec_neg * rec_neg) / (prec_neg + rec_neg) if (prec_neg + rec_neg) > 0 else 0.0
    macro_f1 = (f1_pos + f1_neg) / 2.0

    # Mann-Whitney U for ROC-AUC
    n_pos = sum(y_true)
    n_neg = n - n_pos
    if n_pos > 0 and n_neg > 0:
        paired = sorted(zip(y_scores, y_true), key=lambda x: x[0])
        rank_sum_pos = 0.0
        i = 0
        while i < n:
            j = i
            while j < n and paired[j][0] == paired[i][0]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                if paired[k][1] == 1:
                    rank_sum_pos += avg_rank
            i = j
        u = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
        auc_val = u / (n_pos * n_neg)
    else:
        auc_val = None

    return {
        "roc_auc": round(auc_val, 4) if auc_val is not None else None,
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "fpr": round(fpr, 4),
        "total_samples": n,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": [[tn, fp], [fn, tp]],
        },
    }


def save_prediction_records(
    output_path: Path,
    sample_ids: List[int],
    rel_paths: List[str],
    generators: List[str],
    true_labels: List[int],
    ai_probs: List[float],
    pred_labels_05: List[int],
) -> None:
    """Save raw prediction records to parquet (or fallback CSV)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_arrays(
            [
                pa.array(sample_ids, type=pa.int64()),
                pa.array(rel_paths, type=pa.string()),
                pa.array(generators, type=pa.string()),
                pa.array(true_labels, type=pa.int32()),
                pa.array(ai_probs, type=pa.float32()),
                pa.array(pred_labels_05, type=pa.int32()),
            ],
            names=[
                "sample_id",
                "rel_path",
                "generator",
                "true_label",
                "ai_probability",
                "pred_label_05",
            ],
        )
        pq.write_table(table, str(output_path))
    except Exception:
        # Fallback to CSV if pyarrow is unavailable
        csv_path = output_path.with_suffix(".csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "rel_path", "generator", "true_label", "ai_probability", "pred_label_05"])
            for row in zip(sample_ids, rel_paths, generators, true_labels, ai_probs, pred_labels_05):
                writer.writerow(row)


def evaluate_benchmark(
    test_root: Path,
    checkpoint_path: Path,
    output_dir: Path,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    device_str: Optional[str] = None,
    expected_ai_count: int = EXPECTED_AI_IMAGES,
    expected_real_count: int = EXPECTED_REAL_IMAGES,
) -> Dict[str, Any]:
    """Execute the full 100k held-out benchmark evaluation."""
    test_root = Path(test_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir).resolve()

    # 1. Safety validation
    validate_safety_and_paths(test_root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Dataset discovery and validation
    print(f"[1/5] Discovering and validating held-out dataset at: {test_root}")
    samples = validate_test_directory_structure(
        test_root,
        expected_ai_count=expected_ai_count,
        expected_real_count=expected_real_count,
    )
    print(f"      Verified {len(samples):,} images ({expected_ai_count:,} AI, {expected_real_count:,} Real).")

    # 3. Checkpoint SHA256 and Git info
    print(f"[2/5] Inspecting checkpoint: {checkpoint_path}")
    checkpoint_sha256 = compute_file_sha256(checkpoint_path)
    git_commit = get_git_commit_hash(REPO_ROOT)
    print(f"      Git Commit:        {git_commit}")
    print(f"      Checkpoint SHA256: {checkpoint_sha256}")

    # 4. Device and Model Initialization
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if (torch and torch.cuda.is_available()) else "cpu")
    print(f"[3/5] Loading frozen ViT-Base/16 on device: {device}")

    from src.data.transforms import get_eval_transforms
    model, ckpt_dict = load_frozen_vit_model(checkpoint_path, device)
    eval_transforms = get_eval_transforms(image_size=224)

    # 5. Batched Inference
    dataset = HeldOutDataset(samples, transform=eval_transforms)
    actual_workers = 0 if sys.platform == "win32" and num_workers > 0 and not torch.cuda.is_available() else num_workers

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Strictly deterministic order
        num_workers=actual_workers,
        pin_memory=(pin_memory and device.type == "cuda"),
    )

    print(f"[4/5] Running inference over {len(dataset):,} samples...")
    print(f"      Batch size: {batch_size}, Num workers: {actual_workers}, FP16 Autocast: {device.type == 'cuda'}")

    all_sample_ids: List[int] = []
    all_rel_paths: List[str] = []
    all_generators: List[str] = []
    all_true_labels: List[int] = []
    all_ai_probs: List[float] = []

    use_cuda = (device.type == "cuda")
    if use_cuda:
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch_idx, (tensors, true_labels, generators, rel_paths, sample_ids) in enumerate(loader):
            tensors = tensors.to(device, non_blocking=True)

            if use_cuda:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(tensors)
            else:
                logits = model(tensors)

            probs = torch.sigmoid(logits).view(-1).cpu().tolist()

            all_sample_ids.extend(sample_ids.tolist())
            all_rel_paths.extend(rel_paths)
            all_generators.extend(generators)
            all_true_labels.extend(true_labels.tolist())
            all_ai_probs.extend(probs)

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(loader):
                elapsed = time.perf_counter() - start_time
                done_samples = len(all_ai_probs)
                rate = done_samples / elapsed if elapsed > 0 else 0.0
                print(f"      Processed {done_samples:,}/{len(dataset):,} images ({rate:.1f} img/s)...")

    if use_cuda:
        torch.cuda.synchronize()

    total_time = time.perf_counter() - start_time
    throughput = len(dataset) / total_time if total_time > 0 else 0.0
    print(f"      Inference completed in {total_time:.1f}s ({throughput:.1f} img/s).")

    # 6. Save Prediction Artifact (Parquet)
    parquet_out = output_dir / "predictions_heldout_100k.parquet"
    print(f"[5/5] Saving prediction artifact to: {parquet_out}")

    pred_labels_05 = [1 if p >= 0.5 else 0 for p in all_ai_probs]
    save_prediction_records(
        parquet_out,
        all_sample_ids,
        all_rel_paths,
        all_generators,
        all_true_labels,
        all_ai_probs,
        pred_labels_05,
    )

    # 7. Compute Metrics (Overall & Per-Generator)
    print("      Computing comprehensive metrics...")
    overall_metrics = calculate_metrics_dict(all_true_labels, all_ai_probs, threshold=0.5)

    unique_generators = sorted(list(set(all_generators)))
    real_indices = [i for i, y in enumerate(all_true_labels) if y == LABEL_REAL]

    by_generator: Dict[str, Any] = {}
    for gen in unique_generators:
        gen_ai_indices = [i for i, (g, y) in enumerate(zip(all_generators, all_true_labels)) if g == gen and y == LABEL_AI]
        if not gen_ai_indices:
            continue

        subset_indices = gen_ai_indices + real_indices
        subset_labels = [all_true_labels[i] for i in subset_indices]
        subset_scores = [all_ai_probs[i] for i in subset_indices]

        gen_metrics = calculate_metrics_dict(subset_labels, subset_scores, threshold=0.5)
        by_generator[gen] = {
            "ai_sample_count": len(gen_ai_indices),
            "total_evaluated_with_real": len(subset_indices),
            "roc_auc": gen_metrics["roc_auc"],
            "accuracy": gen_metrics["accuracy"],
            "precision": gen_metrics["precision"],
            "recall": gen_metrics["recall"],
            "fpr": gen_metrics["fpr"],
            "confusion_matrix": gen_metrics["confusion_matrix"],
        }

    # 8. Build Benchmark Results Structure
    benchmark_results: Dict[str, Any] = {
        "metadata": {
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": git_commit,
            "checkpoint_path": str(checkpoint_path.name),
            "checkpoint_sha256": checkpoint_sha256,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if (torch and torch.cuda.is_available()) else "CPU",
            "torch_version": torch.__version__ if torch else "unknown",
            "platform": platform.platform(),
            "batch_size": batch_size,
            "num_workers": actual_workers,
            "fp16_autocast": (device.type == "cuda"),
            "dataset_root": str(test_root),
            "dataset_counts": {
                "total": len(samples),
                "ai": expected_ai_count,
                "real": expected_real_count,
            },
            "runtime_seconds": round(total_time, 2),
            "throughput_fps": round(throughput, 2),
        },
        "threshold_used": 0.5,
        "overall": overall_metrics,
        "by_generator": by_generator,
    }

    json_out = output_dir / "benchmark_heldout_100k.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"      Benchmark results saved to: {json_out}")

    return benchmark_results


def parse_args():
    parser = argparse.ArgumentParser(description="SignalScope Official Held-Out Benchmark Runner")
    parser.add_argument(
        "--test-dir",
        type=str,
        default=r"C:\Datasets\SignalScope\test",
        help="Path to held-out test dataset directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model/weights/checkpoint_best.pth",
        help="Path to frozen model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/evaluation",
        help="Directory where output parquet and json artifacts will be stored",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size (default: 64)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader num_workers (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device ('cuda', 'cpu', or None for auto-detect)",
    )
    parser.add_argument(
        "--expected-ai-count",
        type=int,
        default=EXPECTED_AI_IMAGES,
        help=f"Expected number of AI images (default: {EXPECTED_AI_IMAGES})",
    )
    parser.add_argument(
        "--expected-real-count",
        type=int,
        default=EXPECTED_REAL_IMAGES,
        help=f"Expected number of real images (default: {EXPECTED_REAL_IMAGES})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_benchmark(
        test_root=Path(args.test_dir),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_str=args.device,
        expected_ai_count=args.expected_ai_count,
        expected_real_count=args.expected_real_count,
    )
