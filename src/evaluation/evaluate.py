"""SignalScope Evaluation Utilities.

Implements dataset-independent metrics computation:
- ROC-AUC (Overall and sliced by generator family)
- Macro-F1 Score
- Accuracy
- False Positive Rate (FPR) at stated threshold
- Confusion Matrix (TN, FP, FN, TP)
- Full Scikit-Learn Classification Report

The functions accept pure prediction arrays/lists and do not depend on specific dataset loaders.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_scores: Union[np.ndarray, List[float]],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute core classification metrics from true labels and predicted probabilities.

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Binary ground truth labels (0 for Real, 1 for AI-generated).
    y_scores : array-like of shape (N,)
        Continuous confidence or probability scores for the positive class (AI).
    threshold : float, default=0.5
        Operating decision threshold for binary classification.

    Returns
    -------
    dict
        Dictionary containing all standard evaluation metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    if len(y_true) != len(y_scores):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(y_scores)={len(y_scores)}")

    # 1. ROC-AUC (Requires at least one sample from each class)
    has_both_classes = len(np.unique(y_true)) > 1
    if has_both_classes:
        auc_score = float(roc_auc_score(y_true, y_scores))
    else:
        auc_score = float("nan")

    # 2. Binary predictions at operating threshold
    y_pred = (y_scores >= threshold).astype(int)

    # 3. Accuracy and Macro-F1
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    # 4. Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # 5. False Positive Rate (FPR) and True Positive Rate (TPR)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # 6. Classification Report
    cls_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "roc_auc": round(auc_score, 4) if not np.isnan(auc_score) else None,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "precision": round(prec, 4),
        "recall_tpr": round(rec, 4),
        "false_positive_rate": round(fpr, 4),
        "threshold_used": float(threshold),
        "total_samples": int(len(y_true)),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": [[tn, fp], [fn, tp]],
        },
        "classification_report": cls_report,
    }


def evaluate_with_generator_breakdown(
    y_true: Union[np.ndarray, List[int]],
    y_scores: Union[np.ndarray, List[float]],
    generator_tags: List[str],
    unseen_generators: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute overall metrics plus generator-specific and unseen-split breakdowns.

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Binary ground truth labels (0=Real, 1=Synthetic).
    y_scores : array-like of shape (N,)
        Predicted AI probabilities.
    generator_tags : list of str of length N
        Generator identifiers for each sample (e.g. 'midjourney', 'stable_diffusion', 'real').
    unseen_generators : list of str, optional
        List of generator names that belong to the unseen evaluation partition.
    threshold : float, default=0.5

    Returns
    -------
    dict
        Comprehensive evaluation structure including overall and sliced metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    # 1. Overall Metrics
    overall = compute_metrics(y_true, y_scores, threshold=threshold)

    # 2. Per-Generator Metrics
    tags = np.asarray(generator_tags)
    unique_generators = sorted(list(set(g for g in tags if g != "real" and g != "unknown")))

    per_generator = {}
    for gen in unique_generators:
        # Evaluate generator fakes against all real images in test set
        mask = (tags == gen) | (y_true == 0)
        if np.sum(tags == gen) > 0 and np.sum(y_true == 0) > 0:
            gen_metrics = compute_metrics(y_true[mask], y_scores[mask], threshold=threshold)
            per_generator[gen] = {
                "roc_auc": gen_metrics["roc_auc"],
                "accuracy": gen_metrics["accuracy"],
                "macro_f1": gen_metrics["macro_f1"],
                "sample_count": int(np.sum(tags == gen)),
            }

    # 3. Unseen Generator Aggregate Metric
    unseen_metrics = None
    if unseen_generators:
        unseen_set = set(unseen_generators)
        unseen_mask = np.isin(tags, list(unseen_set)) | (y_true == 0)
        if np.sum(np.isin(tags, list(unseen_set))) > 0 and np.sum(y_true == 0) > 0:
            unseen_metrics = compute_metrics(y_true[unseen_mask], y_scores[unseen_mask], threshold=threshold)

    return {
        "overall": overall,
        "unseen_generator_split": unseen_metrics,
        "per_generator": per_generator,
        "unseen_generators_list": unseen_generators or [],
    }


def format_evaluation_report(results: Dict[str, Any]) -> str:
    """Format evaluation results into a human-readable summary table."""
    overall = results.get("overall", results)
    cm = overall.get("confusion_matrix", {})

    lines = [
        "=" * 65,
        "SIGNALSCOPE EVALUATION REPORT",
        "=" * 65,
        f"Operating Threshold : {overall.get('threshold_used')}",
        f"Total Samples       : {overall.get('total_samples')}",
        f"Overall ROC-AUC     : {overall.get('roc_auc')}",
        f"Macro-F1 Score      : {overall.get('macro_f1')}",
        f"Accuracy            : {overall.get('accuracy')}",
        f"False Positive Rate : {overall.get('false_positive_rate')}",
        f"Precision           : {overall.get('precision')}",
        f"Recall (TPR)        : {overall.get('recall_tpr')}",
        "-" * 65,
        f"Confusion Matrix    : TN={cm.get('tn')}, FP={cm.get('fp')}, FN={cm.get('fn')}, TP={cm.get('tp')}",
    ]

    unseen = results.get("unseen_generator_split")
    if unseen:
        lines.extend([
            "-" * 65,
            f"Unseen Generators   : {results.get('unseen_generators_list')}",
            f"Unseen-Split ROC-AUC: {unseen.get('roc_auc')}",
            f"Unseen-Split F1     : {unseen.get('macro_f1')}",
            f"Unseen-Split Acc    : {unseen.get('accuracy')}",
        ])

    lines.append("=" * 65)
    return "\n".join(lines)
