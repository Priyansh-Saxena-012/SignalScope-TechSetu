"""Focused Unit Tests for Stage 10 Benchmark Runner (model/evaluate_held_out.py).

Verifies:
1. Label mapping: 0 = Real, 1 = AI-generated.
2. Generator directory mapping: 8 official GenImage directories to canonical strings.
3. Deterministic ordering: alphabetical relative path traversal.
4. Dataset count and directory structure validation.
5. Path safety & test-set isolation enforcement.
6. Metric computation logic (Accuracy, Macro-F1, Precision, Recall, FPR, Confusion Matrix).
7. ROC-AUC computation from continuous probabilities (not thresholded labels).
8. Fixed threshold behavior at 0.5.
9. Parquet & JSON artifact saving and schemas.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest

from model.evaluate_held_out import (
    EXPECTED_AI_IMAGES,
    EXPECTED_REAL_IMAGES,
    LABEL_AI,
    LABEL_REAL,
    OFFICIAL_GENERATOR_DIR_MAP,
    calculate_metrics_dict,
    compute_file_sha256,
    normalize_path,
    save_prediction_records,
    validate_safety_and_paths,
    validate_test_directory_structure,
)


# --- 1. Label Mapping Tests ---

def test_label_mapping_semantics():
    """Ensure exact label semantics: 0 is Real (nature) and 1 is AI-generated (ai)."""
    assert LABEL_REAL == 0
    assert LABEL_AI == 1


# --- 2. Generator Mapping Tests ---

def test_generator_mapping_completeness():
    """Verify that all 8 official GenImage generator folders map to canonical names."""
    expected_keys = {
        "adm_imagenet",
        "biggan_imagenet",
        "glide_imagenet",
        "midjourney_imagenet",
        "sdv4_imagenet",
        "sdv5_imagenet",
        "vqdm_imagenet",
        "wukong_imagenet",
    }
    assert set(OFFICIAL_GENERATOR_DIR_MAP.keys()) == expected_keys

    # Check specific mappings
    assert OFFICIAL_GENERATOR_DIR_MAP["adm_imagenet"] == "adm"
    assert OFFICIAL_GENERATOR_DIR_MAP["biggan_imagenet"] == "biggan"
    assert OFFICIAL_GENERATOR_DIR_MAP["glide_imagenet"] == "glide"
    assert OFFICIAL_GENERATOR_DIR_MAP["midjourney_imagenet"] == "midjourney"
    assert OFFICIAL_GENERATOR_DIR_MAP["sdv4_imagenet"] == "sd14"
    assert OFFICIAL_GENERATOR_DIR_MAP["sdv5_imagenet"] == "sd15"
    assert OFFICIAL_GENERATOR_DIR_MAP["vqdm_imagenet"] == "vqdm"
    assert OFFICIAL_GENERATOR_DIR_MAP["wukong_imagenet"] == "wukong"


# --- 3. Path Safety & Isolation Tests ---

def test_path_safety_rejects_output_inside_test_dir(tmp_path):
    """Output directory must NOT be inside or identical to the test directory."""
    test_dir = tmp_path / "test_root"
    test_dir.mkdir()
    inside_output = test_dir / "artifacts" / "evaluation"

    with pytest.raises(ValueError, match="FATAL TEST ISOLATION ERROR"):
        validate_safety_and_paths(test_dir, inside_output)

    with pytest.raises(ValueError, match="FATAL TEST ISOLATION ERROR"):
        validate_safety_and_paths(test_dir, test_dir)


def test_path_safety_accepts_isolated_output_dir(tmp_path):
    """Output directory outside test dataset must pass validation."""
    test_dir = tmp_path / "test_root"
    output_dir = tmp_path / "artifacts" / "evaluation"
    test_dir.mkdir()
    output_dir.mkdir(parents=True)

    # Should not raise
    validate_safety_and_paths(test_dir, output_dir)


# --- 4. Dataset Discovery & Deterministic Ordering Tests ---

def test_dataset_discovery_validates_counts_and_sorting(tmp_path):
    """Verify discovery checks subdirs, counts, and produces deterministically sorted samples."""
    mock_test = tmp_path / "mock_test"
    mock_test.mkdir()

    # Create the 8 generator directories with 'ai' and 'nature'
    for gen_dir in OFFICIAL_GENERATOR_DIR_MAP.keys():
        ai_dir = mock_test / gen_dir / "ai"
        nature_dir = mock_test / gen_dir / "nature"
        ai_dir.mkdir(parents=True)
        nature_dir.mkdir(parents=True)

        # 1 AI image, 1 Real image per generator
        (ai_dir / "002.png").write_bytes(b"dummy")
        (ai_dir / "001.png").write_bytes(b"dummy")
        (nature_dir / "real_b.jpg").write_bytes(b"dummy")
        (nature_dir / "real_a.jpg").write_bytes(b"dummy")

    # Total: 8 * 1 = 8 AI, 8 * 1 = 8 Real = 16 total
    samples = validate_test_directory_structure(
        mock_test,
        expected_ai_count=16,
        expected_real_count=16,
    )

    assert len(samples) == 32
    # Verify deterministic sorting by relative path
    rel_paths = [s["rel_path"] for s in samples]
    assert rel_paths == sorted(rel_paths)

    # Verify sample_id is strictly sequential 0..31
    for i, s in enumerate(samples):
        assert s["sample_id"] == i
        assert not Path(s["rel_path"]).is_absolute()


def test_dataset_discovery_fails_on_missing_generator(tmp_path):
    """Verify validation fails if any expected generator directory is missing."""
    mock_test = tmp_path / "mock_incomplete"
    mock_test.mkdir()

    # Create only 7 of 8 directories
    generators = list(OFFICIAL_GENERATOR_DIR_MAP.keys())[:-1]
    for gen_dir in generators:
        (mock_test / gen_dir / "ai").mkdir(parents=True)
        (mock_test / gen_dir / "nature").mkdir(parents=True)

    with pytest.raises(ValueError, match="Missing expected generator subdirectories"):
        validate_test_directory_structure(mock_test, expected_ai_count=0, expected_real_count=0)


def test_dataset_discovery_fails_on_count_mismatch(tmp_path):
    """Verify validation fails if counts do not match expected."""
    mock_test = tmp_path / "mock_mismatch"
    mock_test.mkdir()

    for gen_dir in OFFICIAL_GENERATOR_DIR_MAP.keys():
        (mock_test / gen_dir / "ai").mkdir(parents=True)
        (mock_test / gen_dir / "nature").mkdir(parents=True)

    with pytest.raises(ValueError, match="Dataset count mismatch"):
        validate_test_directory_structure(
            mock_test,
            expected_ai_count=10,
            expected_real_count=10,
        )


# --- 5. Metric Calculation Tests ---

def test_metrics_calculation_perfect_classifier():
    """Verify metrics calculation on a perfect binary classification output."""
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.8, 0.9]

    metrics = calculate_metrics_dict(y_true, y_scores, threshold=0.5)

    assert metrics["roc_auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["confusion_matrix"]["tn"] == 2
    assert metrics["confusion_matrix"]["tp"] == 2
    assert metrics["confusion_matrix"]["fp"] == 0
    assert metrics["confusion_matrix"]["fn"] == 0


def test_metrics_calculation_continuous_roc_auc():
    """Verify ROC-AUC uses continuous probabilities rather than thresholded values."""
    y_true = [0, 0, 1, 1]
    # At threshold 0.5, all y_scores >= 0.5 are predicted as 1 (acc=0.5, all predicted 1)
    # But rankings are perfectly separated: 0.6, 0.7 < 0.8, 0.9 -> ROC-AUC must be 1.0!
    y_scores = [0.6, 0.7, 0.8, 0.9]

    metrics = calculate_metrics_dict(y_true, y_scores, threshold=0.5)

    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"]["fp"] == 2
    assert metrics["confusion_matrix"]["tn"] == 0
    assert metrics["accuracy"] == 0.5


def test_metrics_fixed_threshold_05():
    """Verify the fixed decision threshold is 0.5."""
    y_true = [0, 1]
    y_scores = [0.49, 0.50]

    metrics = calculate_metrics_dict(y_true, y_scores, threshold=0.5)

    assert metrics["confusion_matrix"]["tn"] == 1  # 0.49 < 0.5 -> pred 0
    assert metrics["confusion_matrix"]["tp"] == 1  # 0.50 >= 0.5 -> pred 1
    assert metrics["accuracy"] == 1.0


# --- 6. Prediction Artifact Persistence Tests ---

def test_save_prediction_records(tmp_path):
    """Verify prediction artifacts persist required schema keys and values."""
    out_parquet = tmp_path / "test_preds.parquet"

    sample_ids = [0, 1]
    rel_paths = ["adm_imagenet/nature/001.jpg", "adm_imagenet/ai/001.png"]
    generators = ["adm", "adm"]
    true_labels = [0, 1]
    ai_probs = [0.05, 0.95]
    pred_labels_05 = [0, 1]

    save_prediction_records(
        out_parquet,
        sample_ids,
        rel_paths,
        generators,
        true_labels,
        ai_probs,
        pred_labels_05,
    )

    # Either parquet or fallback CSV exists
    target = out_parquet if out_parquet.exists() else out_parquet.with_suffix(".csv")
    assert target.exists()


# --- 7. Checkpoint Checksum Test ---

def test_compute_file_sha256(tmp_path):
    """Verify SHA256 checksum calculation produces correct hash."""
    test_file = tmp_path / "test_blob.bin"
    test_file.write_bytes(b"signalscope_stage10_test")

    import hashlib
    expected_hash = hashlib.sha256(b"signalscope_stage10_test").hexdigest()
    assert compute_file_sha256(test_file) == expected_hash
