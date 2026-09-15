"""Unit tests for SignalScope Robustness Evaluation Framework (Stage 12).

Tests:
1. JPEG compression transformation correctness and parameter validation.
2. Bilinear downscale and restoration correctness and parameter validation.
3. Gaussian blur filtering correctness and parameter validation.
4. Screenshot-like compound degradation proxy correctness.
5. Light benign photometric edits (brightness, contrast) correctness.
6. Deterministic output behavior given identical inputs.
7. Input image immutability (original image pixels and object unaffected).
8. Output format, dimensions, RGB mode, and valid pixel value ranges [0, 255].
9. Strict rejection of the official held-out test directory (C:\\Datasets\\SignalScope\\test).
10. Mathematical accuracy of probability shift metrics (Delta p, MAPS).
11. Mathematical accuracy of verdict flips, AI evasion rate, and Real false-alarm rate.
12. Lightweight end-to-end execution using mock inference and mock samples.

CRITICAL: All tests run with synthetic in-memory images. Zero dependency on the
real 7k dataset or 1GB checkpoint weights is required.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.evaluation.robustness import (
    RobustnessCondition,
    SampleRecord,
    TestSetContaminationError,
    validate_path_safety,
    apply_downscale_restore,
    apply_gaussian_blur,
    apply_jpeg_compression,
    apply_photometric_edit,
    apply_screenshot_proxy,
    compute_robustness_metrics,
    evaluate_samples_on_model,
    get_standard_robustness_conditions,
    save_robustness_artifacts,
)


@pytest.fixture
def synthetic_rgb_image() -> Image.Image:
    """Create a deterministic synthetic 3-channel RGB image (128x128)."""
    rng = np.random.RandomState(42)
    data = rng.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
    return Image.fromarray(data, mode="RGB")


@pytest.fixture
def synthetic_mock_samples(synthetic_rgb_image) -> list[SampleRecord]:
    """Provide a small list of synthetic SampleRecord instances for testing."""
    samples = []
    # 2 Real
    for i in range(2):
        samples.append(
            SampleRecord(
                sample_id=f"real_{i}",
                path=f"path/to/real_{i}.png",
                label=0,
                generator="real",
                generator_id=0,
                image=synthetic_rgb_image.copy(),
            )
        )
    # 2 AI from adm
    for i in range(2):
        samples.append(
            SampleRecord(
                sample_id=f"ai_adm_{i}",
                path=f"path/to/adm_{i}.png",
                label=1,
                generator="adm",
                generator_id=1,
                image=synthetic_rgb_image.copy(),
            )
        )
    return samples


# ---------------------------------------------------------------------------
# 1. Transformation Correctness & Validation
# ---------------------------------------------------------------------------

def test_jpeg_compression_correctness(synthetic_rgb_image):
    """Verify JPEG compression alters high-frequency noise and returns valid RGB."""
    compressed_95 = apply_jpeg_compression(synthetic_rgb_image, quality=95)
    compressed_30 = apply_jpeg_compression(synthetic_rgb_image, quality=30)

    assert compressed_95.size == synthetic_rgb_image.size
    assert compressed_95.mode == "RGB"
    assert compressed_30.size == synthetic_rgb_image.size
    assert compressed_30.mode == "RGB"

    arr_orig = np.asarray(synthetic_rgb_image, dtype=np.float32)
    arr_95 = np.asarray(compressed_95, dtype=np.float32)
    arr_30 = np.asarray(compressed_30, dtype=np.float32)

    mae_95 = np.mean(np.abs(arr_95 - arr_orig))
    mae_30 = np.mean(np.abs(arr_30 - arr_orig))

    # Lower quality must produce strictly higher distortion
    assert mae_30 > mae_95 > 0.0

    # Invalid qualities
    with pytest.raises(ValueError, match="JPEG quality"):
        apply_jpeg_compression(synthetic_rgb_image, quality=0)
    with pytest.raises(ValueError, match="JPEG quality"):
        apply_jpeg_compression(synthetic_rgb_image, quality=101)


def test_downscale_restore_correctness(synthetic_rgb_image):
    """Verify downscale + bilinear restoration preserves original dimensions."""
    restored_75 = apply_downscale_restore(synthetic_rgb_image, scale=0.75)
    restored_50 = apply_downscale_restore(synthetic_rgb_image, scale=0.50)
    restored_25 = apply_downscale_restore(synthetic_rgb_image, scale=0.25)

    assert restored_75.size == synthetic_rgb_image.size
    assert restored_50.size == synthetic_rgb_image.size
    assert restored_25.size == synthetic_rgb_image.size
    assert restored_25.mode == "RGB"

    arr_orig = np.asarray(synthetic_rgb_image, dtype=np.float32)
    arr_50 = np.asarray(restored_50, dtype=np.float32)
    arr_25 = np.asarray(restored_25, dtype=np.float32)

    mae_50 = np.mean(np.abs(arr_50 - arr_orig))
    mae_25 = np.mean(np.abs(arr_25 - arr_orig))

    # Lower scale factor destroys more high-frequency texture
    assert mae_25 > mae_50 > 0.0

    # Invalid scale factors
    with pytest.raises(ValueError, match="Scale factor"):
        apply_downscale_restore(synthetic_rgb_image, scale=0.0)
    with pytest.raises(ValueError, match="Scale factor"):
        apply_downscale_restore(synthetic_rgb_image, scale=1.5)


def test_gaussian_blur_correctness(synthetic_rgb_image):
    """Verify Gaussian blur filters high frequencies and handles edge cases."""
    blurred_05 = apply_gaussian_blur(synthetic_rgb_image, sigma=0.5)
    blurred_20 = apply_gaussian_blur(synthetic_rgb_image, sigma=2.0)
    blurred_00 = apply_gaussian_blur(synthetic_rgb_image, sigma=0.0)

    assert blurred_05.size == synthetic_rgb_image.size
    assert blurred_20.size == synthetic_rgb_image.size
    assert blurred_00.size == synthetic_rgb_image.size

    arr_orig = np.asarray(synthetic_rgb_image, dtype=np.float32)
    arr_05 = np.asarray(blurred_05, dtype=np.float32)
    arr_20 = np.asarray(blurred_20, dtype=np.float32)

    mae_05 = np.mean(np.abs(arr_05 - arr_orig))
    mae_20 = np.mean(np.abs(arr_20 - arr_orig))

    assert mae_20 > mae_05 > 0.0
    # Sigma 0.0 returns identical copy
    np.testing.assert_array_equal(np.asarray(blurred_00), np.asarray(synthetic_rgb_image))

    with pytest.raises(ValueError, match="sigma must be >= 0.0"):
        apply_gaussian_blur(synthetic_rgb_image, sigma=-0.5)


def test_screenshot_proxy_correctness(synthetic_rgb_image):
    """Verify screenshot proxy applies downscale-restore + JPEG re-save."""
    screenshot = apply_screenshot_proxy(synthetic_rgb_image, scale=0.70, quality=80)
    assert screenshot.size == synthetic_rgb_image.size
    assert screenshot.mode == "RGB"

    arr_orig = np.asarray(synthetic_rgb_image, dtype=np.float32)
    arr_shot = np.asarray(screenshot, dtype=np.float32)
    assert np.mean(np.abs(arr_shot - arr_orig)) > 0.0


def test_photometric_edits_correctness(synthetic_rgb_image):
    """Verify brightness and contrast enhancements adjust pixel luminance."""
    brighter = apply_photometric_edit(synthetic_rgb_image, "brightness", 1.15)
    darker = apply_photometric_edit(synthetic_rgb_image, "brightness", 0.85)
    high_contrast = apply_photometric_edit(synthetic_rgb_image, "contrast", 1.15)
    low_contrast = apply_photometric_edit(synthetic_rgb_image, "contrast", 0.85)

    arr_orig = np.asarray(synthetic_rgb_image, dtype=np.float32)
    arr_brighter = np.asarray(brighter, dtype=np.float32)
    arr_darker = np.asarray(darker, dtype=np.float32)

    # Brighter image has strictly higher mean pixel intensity
    assert np.mean(arr_brighter) > np.mean(arr_orig) > np.mean(arr_darker)

    # Invalid edit types and factors
    with pytest.raises(ValueError, match="Unsupported edit_type"):
        apply_photometric_edit(synthetic_rgb_image, "invalid_type", 1.0)
    with pytest.raises(ValueError, match="factor must be > 0.0"):
        apply_photometric_edit(synthetic_rgb_image, "brightness", -0.5)


# ---------------------------------------------------------------------------
# 2. Determinism, Immutability & Dimensions
# ---------------------------------------------------------------------------

def test_transformation_determinism(synthetic_rgb_image):
    """Verify all transformations produce bitwise identical output across repeated runs."""
    t1 = apply_screenshot_proxy(synthetic_rgb_image, scale=0.70, quality=80)
    t2 = apply_screenshot_proxy(synthetic_rgb_image, scale=0.70, quality=80)
    np.testing.assert_array_equal(np.asarray(t1), np.asarray(t2))

    j1 = apply_jpeg_compression(synthetic_rgb_image, quality=50)
    j2 = apply_jpeg_compression(synthetic_rgb_image, quality=50)
    np.testing.assert_array_equal(np.asarray(j1), np.asarray(j2))


def test_input_image_immutability(synthetic_rgb_image):
    """Verify that source PIL Image is never modified in-place by any transformation."""
    orig_copy = synthetic_rgb_image.copy()
    orig_bytes = synthetic_rgb_image.tobytes()

    _ = apply_jpeg_compression(synthetic_rgb_image, 30)
    _ = apply_downscale_restore(synthetic_rgb_image, 0.25)
    _ = apply_gaussian_blur(synthetic_rgb_image, 2.0)
    _ = apply_screenshot_proxy(synthetic_rgb_image, 0.70, 80)
    _ = apply_photometric_edit(synthetic_rgb_image, "brightness", 1.15)

    assert synthetic_rgb_image.tobytes() == orig_bytes
    np.testing.assert_array_equal(np.asarray(synthetic_rgb_image), np.asarray(orig_copy))


def test_output_pixel_range_and_mode(synthetic_rgb_image):
    """Verify output pixels are strictly in [0, 255] and mode is RGB."""
    conditions = get_standard_robustness_conditions()
    for cond in conditions:
        out = cond.transform_fn(synthetic_rgb_image)
        assert out.mode == "RGB"
        assert out.size == synthetic_rgb_image.size
        arr = np.asarray(out)
        assert arr.dtype == np.uint8
        assert 0 <= arr.min() and arr.max() <= 255


# ---------------------------------------------------------------------------
# 3. Path Safety & Test-Set Isolation
# ---------------------------------------------------------------------------

def test_strict_held_out_test_directory_rejection():
    """Verify that targeting the official held-out test set is rejected immediately."""
    forbidden = r"C:\Datasets\SignalScope\test"
    with pytest.raises(TestSetContaminationError, match="FATAL TEST CONTAMINATION DETECTED"):
        validate_path_safety(forbidden)

    forbidden_sub = r"C:\Datasets\SignalScope\test\adm_imagenet"
    with pytest.raises(TestSetContaminationError):
        validate_path_safety(forbidden_sub)


# ---------------------------------------------------------------------------
# 4. Metric Math & Verdict Flip Verification
# ---------------------------------------------------------------------------

def test_metric_math_accuracy():
    """Verify probability shift, MAPS, and flip rates against synthetic ground truth."""
    # 4 samples: 2 Real (0), 2 AI (1)
    y_true = np.array([0, 0, 1, 1])
    generator_tags = ["real", "real", "adm", "midjourney"]

    # Baseline clean scores
    clean_scores = np.array([0.10, 0.20, 0.80, 0.90])
    # Degraded scores:
    # Sample 0: 0.10 -> 0.15 (no flip, Real stays Real)
    # Sample 1: 0.20 -> 0.60 (FLIP! Real becomes AI: False Alarm!)
    # Sample 2: 0.80 -> 0.40 (FLIP! AI becomes Real: Evasion!)
    # Sample 3: 0.90 -> 0.85 (no flip, AI stays AI)
    deg_scores = np.array([0.15, 0.60, 0.40, 0.85])

    metrics = compute_robustness_metrics(
        clean_scores=clean_scores,
        degraded_scores=deg_scores,
        y_true=y_true,
        generator_tags=generator_tags,
        threshold=0.50,
    )

    # Delta p: [+0.05, +0.40, -0.40, -0.05] -> mean = 0.00
    assert metrics["mean_probability_shift"] == 0.00
    # MAPS: [0.05, 0.40, 0.40, 0.05] / 4 = 0.90 / 4 = 0.225
    assert metrics["maps"] == 0.225

    # Flips: sample 1 and sample 2 flipped -> 2 / 4 = 0.50
    assert metrics["total_flips"] == 2
    assert metrics["verdict_flip_rate"] == 0.50

    # Evasion: AI sample 2 flipped -> 1 evasion out of 2 AI = 0.50
    assert metrics["n_evasions"] == 1
    assert metrics["evasion_rate"] == 0.50

    # False Alarm: Real sample 1 flipped -> 1 false alarm out of 2 Real = 0.50
    assert metrics["n_false_alarms"] == 1
    assert metrics["false_alarm_rate"] == 0.50

    # Clean accuracy was 4/4 = 1.0; degraded accuracy is 2/4 = 0.5
    assert metrics["clean_perf"]["accuracy"] == 1.0
    assert metrics["degraded_perf"]["accuracy"] == 0.5
    assert metrics["delta_accuracy"] == -0.5


# ---------------------------------------------------------------------------
# 5. Lightweight End-to-End Execution with Mock Inference
# ---------------------------------------------------------------------------

def test_lightweight_end_to_end_mock_evaluation(synthetic_mock_samples, tmp_path):
    """Verify full evaluation pipeline runs end-to-end and creates valid artifacts."""
    conditions = get_standard_robustness_conditions()

    # Deterministic mock predict function
    def mock_predict(img: Image.Image) -> float:
        arr = np.asarray(img, dtype=float)
        val = float(np.mean(arr)) / 255.0
        return min(max(val, 0.01), 0.99)

    clean_cond = conditions[0]
    clean_scores = evaluate_samples_on_model(synthetic_mock_samples, clean_cond, mock_predict)
    assert len(clean_scores) == len(synthetic_mock_samples)

    y_true = np.array([s.label for s in synthetic_mock_samples])
    tags = [s.generator for s in synthetic_mock_samples]

    summary_by_cond = {}
    for cond in conditions:
        deg_scores = evaluate_samples_on_model(synthetic_mock_samples, cond, mock_predict)
        m = compute_robustness_metrics(clean_scores, deg_scores, y_true, tags, threshold=0.50)
        summary_by_cond[cond.name] = {
            "name": cond.name,
            "category": cond.category,
            "parameter_value": cond.parameter_value,
            **m,
        }

    results_data = {
        "metadata": {
            "model": "vit_base_patch16_224",
            "checkpoint": "mock_weights",
            "total_samples": len(synthetic_mock_samples),
            "sample_composition": {"real": 2, "ai_total": 2},
            "decision_threshold": 0.50,
        },
        "conditions": [c.name for c in conditions],
        "summary_by_condition": summary_by_cond,
    }

    out_dir = tmp_path / "robustness_artifacts"
    rep_file = tmp_path / "report" / "robustness_report.md"

    json_p, csv_p, md_p = save_robustness_artifacts(
        results_data,
        output_dir=out_dir,
        report_path=rep_file,
    )

    assert json_p.exists()
    assert csv_p.exists()
    assert md_p.exists()

    # Verify JSON content
    with open(json_p, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        assert "summary_by_condition" in loaded
        assert "clean" in loaded["summary_by_condition"]
        assert "screenshot_proxy" in loaded["summary_by_condition"]

    # Verify CSV rows
    with open(csv_p, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 1 + len(conditions)  # Header + 16 conditions
