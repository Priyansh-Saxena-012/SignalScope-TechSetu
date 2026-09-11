r"""Unit tests for SignalScope Test-Set Isolation and Path Safety Validation.

Tests:
1. Rejection when training root is exactly the held-out test directory (C:\Datasets\SignalScope\test).
2. Rejection when training root is a parent/ancestor directory (C:\Datasets\SignalScope, C:\Datasets, C:\).
3. Rejection when training root is a subdirectory of the test directory (e.g. C:\Datasets\SignalScope\test\adm).
4. Acceptance when training root is a legitimate separate directory (e.g. C:\Datasets\SignalScope_Train, data/train).
5. Configurable held-out test directory support across custom paths and environment variables.
6. Integration with create_development_splits(): rejects test and parent paths before crawling.
7. Integration with config loading and validation: rejects unsafe data_dir entries.
8. Integration with SignalScopeTrainer: rejects unsafe config data paths.
9. Verification of error message content and actionable instructions.
"""

import os
from pathlib import Path
import pytest

from model.train import SignalScopeTrainer
from src.data.dataset import create_development_splits
from src.data.path_safety import (
    DEFAULT_PROTECTED_TEST_DIRS,
    TestSetContaminationError,
    get_protected_test_dirs,
    is_path_contaminated,
    normalize_path,
    validate_path_safety,
)
from src.utils.config import get_default_config, validate_config


# --- 1. Core Path Safety Logic Tests ---

def test_rejection_exact_test_directory():
    """Verify that pointing directly to the official held-out test set is rejected."""
    exact_path = r"C:\Datasets\SignalScope\test"
    with pytest.raises(TestSetContaminationError) as exc_info:
        validate_path_safety(exact_path)

    err_msg = str(exc_info.value)
    assert "[FATAL TEST CONTAMINATION DETECTED]" in err_msg
    assert "is identical to" in err_msg
    assert "100,000-image" in err_msg


def test_rejection_exact_test_directory_variations():
    """Verify case-insensitivity, forward slashes, and redundant slashes are normalized and rejected."""
    variations = [
        "c:/datasets/signalscope/test",
        "C:/Datasets/SignalScope/test/",
        r"c:\datasets\signalscope\test\\",
        r"C:\DATASETS\SIGNALSCOPE\TEST",
        r"C:\Datasets\SignalScope\test\..\test",
    ]
    for p in variations:
        with pytest.raises(TestSetContaminationError):
            validate_path_safety(p)


def test_rejection_parent_of_test_directory():
    """Verify that any parent directory containing the held-out test set is rejected."""
    parents = [
        r"C:\Datasets\SignalScope",
        "c:/datasets/signalscope",
        r"C:\Datasets",
        r"C:\\",
    ]
    for p in parents:
        with pytest.raises(TestSetContaminationError) as exc_info:
            validate_path_safety(p)

        err_msg = str(exc_info.value)
        assert "[FATAL TEST CONTAMINATION DETECTED]" in err_msg
        assert "is a parent/ancestor directory containing" in err_msg


def test_rejection_subdirectory_of_test_directory():
    """Verify that any subfolder inside the held-out test set is rejected."""
    subfolders = [
        r"C:\Datasets\SignalScope\test\adm",
        r"C:\Datasets\SignalScope\test\midjourney\fake",
        "c:/datasets/signalscope/test/0_real",
    ]
    for p in subfolders:
        with pytest.raises(TestSetContaminationError) as exc_info:
            validate_path_safety(p)

        err_msg = str(exc_info.value)
        assert "[FATAL TEST CONTAMINATION DETECTED]" in err_msg
        assert "is located inside" in err_msg


def test_acceptance_legitimate_separate_training_roots():
    """Verify that legitimate separate training roots pass validation without error."""
    legitimate_paths = [
        r"C:\Datasets\SignalScope_Train",
        r"C:\Datasets\SignalScope_Development",
        "data/genimage_train",
        "experiments/datasets/train",
        r"D:\Datasets\SignalScope\train",
    ]
    for p in legitimate_paths:
        # Must not raise an exception
        validate_path_safety(p)
        assert is_path_contaminated(p) is False


def test_empty_or_none_path_handling():
    """Verify None or empty path does not trigger validation failure (handled by caller)."""
    validate_path_safety(None)
    validate_path_safety("")
    assert is_path_contaminated(None) is False
    assert is_path_contaminated("") is False


# --- 2. Configurable Protection Tests (Cross-Platform & Custom Roots) ---

def test_custom_protected_paths_rejection(tmp_path):
    """Verify configurable protection supports arbitrary custom test directories."""
    mock_test = tmp_path / "benchmark" / "eval_test"
    mock_parent = tmp_path / "benchmark"
    mock_separate = tmp_path / "training_data"
    mock_test.mkdir(parents=True)
    mock_separate.mkdir(parents=True)

    # 1. Exact match rejected
    with pytest.raises(TestSetContaminationError):
        validate_path_safety(mock_test, protected_paths=[mock_test])

    # 2. Parent rejected
    with pytest.raises(TestSetContaminationError):
        validate_path_safety(mock_parent, protected_paths=[mock_test])

    # 3. Separate directory accepted
    validate_path_safety(mock_separate, protected_paths=[mock_test])
    assert is_path_contaminated(mock_separate, protected_paths=[mock_test]) is False


def test_environment_variable_override(tmp_path, monkeypatch):
    """Verify SIGNALSCOPE_HELD_OUT_TEST_DIR environment variable adds protected directory."""
    env_test = tmp_path / "env_held_out_test"
    env_parent = tmp_path
    monkeypatch.setenv("SIGNALSCOPE_HELD_OUT_TEST_DIR", str(env_test))

    # Should reject the env-configured test directory and its parent
    with pytest.raises(TestSetContaminationError):
        validate_path_safety(env_test)

    with pytest.raises(TestSetContaminationError):
        validate_path_safety(env_parent)


# --- 3. Integration with Development Splitting Engine ---

def test_create_development_splits_rejects_test_directory():
    """Verify create_development_splits rejects official held-out test path before inspecting/crawling."""
    with pytest.raises(TestSetContaminationError) as exc_info:
        create_development_splits(r"C:\Datasets\SignalScope\test")
    assert "[FATAL TEST CONTAMINATION DETECTED]" in str(exc_info.value)


def test_create_development_splits_rejects_parent_directory():
    r"""Verify create_development_splits rejects parent directory C:\Datasets\SignalScope."""
    with pytest.raises(TestSetContaminationError) as exc_info:
        create_development_splits(r"C:\Datasets\SignalScope")
    assert "[FATAL TEST CONTAMINATION DETECTED]" in str(exc_info.value)


def test_create_development_splits_accepts_isolated_mock_dataset(tmp_path):
    """Verify create_development_splits works normally with an isolated mock directory."""
    from PIL import Image

    mock_dir = tmp_path / "isolated_train_dataset"
    (mock_dir / "real").mkdir(parents=True)
    (mock_dir / "fake").mkdir(parents=True)

    for i in range(5):
        Image.new("RGB", (32, 32), (1, 2, 3)).save(mock_dir / "real" / f"r_{i}.jpg")
        Image.new("RGB", (32, 32), (4, 5, 6)).save(mock_dir / "fake" / f"f_{i}.jpg")

    # Should succeed without raising TestSetContaminationError
    splits = create_development_splits(str(mock_dir))
    assert len(splits["train"]) > 0


# --- 4. Integration with Config Validation & Trainer Scaffold ---

def test_validate_config_rejects_contaminated_data_dir():
    """Verify validate_config catches unsafe paths in data.data_dir."""
    cfg = get_default_config()

    # Unsafe 1: Exact test set
    cfg["data"]["data_dir"] = r"C:\Datasets\SignalScope\test"
    with pytest.raises(TestSetContaminationError):
        validate_config(cfg)

    # Unsafe 2: Parent directory
    cfg["data"]["data_dir"] = r"C:\Datasets\SignalScope"
    with pytest.raises(TestSetContaminationError):
        validate_config(cfg)

    # Safe: Separate training directory
    cfg["data"]["data_dir"] = r"C:\Datasets\SignalScope_Train"
    assert validate_config(cfg) is True


def test_trainer_scaffold_rejects_contaminated_config():
    """Verify SignalScopeTrainer initialization rejects contaminated dataset paths."""
    cfg = get_default_config()
    cfg["data"]["data_dir"] = r"C:\Datasets\SignalScope"
    cfg["model"]["pretrained"] = False

    with pytest.raises(TestSetContaminationError):
        SignalScopeTrainer(cfg)
