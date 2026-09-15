"""SignalScope Experiment Configuration Management.

Provides YAML configuration loading, validation, and serialization.
Enables defining data paths, backbone architectures, training hyperparameters,
and output directories without hardcoding paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment": {
        "name": "signalscope_experiment",
        "output_dir": "experiments/run_default",
        "seed": 42,
        "description": "SignalScope binary real-vs-ai training run",
    },
    "model": {
        "backbone": "vit_base_patch16_224",  # Primary candidate
        "fallback_backbone": "convnext_tiny",
        "pretrained": True,
        "dropout": 0.2,
        "image_size": 224,
    },
    "data": {
        "data_dir": None,               # Real path to be supplied when dataset arrives
        "held_out_test_dir": r"C:\Datasets\SignalScope\test",  # Official evaluation benchmark (STRICTLY ISOLATED)
        "train_data_path": r"C:\Datasets\SignalScope_Train\data\train-*.parquet",
        "val_data_path": r"C:\Datasets\SignalScope_Train\data\validation-*.parquet",
        "unseen_data_path": None,       # Optional explicit unseen-generator directory
        "train_ratio": 0.70,
        "val_seen_ratio": 0.15,
        "val_unseen_ratio": 0.15,
        "unseen_generators": None,      # List of generators reserved for unseen test
    },
    "training": {
        "batch_size": 32,
        "epochs": 5,
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "warmup_epochs": 1,
        "mixed_precision": True,
        "early_stopping_patience": 2,
        "num_workers": 0,
        "max_batches_per_epoch": None,  # Optional limit for local CPU smoke runs
        "log_interval": 50,             # Periodic batch logging frequency
        "device": "auto",               # "cuda", "cpu", or "auto"
        "num_workers": 2,               # DataLoader worker processes
    },
    "evaluation": {
        "default_threshold": 0.5,
        "save_confusion_matrix": True,
        "save_roc_curve": True,
    },
}


def get_default_config() -> Dict[str, Any]:
    """Return a deep copy of the default configuration."""
    import copy
    return copy.deepcopy(DEFAULT_CONFIG)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate a YAML experiment configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    # Merge user configuration onto default configuration
    merged = get_default_config()
    for section, values in user_config.items():
        if section in merged and isinstance(values, dict):
            merged[section].update(values)
        else:
            merged[section] = values

    validate_config(merged)
    return merged


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration dictionary to a YAML file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate structure and safety of experiment configuration dictionary."""
    from src.data.path_safety import validate_path_safety

    required_sections = ["experiment", "model", "data", "training"]
    for sec in required_sections:
        if sec not in config:
            raise ValueError(f"Missing required configuration section: '{sec}'")

    if not config["model"].get("backbone"):
        raise ValueError("model.backbone cannot be empty.")

    if config["training"].get("epochs", 0) <= 0:
        raise ValueError("training.epochs must be positive.")

    # Validate test-set isolation for configured dataset paths
    data_cfg = config.get("data", {})
    held_out_dir = data_cfg.get("held_out_test_dir")
    for key in ["data_dir", "train_data_path", "val_data_path", "unseen_data_path"]:
        path_val = data_cfg.get(key)
        if path_val:
            validate_path_safety(
                path_val,
                protected_paths=held_out_dir,
                context_desc=f"configuration path 'data.{key}'",
            )

    return True
