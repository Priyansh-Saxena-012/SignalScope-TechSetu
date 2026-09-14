"""SignalScope Model Training Scaffold.

Provides modular training and checkpointing routines:
- Swappable backbone initialization (ViT-Base/16, ConvNeXt-Tiny)
- Configurable optimizer (AdamW), learning rate, weight decay, and schedulers
- Mixed precision support (torch.cuda.amp) for accelerated GPU training on Colab
- Checkpoint management (saves best checkpoint, config copy, and FP16 weights)
- Device selection (auto-detects CUDA/VRAM, falls back to CPU)
- Strict reproducibility seed control

NOTE: This is a training scaffold. It does NOT execute training without a configured dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is in sys.path when executed directly (python src/model/train.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.backbone import build_classifier
from src.data.dataset import SignalScopeDataset, create_development_splits
from src.data.path_safety import validate_path_safety
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.evaluation.evaluate import compute_metrics
from src.utils.config import load_config, save_config


def set_seed(seed: int = 42) -> None:
    """Enforce reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_setting: str = "auto") -> Tuple[torch.device, Dict[str, Any]]:
    """Determine compute device and record hardware diagnostics."""
    info: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device_requested": device_setting,
    }

    if device_setting == "cuda" or (device_setting == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
        info["device"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    else:
        device = torch.device("cpu")
        info["device"] = "cpu"
        info["gpu_name"] = None
        info["vram_gb"] = 0.0

    return device, info


class SignalScopeTrainer:
    """Trainer orchestrator for SignalScope Real vs. AI image classification.

    Parameters
    ----------
    config : dict
        Experiment configuration dictionary.
    train_loader : DataLoader, optional
    val_loader : DataLoader, optional
    """
    def __init__(
        self,
        config: Dict[str, Any],
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 0. Test-Set Isolation Validation
        data_cfg = config.get("data", {})
        held_out_dir = data_cfg.get("held_out_test_dir")
        for key in ["data_dir", "train_data_path", "val_data_path", "unseen_data_path"]:
            p = data_cfg.get(key)
            if p:
                validate_path_safety(
                    p,
                    protected_paths=held_out_dir,
                    context_desc=f"trainer configuration 'data.{key}'",
                )

        # 1. Reproducibility & Output Directory
        self.seed = config["experiment"].get("seed", 42)
        set_seed(self.seed)

        self.output_dir = Path(config["experiment"].get("output_dir", "experiments/default_run"))
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 2. Compute Device
        self.device, self.device_info = resolve_device(config["training"].get("device", "auto"))

        # 3. Model Architecture
        model_cfg = config["model"]
        self.model = build_classifier(
            backbone_name=model_cfg.get("backbone", "vit_base_patch16_224"),
            pretrained=model_cfg.get("pretrained", True),
            drop_rate=model_cfg.get("dropout", 0.2),
        )
        self.model.to(self.device)

        # 4. Criterion & Optimizer
        train_cfg = config["training"]
        self.criterion = nn.BCEWithLogitsLoss()

        lr = float(train_cfg.get("learning_rate", 1e-4))
        weight_decay = float(train_cfg.get("weight_decay", 1e-2))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # 5. Mixed Precision Scaler
        use_amp = bool(train_cfg.get("mixed_precision", True)) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.use_amp = use_amp

        # 6. Training History
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_roc_auc": [],
            "val_accuracy": [],
            "val_macro_f1": [],
        }
        self.best_val_auc = 0.0

        # Save copy of configuration into output directory
        save_config(self.config, str(self.output_dir / "config.yaml"))

    def save_checkpoint(self, epoch: int, val_auc: float, is_best: bool = False) -> str:
        """Save model checkpoint and export FP16 weights."""
        checkpoint_data = {
            "epoch": epoch,
            "backbone": self.model.backbone_name,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_roc_auc": val_auc,
            "config": self.config,
        }

        # 1. Save standard full-precision checkpoint
        ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:02d}.pth"
        torch.save(checkpoint_data, ckpt_path)

        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pth"
            torch.save(checkpoint_data, best_path)

            # 2. Export FP16 weights for lightweight distribution (<100 MB)
            fp16_weights_path = self.checkpoint_dir / "model_weights_fp16.pth"
            fp16_state = {k: v.half() if v.is_floating_point() else v for k, v in self.model.state_dict().items()}
            torch.save(fp16_state, fp16_weights_path)

        return str(ckpt_path)

    def evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        """Compute validation loss and core metrics across a DataLoader."""
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels, _ in loader:
                images = images.to(self.device)
                labels = labels.to(self.device).float()

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                total_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(logits).cpu().numpy()

                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0.0
        metrics = compute_metrics(all_labels, all_probs, threshold=0.5)
        metrics["loss"] = round(avg_loss, 4)
        return metrics

    def fit(self) -> Dict[str, Any]:
        """Execute training loop over configured epochs with early stopping."""
        if self.train_loader is None or self.val_loader is None:
            raise ValueError("Cannot train: train_loader or val_loader is not set.")

        epochs = self.config["training"].get("epochs", 5)
        patience = self.config["training"].get("early_stopping_patience", 2)
        no_improvement_count = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0

            for images, labels, _ in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).float()

                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss += loss.item() * images.size(0)

            epoch_train_loss = running_loss / len(self.train_loader.dataset)
            val_metrics = self.evaluate(self.val_loader)
            current_auc = val_metrics.get("roc_auc") or 0.0

            self.history["train_loss"].append(round(epoch_train_loss, 4))
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_roc_auc"].append(current_auc)
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_macro_f1"].append(val_metrics["macro_f1"])

            is_best = current_auc > self.best_val_auc
            if is_best:
                self.best_val_auc = current_auc
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            self.save_checkpoint(epoch, current_auc, is_best=is_best)

            # Early stopping check
            if no_improvement_count >= patience:
                break

        # Save history log
        with open(self.output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        return {
            "best_val_roc_auc": self.best_val_auc,
            "final_epoch": epoch,
            "history": self.history,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="SignalScope Training Entry Point")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to experiment configuration YAML",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg["data"].get("data_dir")

    if not data_dir or not os.path.exists(data_dir):
        print("=" * 65)
        print("[NOTICE] Training Scaffold Initialized Successfully.")
        print(f"Target Backbone: {cfg['model']['backbone']}")
        print(f"Output Directory: {cfg['experiment']['output_dir']}")
        print("Dataset path is not yet configured or mounted on disk.")
        print("Once GenImage is mounted, specify data.data_dir in the configuration.")
        print("=" * 65)
        sys.exit(0)

    data_cfg = cfg["data"]
    splits = create_development_splits(
        data_dir=data_dir,
        train_ratio=data_cfg.get("train_ratio", 0.70),
        val_seen_ratio=data_cfg.get("val_seen_ratio", 0.15),
        val_unseen_ratio=data_cfg.get("val_unseen_ratio", 0.15),
        seed=cfg["experiment"].get("seed", 42),
        unseen_generators=data_cfg.get("unseen_generators"),
        held_out_test_dir=data_cfg.get("held_out_test_dir"),
    )
    print("=" * 65)
    print(f"Split Strategy: {splits['split_strategy']}")
    print(json.dumps(splits["split_summary"], indent=2))
    print("=" * 65)

    image_size = cfg["model"].get("image_size", 224)
    train_dataset = SignalScopeDataset(splits["train"], transform=get_train_transforms(image_size))
    val_dataset = SignalScopeDataset(splits["internal_val_seen"], transform=get_eval_transforms(image_size))

    train_cfg = cfg["training"]
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 2)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    trainer = SignalScopeTrainer(cfg, train_loader=train_loader, val_loader=val_loader)
    results = trainer.fit()
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
