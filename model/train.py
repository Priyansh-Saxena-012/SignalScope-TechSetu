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

# Ensure project root is in sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.backbone import build_classifier
from src.data.parquet_dataset import (
    DEFAULT_TINY_GENIMAGE_TRAIN_SOURCE,
    DEFAULT_TINY_GENIMAGE_VAL_SOURCE,
    create_tiny_genimage_datasets,
)
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

        # 5. Learning Rate Scheduler
        scheduler_type = train_cfg.get("scheduler", "cosine")
        epochs = int(train_cfg.get("epochs", 5))
        min_lr = float(train_cfg.get("min_lr", 1e-6))
        if scheduler_type == "cosine":
            self.scheduler: Optional[Any] = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, epochs), eta_min=min_lr
            )
        elif scheduler_type is None or scheduler_type == "none":
            self.scheduler = None
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, epochs), eta_min=min_lr
            )

        # 6. Mixed Precision Scaler
        use_amp = bool(train_cfg.get("mixed_precision", True)) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.use_amp = use_amp

        # 7. Training History
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_roc_auc": [],
            "val_accuracy": [],
            "val_macro_f1": [],
            "learning_rate": [],
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
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
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
        total_val_samples = 0
        max_batches = self.config["training"].get("max_batches_per_epoch")

        with torch.no_grad():
            for batch_idx, (images, labels, _) in enumerate(loader):
                if max_batches and batch_idx >= max_batches:
                    break
                images = images.to(self.device)
                labels = labels.to(self.device).float()

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                total_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(logits).cpu().numpy()

                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())
                total_val_samples += images.size(0)

        avg_loss = total_loss / total_val_samples if total_val_samples > 0 else 0.0
        metrics = compute_metrics(all_labels, all_probs, threshold=0.5)
        metrics["loss"] = round(avg_loss, 4)
        return metrics

    def fit(self) -> Dict[str, Any]:
        """Execute training loop over configured epochs with early stopping and lr scheduling."""
        if self.train_loader is None or self.val_loader is None:
            raise ValueError("Cannot train: train_loader or val_loader is not set.")

        epochs = self.config["training"].get("epochs", 5)
        patience = self.config["training"].get("early_stopping_patience", 2)
        max_batches = self.config["training"].get("max_batches_per_epoch")
        log_interval = self.config["training"].get("log_interval", 50)
        no_improvement_count = 0

        total_train_batches = len(self.train_loader) if hasattr(self.train_loader, "__len__") else 0
        effective_batches = min(total_train_batches, max_batches) if (max_batches and total_train_batches) else (max_batches or total_train_batches)

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            total_train_samples = 0
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["learning_rate"].append(current_lr)

            print(f"\n--- Epoch {epoch}/{epochs} [LR: {current_lr:.6e}] ---")

            for batch_idx, (images, labels, _) in enumerate(self.train_loader):
                if max_batches and batch_idx >= max_batches:
                    break
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
                total_train_samples += images.size(0)

                step_num = batch_idx + 1
                is_log_step = (
                    step_num == 1
                    or step_num % log_interval == 0
                    or (effective_batches and step_num == effective_batches)
                )
                if is_log_step:
                    batch_avg_loss = running_loss / total_train_samples
                    batch_str = f"[{step_num}/{effective_batches}]" if effective_batches else f"[{step_num}]"
                    print(f"  Epoch {epoch} {batch_str} - Running Loss: {batch_avg_loss:.4f}")

            epoch_train_loss = running_loss / total_train_samples if total_train_samples > 0 else 0.0

            # Step scheduler after training epoch
            if self.scheduler is not None:
                self.scheduler.step()

            print(f"  Validating Epoch {epoch}...")
            val_metrics = self.evaluate(self.val_loader)
            raw_auc = val_metrics.get("roc_auc")
            current_auc = 0.0 if (raw_auc is None or (isinstance(raw_auc, float) and np.isnan(raw_auc))) else float(raw_auc)

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

            best_indicator = " (* Best)" if is_best else ""
            print(
                f"  Epoch {epoch} Summary - "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val ROC-AUC: {current_auc:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['macro_f1']:.4f}"
                f"{best_indicator}"
            )

            self.save_checkpoint(epoch, current_auc, is_best=is_best)

            # Early stopping check
            if no_improvement_count >= patience:
                print(f"  Early stopping triggered: no improvement for {patience} consecutive epochs.")
                break

        # Save history log
        with open(self.output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        return {
            "best_val_roc_auc": self.best_val_auc,
            "final_epoch": epoch,
            "history": self.history,
        }


def build_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Construct PyTorch DataLoaders for Tiny-GenImage training and validation."""
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    image_size = model_cfg.get("image_size", 224)
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 0)
    held_out_dir = data_cfg.get("held_out_test_dir")

    train_source = data_cfg.get("train_data_path") or DEFAULT_TINY_GENIMAGE_TRAIN_SOURCE
    val_source = data_cfg.get("val_data_path") or DEFAULT_TINY_GENIMAGE_VAL_SOURCE

    train_transform = get_train_transforms(image_size=image_size)
    val_transform = get_eval_transforms(image_size=image_size)

    train_ds, val_ds = create_tiny_genimage_datasets(
        train_source=train_source,
        val_source=val_source,
        train_transform=train_transform,
        val_transform=val_transform,
        held_out_test_dir=held_out_dir,
    )

    device, _ = resolve_device(train_cfg.get("device", "auto"))
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="SignalScope Training Entry Point")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to experiment configuration YAML",
    )
    parser.add_argument(
        "--max-batches-per-epoch",
        type=int,
        default=None,
        help="Optional limit on batches per epoch for fast verification/smoke runs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override execution device ('cpu', 'cuda', 'auto')",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_batches_per_epoch is not None:
        cfg["training"]["max_batches_per_epoch"] = args.max_batches_per_epoch
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        cfg["training"]["device"] = args.device

    print("=" * 65)
    print("[INFO] Initializing SignalScope Training Pipeline...")
    print(f"Target Backbone: {cfg['model']['backbone']}")
    print(f"Output Directory: {cfg['experiment']['output_dir']}")
    print(f"Requested Device: {cfg['training'].get('device', 'auto')}")
    print(f"Batch Size: {cfg['training'].get('batch_size', 32)}")
    print(f"Epochs: {cfg['training'].get('epochs', 5)}")
    if cfg["training"].get("max_batches_per_epoch"):
        print(f"Controlled Batch Limit: {cfg['training']['max_batches_per_epoch']} batches/epoch")
    print("=" * 65)

    train_loader, val_loader = build_dataloaders(cfg)

    trainer = SignalScopeTrainer(
        config=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    results = trainer.fit()
    print("=" * 65)
    print(f"[SUCCESS] Training completed at epoch {results['final_epoch']}.")
    print(f"Best Validation ROC-AUC: {results['best_val_roc_auc']:.4f}")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
