"""SignalScope Vision Classifier & Swappable Backbone Architecture.

Provides a unified interface for binary Real vs. AI image classification supporting:
- Primary Candidate: ViT-Base/16 (vit_base_patch16_224)
- Fallback Candidate: ConvNeXt-Tiny (convnext_tiny)
- Optional Lightweight Alternate: EfficientNet-V2-S (efficientnet_v2_s)

The interface guarantees identical forward signature and output shapes across all backbones,
enabling zero-code-change switching between models.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import timm
import torch
import torch.nn as nn

SUPPORTED_BACKBONES = {
    "vit_base_patch16_224": "Vision Transformer Base (16x16 patch, 224x224 input) - Primary",
    "convnext_tiny": "ConvNeXt Tiny - Modern Convolutional Architecture (Fallback)",
    "efficientnet_v2_s": "EfficientNet-V2 Small - Lightweight Convolutional Architecture",
}


class SignalScopeClassifier(nn.Module):
    """Binary Real-vs-AI Image Classifier with swappable vision backbone.

    Parameters
    ----------
    backbone_name : str, default="vit_base_patch16_224"
        Name of the timm backbone architecture.
    pretrained : bool, default=False
        Whether to load ImageNet pretrained weights.
    drop_rate : float, default=0.2
        Dropout probability before final binary classification head.
    """
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = False,
        drop_rate: float = 0.2,
    ):
        super().__init__()
        if backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: '{backbone_name}'. "
                f"Supported backbones: {list(SUPPORTED_BACKBONES.keys())}"
            )

        self.backbone_name = backbone_name
        self.drop_rate = drop_rate

        # Initialize backbone with num_classes=1 for binary classification logit
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=1,
            drop_rate=drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through vision backbone and binary classification head.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input images of shape (B, 3, H, W).

        Returns
        -------
        torch.Tensor
            1D tensor of raw logits of shape (B,) representing log-odds of being AI-generated.
        """
        logits = self.model(x)  # Shape (B, 1)
        return logits.squeeze(-1)  # Shape (B,)

    @torch.no_grad()
    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Compute uncalibrated sigmoid probabilities for an input batch.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input images of shape (B, 3, H, W).

        Returns
        -------
        torch.Tensor
            1D tensor of probabilities of shape (B,) where p >= 0.5 suggests AI-generated.
        """
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def get_parameter_summary(self) -> Dict[str, Any]:
        """Compute total and trainable parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "backbone": self.backbone_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "estimated_fp32_mb": round((total_params * 4) / (1024 * 1024), 2),
            "estimated_fp16_mb": round((total_params * 2) / (1024 * 1024), 2),
        }


def build_classifier(
    backbone_name: str = "vit_base_patch16_224",
    pretrained: bool = False,
    drop_rate: float = 0.2,
) -> SignalScopeClassifier:
    """Factory helper to build a SignalScopeClassifier instance."""
    return SignalScopeClassifier(
        backbone_name=backbone_name,
        pretrained=pretrained,
        drop_rate=drop_rate,
    )
