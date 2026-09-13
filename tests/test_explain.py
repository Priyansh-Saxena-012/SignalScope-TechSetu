"""Unit tests for SignalScope ViT Contrast-Centered Patch Decision Saliency.

Verifies:
- Output shape is exactly (224, 224)
- Values are finite and normalized in [0, 1]
- Forward and backward hooks are removed cleanly (no memory leaks)
- Normal evaluation and prediction behavior are not altered
- Preprocessing crop geometry matches model evaluation window
- Unsupported architectures raise descriptive ValueError
- Tests run with lightweight uninitialized models without depending on 1GB production weights
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn

from model.backbone import build_classifier
from model.explain import (
    ViTPatchGradCAM,
    apply_colormap_jet,
    generate_explanation,
    get_evaluation_crop,
    overlay_saliency,
)
from src.data.transforms import get_eval_transforms


@pytest.fixture
def mock_classifier() -> nn.Module:
    """Provide a lightweight, uninitialized ViT-Base/16 classifier."""
    classifier = build_classifier("vit_base_patch16_224", pretrained=False)
    classifier.eval()
    return classifier


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Provide a synthetic 3-channel PIL image with varied dimensions."""
    data = (np.random.RandomState(42).rand(300, 400, 3) * 255).astype(np.uint8)
    return Image.fromarray(data, mode="RGB")


def test_saliency_map_shape_and_range(mock_classifier, sample_pil_image):
    """Verify that generated saliency map is (224, 224), finite, and bounded in [0, 1]."""
    transform = get_eval_transforms(224)
    tensor = transform(sample_pil_image)

    explainer = ViTPatchGradCAM(mock_classifier)
    saliency = explainer.generate_saliency_map(tensor)

    assert isinstance(saliency, np.ndarray)
    assert saliency.shape == (224, 224)
    assert np.all(np.isfinite(saliency))
    assert 0.0 <= saliency.min()
    assert saliency.max() <= 1.0


def test_hooks_are_cleaned_up(mock_classifier, sample_pil_image):
    """Verify that forward and backward hooks are removed after execution."""
    target_layer = mock_classifier.model.blocks[-1].norm1

    initial_forward_hooks = len(target_layer._forward_hooks)
    initial_backward_hooks = len(target_layer._backward_hooks)

    transform = get_eval_transforms(224)
    tensor = transform(sample_pil_image)

    explainer = ViTPatchGradCAM(mock_classifier)
    _ = explainer.generate_saliency_map(tensor)

    # Hooks must be removed
    assert len(target_layer._forward_hooks) == initial_forward_hooks
    assert len(target_layer._backward_hooks) == initial_backward_hooks


def test_hooks_cleaned_up_on_exception(mock_classifier):
    """Verify that hooks are removed even if forward or backward fails."""
    target_layer = mock_classifier.model.blocks[-1].norm1
    initial_f = len(target_layer._forward_hooks)
    initial_b = len(target_layer._backward_hooks)

    explainer = ViTPatchGradCAM(mock_classifier)

    # Pass invalid tensor shape to trigger an error during execution
    invalid_tensor = torch.randn(1, 1, 10, 10)
    with pytest.raises(Exception):
        explainer.generate_saliency_map(invalid_tensor)

    assert len(target_layer._forward_hooks) == initial_f
    assert len(target_layer._backward_hooks) == initial_b


def test_explanation_preserves_prediction_behavior(mock_classifier, sample_pil_image):
    """Verify that running an explanation does not alter the model's normal predictions."""
    transform = get_eval_transforms(224)
    tensor = transform(sample_pil_image).unsqueeze(0)

    with torch.no_grad():
        pred_before = mock_classifier(tensor).clone()

    explainer = ViTPatchGradCAM(mock_classifier)
    _ = explainer.generate_saliency_map(tensor)

    with torch.no_grad():
        pred_after = mock_classifier(tensor).clone()

    assert torch.allclose(pred_before, pred_after, atol=1e-6)
    assert not mock_classifier.training  # Preserved eval mode


def test_unsupported_model_raises_value_error():
    """Verify that attempting to create a ViTPatchGradCAM with an invalid model raises ValueError."""
    dummy_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    with pytest.raises(ValueError, match="does not contain transformer blocks"):
        ViTPatchGradCAM(dummy_model)


def test_evaluation_crop_geometry():
    """Verify get_evaluation_crop extracts central 224x224 crop across various aspect ratios."""
    # Landscape
    landscape = Image.new("RGB", (640, 480))
    crop_land = get_evaluation_crop(landscape, 224)
    assert crop_land.size == (224, 224)

    # Portrait
    portrait = Image.new("RGB", (300, 600))
    crop_port = get_evaluation_crop(portrait, 224)
    assert crop_port.size == (224, 224)

    # Square
    square = Image.new("RGB", (512, 512))
    crop_sq = get_evaluation_crop(square, 224)
    assert crop_sq.size == (224, 224)


def test_overlay_saliency():
    """Verify overlay_saliency generates a valid RGB PIL Image of size (224, 224)."""
    eval_crop = Image.new("RGB", (224, 224), color=(100, 100, 100))
    saliency = np.random.rand(224, 224).astype(np.float32)

    overlay = overlay_saliency(eval_crop, saliency, alpha=0.5)
    assert isinstance(overlay, Image.Image)
    assert overlay.size == (224, 224)
    assert overlay.mode == "RGB"


def test_generate_explanation_pipeline(mock_classifier, sample_pil_image):
    """Verify full end-to-end generate_explanation pipeline."""
    saliency_map, eval_crop, overlay_img = generate_explanation(sample_pil_image, mock_classifier)

    assert saliency_map.shape == (224, 224)
    assert 0.0 <= saliency_map.min() <= saliency_map.max() <= 1.0
    assert eval_crop.size == (224, 224)
    assert overlay_img.size == (224, 224)
