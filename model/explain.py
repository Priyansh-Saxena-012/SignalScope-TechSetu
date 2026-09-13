"""SignalScope ViT Contrast-Centered Patch Decision Saliency.

Computes gradient-weighted patch activations with contrast centering for the
ViT-Base/16 classifier, providing an experimental visual attribution of which
16x16 image regions contributed above average toward the AI-generated decision logit.

Methodology:
- Target layer: `model.blocks[-1].norm1` in VisionTransformer (vit_base_patch16_224).
- Tokens: Excludes the CLS token (index 0); isolates the 196 spatial patch tokens (indices 1..196).
- Weighting: Channel importance weights alpha_k are calculated by average-pooling
  gradients across the 196 spatial tokens.
- Contrast Centering: In pre-attention LayerNorm representations, tokens share an overall
  spatial DC offset that often makes raw linear projections globally negative.
  Spatial centering shifts the baseline to isolate relative, above-average patch contributions.
- Rectification & Upsampling: Applies ReLU to isolate positive relative contributions
  toward the AI logit, reshapes the 196 patch values into a 14x14 grid, and bilinearly
  upsamples to (224, 224).
- Normalization: Scaled to [0, 1].

Integrity notes:
- This is an experimental relative patch attribution method illustrating model decision focus.
- It is NOT standard classical Grad-CAM, nor an authoritative forensic locator.
- It does NOT provide a pixel-level forgery or tamper mask, and does NOT explain why an image is real.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.transforms import get_eval_transforms


def get_evaluation_crop(image: Image.Image, image_size: int = 224) -> Image.Image:
    """Extract the exact cropped window that the model evaluated (Resize 256 -> CenterCrop 224).

    Parameters
    ----------
    image : PIL.Image.Image
        Source input image.
    image_size : int, default=224
        Target evaluation dimension.

    Returns
    -------
    PIL.Image.Image
        The central 224x224 cropped image.
    """
    resize_size = int(image_size * (256.0 / 224.0))
    w, h = image.size
    if w < h:
        new_w = resize_size
        new_h = int(h * (resize_size / w))
    else:
        new_h = resize_size
        new_w = int(w * (resize_size / h))
    resized = image.convert("RGB").resize((new_w, new_h), resample=Image.BILINEAR)

    left = (new_w - image_size) // 2
    top = (new_h - image_size) // 2
    return resized.crop((left, top, left + image_size, top + image_size))


def apply_colormap_jet(intensity: np.ndarray) -> np.ndarray:
    """Convert a 2D intensity array in [0, 1] to an RGB uint8 array using the Jet colormap.

    Pure NumPy implementation requiring no external plotting libraries.
    """
    val = np.clip(intensity, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * val - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * val - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * val - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


class ViTPatchGradCAM:
    """Experimental contrast-centered ViT patch decision saliency generator.

    Parameters
    ----------
    model : nn.Module
        The classifier instance (e.g. SignalScopeClassifier or timm VisionTransformer).
    """

    def __init__(self, model: nn.Module):
        self.model = model
        raw_vit = getattr(model, "model", model)
        if not hasattr(raw_vit, "blocks") or len(raw_vit.blocks) == 0:
            raise ValueError(f"Model {type(raw_vit)} does not contain transformer blocks.")
        if not hasattr(raw_vit.blocks[-1], "norm1"):
            raise ValueError(f"Last block {type(raw_vit.blocks[-1])} does not have 'norm1' layer.")

        self.target_layer = raw_vit.blocks[-1].norm1
        self.raw_vit = raw_vit

    def generate_saliency_map(
        self,
        image_tensor: torch.Tensor,
    ) -> np.ndarray:
        """Generate normalized 2D patch saliency map of shape (224, 224).

        Parameters
        ----------
        image_tensor : torch.Tensor
            Preprocessed input tensor of shape (1, 3, 224, 224) or (3, 224, 224).

        Returns
        -------
        saliency_map : np.ndarray
            2D float32 array of shape (224, 224) with values in [0, 1].
        """
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        device = next(self.model.parameters()).device
        x = image_tensor.to(device).clone().detach().requires_grad_(True)

        activations: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []

        def forward_hook(module, inp, out):
            activations.append(out)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        f_hook = self.target_layer.register_forward_hook(forward_hook)
        b_hook = self.target_layer.register_full_backward_hook(backward_hook)

        was_training = self.model.training
        self.model.eval()

        try:
            # Forward pass: obtain scalar logit
            if hasattr(self.model, "forward"):
                out = self.model(x)
            else:
                out = self.raw_vit(x)

            logit = out.squeeze()
            if logit.ndim > 0:
                logit = logit[0]

            self.model.zero_grad(set_to_none=True)
            logit.backward(retain_graph=False)

            if not activations or not gradients:
                raise RuntimeError("Failed to capture activations or gradients during hook execution.")

            act = activations[0]   # Shape (1, 197, 768)
            grad = gradients[0]    # Shape (1, 197, 768)

            # Exclude CLS token at index 0, isolate 196 spatial patch tokens
            act_patches = act[:, 1:, :]   # Shape (1, 196, 768)
            grad_patches = grad[:, 1:, :] # Shape (1, 196, 768)

            # Channel importance weights: average gradients across the 196 spatial tokens
            weights = grad_patches.mean(dim=1, keepdim=True) # Shape (1, 1, 768)

            # Weighted sum over channel features
            cam = (act_patches * weights).sum(dim=-1) # Shape (1, 196)

            # Contrast-centering: pre-attention LayerNorm tokens share an overall spatial DC offset.
            # Centering relative to spatial mean/min isolates above-average contributing patches.
            if cam.max() <= 0:
                cam = cam - cam.mean(dim=-1, keepdim=True)
            elif cam.min() >= 0:
                cam = cam - cam.min(dim=-1, keepdim=True).values

            # Apply ReLU: keep features that contributed positively toward AI logit
            cam = F.relu(cam)

            # Reshape 196 tokens into 14x14 spatial patch grid
            cam_grid = cam.reshape(1, 1, 14, 14)

            # Upsample to (224, 224) via bilinear interpolation
            cam_upsampled = F.interpolate(
                cam_grid,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )

            # Extract 2D float32 array
            cam_map = cam_upsampled.squeeze().detach().cpu().numpy().astype(np.float32)

            # Min-Max normalize to [0, 1]
            min_val = float(cam_map.min())
            max_val = float(cam_map.max())
            if max_val - min_val > 1e-7:
                cam_map = (cam_map - min_val) / (max_val - min_val)
            else:
                cam_map = np.zeros_like(cam_map)

            return cam_map

        finally:
            f_hook.remove()
            b_hook.remove()
            self.model.zero_grad(set_to_none=True)
            if was_training:
                self.model.train()


def overlay_saliency(
    eval_crop: Image.Image,
    saliency_map: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Blend a 2D saliency map onto the 224x224 evaluation crop.

    Parameters
    ----------
    eval_crop : PIL.Image.Image
        Cropped image of size (224, 224).
    saliency_map : np.ndarray
        2D array of shape (224, 224) in [0, 1].
    alpha : float, default=0.45
        Overlay blending weight.

    Returns
    -------
    PIL.Image.Image
        Blended visualization image of size (224, 224).
    """
    crop_rgb = eval_crop.convert("RGB").resize((224, 224), resample=Image.BILINEAR)
    heat_rgb = apply_colormap_jet(saliency_map)
    heat_img = Image.fromarray(heat_rgb, mode="RGB")
    return Image.blend(crop_rgb, heat_img, alpha=alpha)


def generate_explanation(
    image: Image.Image,
    model: nn.Module,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, Image.Image, Image.Image]:
    """Convenience pipeline to generate saliency map and blended overlay for a PIL image.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image.
    model : nn.Module
        Loaded classifier.
    device : torch.device, optional
        Target device.

    Returns
    -------
    saliency_map : np.ndarray
        2D array of shape (224, 224) with values in [0, 1].
    eval_crop : PIL.Image.Image
        The central 224x224 crop evaluated by the model.
    overlay_image : PIL.Image.Image
        The blended overlay image of size (224, 224).
    """
    transform = get_eval_transforms(224)
    tensor = transform(image.convert("RGB"))

    explainer = ViTPatchGradCAM(model)
    saliency_map = explainer.generate_saliency_map(tensor)

    eval_crop = get_evaluation_crop(image, 224)
    overlay_img = overlay_saliency(eval_crop, saliency_map)

    return saliency_map, eval_crop, overlay_img
