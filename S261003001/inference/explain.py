"""Grad-CAM visualization for the deployed MobileNetV3 classifier."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def grad_cam(
    image: Image.Image,
    model: torch.nn.Module,
    transform,
    device: torch.device,
    class_index: int,
    image_size: int,
) -> tuple[Image.Image, Image.Image]:
    """Return the classifier's center crop and a Grad-CAM overlay for one class."""
    if not hasattr(model, "features"):
        raise ValueError("Grad-CAM currently supports MobileNetV3 classifiers only")

    activations: list[torch.Tensor] = []
    handle = model.features[-1].register_forward_hook(
        lambda _module, _inputs, output: activations.append(output)
    )
    try:
        with torch.enable_grad():
            tensor = transform(image).unsqueeze(0).to(device)
            logits = model(tensor)
            gradients = torch.autograd.grad(logits[0, class_index], activations[0])[0]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            heatmap = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
            heatmap = torch.nn.functional.interpolate(
                heatmap, size=(image_size, image_size), mode="bilinear", align_corners=False
            )[0, 0]
            heatmap = heatmap.detach().float().cpu().numpy()
    finally:
        handle.remove()

    heatmap -= heatmap.min()
    maximum = float(heatmap.max())
    if maximum > 0:
        heatmap /= maximum

    # Match the Resize + CenterCrop geometry used for classification.
    crop = transforms.CenterCrop(image_size)(
        transforms.Resize(int(image_size * 1.14))(image)
    ).convert("RGB")
    base = np.asarray(crop, dtype=np.float32)
    color = np.zeros_like(base)
    color[..., 0] = 255 * heatmap
    color[..., 1] = 180 * np.maximum(0, 1 - np.abs(heatmap - 0.5) * 2)
    color[..., 2] = 180 * (1 - heatmap)
    alpha = 0.55 * heatmap[..., None]
    overlay = Image.fromarray(np.clip(base * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8))
    return crop, overlay
