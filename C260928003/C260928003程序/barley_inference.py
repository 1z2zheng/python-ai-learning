"""Inference utilities for five-band spring-barley UAV TIFF imagery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image

from train_spring_barley_multitask import PatchCNN


def load_five_band(path: Path) -> np.ndarray:
    image = tifffile.imread(path)
    if image.ndim != 3:
        raise ValueError(f"需要五通道TIFF，当前数据形状为{image.shape}")
    if image.shape[0] == 5 and image.shape[-1] != 5:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 5:
        raise ValueError(f"需要蓝、绿、红、红边、近红外五通道TIFF，当前数据形状为{image.shape}")
    return np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def unit_scale(image: np.ndarray) -> np.ndarray:
    if image.max(initial=0) <= 1.5 and image.min(initial=0) >= 0:
        return np.clip(image, 0, 1)
    if image.max(initial=0) <= 255 and image.min(initial=0) >= 0:
        return np.clip(image / 255.0, 0, 1)
    output = np.zeros_like(image, dtype=np.float32)
    for band in range(image.shape[-1]):
        low, high = np.percentile(image[..., band], (1, 99))
        output[..., band] = np.clip((image[..., band] - low) / max(high - low, 1e-6), 0, 1)
    return output


def rgb_preview(image: np.ndarray) -> Image.Image:
    rgb = unit_scale(image)[..., [2, 1, 0]]
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")




def load_task_ensemble(checkpoints: list[Path], device: torch.device):
    entries, classes = [], None
    for path in checkpoints:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = PatchCNN(in_channels=5).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        entries.append((model, torch.tensor(checkpoint["mean"], device=device).view(1, 5, 1, 1),
                        torch.tensor(checkpoint["std"], device=device).view(1, 5, 1, 1)))
        classes = list(checkpoint["classes"])
    return entries, classes




def _tiles(image: np.ndarray, size: int = 30):
    height, width = image.shape[:2]
    if height < size or width < size:
        padded = np.pad(image, ((0, max(0, size - height)), (0, max(0, size - width)), (0, 0)), mode="edge")
        return padded[None, :size, :size], [(0, 0, height, width)]
    samples, boxes = [], []
    for y in range(0, height - size + 1, size):
        for x in range(0, width - size + 1, size):
            samples.append(image[y:y + size, x:x + size])
            boxes.append((y, x, size, size))
    return np.stack(samples), boxes


@torch.inference_mode()
def predict_map(image: np.ndarray, entries, classes, device: torch.device, batch_size: int = 256):
    # Task models were trained from reflectance values on a 0-1 scale.  Uploaded
    # five-band TIFFs may be uint8 (0-255), floating reflectance, or higher-range
    # sensor values.  Normalise before tiling so inference matches training.
    samples, boxes = _tiles(unit_scale(image))
    raw = torch.from_numpy(np.ascontiguousarray(samples)).permute(0, 3, 1, 2).to(device)
    batches = []
    for start in range(0, len(raw), batch_size):
        batch = raw[start:start + batch_size]
        ensemble = torch.stack([model((batch - mean) / std).softmax(1) for model, mean, std in entries]).mean(0)
        batches.append(ensemble[:, 1].cpu())
    probabilities = torch.cat(batches).numpy()
    heatmap = np.zeros(image.shape[:2], dtype=np.float32)
    for probability, (y, x, height, width) in zip(probabilities, boxes):
        heatmap[y:y + height, x:x + width] = probability
    mask = heatmap >= 0.5
    mean_probability = float(probabilities.mean())
    result = {"label": classes[int(mean_probability >= 0.5)],
              "confidence": max(mean_probability, 1 - mean_probability),
              "positive_probability": mean_probability, "positive_coverage": float(mask.mean()),
              "tile_count": len(boxes)}
    return result, heatmap, mask


def visualizations(image: np.ndarray, heatmap: np.ndarray, mask: np.ndarray):
    preview = np.asarray(rgb_preview(image)).astype(np.float32)
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    colour = preview.copy()
    colour[..., 0], colour[..., 1], colour[..., 2] = 255, colour[..., 1] * 0.25, colour[..., 2] * 0.25
    alpha = (0.55 * heatmap)[..., None]
    overlay = preview * (1 - alpha) + colour * alpha
    return mask_image, Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")
