"""Render representative DeepLabV3+ predictions without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_deeplabv3plus import (
    DeepLabV3PlusMobileNetV3,
    MEAN,
    NormalizedLesionDataset,
    STD,
    class_name,
)
from train_unet import collect_pairs, load_or_create_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = args.data.resolve()
    run = args.run.resolve()
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    pairs = collect_pairs(data)
    splits, _ = load_or_create_splits(pairs, data, data / "splits.json", seed=42)
    wanted = (
        "Apple_Apple_scab",
        "Apple_Black_rot",
        "Potato_Early_blight",
        "Tomato_Late_blight",
    )
    selected = []
    for target in wanted:
        selected.append(next(pair for pair in splits["test"] if class_name(pair[0].stem) == target))
    dataset = NormalizedLesionDataset(selected, checkpoint["image_size"], augment=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3PlusMobileNetV3(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    figure, axes = plt.subplots(len(dataset), 4, figsize=(13, 3.2 * len(dataset)), squeeze=False)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    with torch.no_grad():
        for row in range(len(dataset)):
            normalized, mask, name = dataset[row]
            probability = torch.sigmoid(model(normalized.unsqueeze(0).to(device)))[0, 0].cpu()
            prediction = probability >= checkpoint["threshold"]
            image = (normalized * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            overlay = image.copy()
            selected_pixels = prediction.numpy()
            overlay[selected_pixels] = 0.55 * overlay[selected_pixels] + 0.45 * np.array([1.0, 0.0, 0.0])
            panels = (image, mask[0].numpy(), prediction.numpy(), overlay)
            titles = (name, "Ground truth", "Prediction", "Overlay")
            for column, (panel, title) in enumerate(zip(panels, titles)):
                axes[row, column].imshow(panel, cmap="gray" if panel.ndim == 2 else None)
                axes[row, column].set_title(title, fontsize=9)
                axes[row, column].axis("off")
    figure.tight_layout()
    output = run / "test_predictions.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
