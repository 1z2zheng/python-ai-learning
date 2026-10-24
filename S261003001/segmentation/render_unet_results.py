"""Render U-Net training curves and test predictions without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from train_unet import (
    LesionDataset,
    UNet,
    collect_pairs,
    load_or_create_splits,
    save_prediction_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = args.data.resolve()
    run = args.run.resolve()
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)

    history = metrics["history"]
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train"]["loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val"]["loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, [row["train"]["dice"] for row in history], label="train")
    axes[1].plot(epochs, [row["val"]["dice"] for row in history], label="val")
    axes[1].set_title("Dice")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(run / "training_curves.png", dpi=160)
    plt.close(figure)

    pairs = collect_pairs(data)
    splits, _ = load_or_create_splits(pairs, data, data / "splits.json", seed=42)
    test_dataset = LesionDataset(
        splits["test"], checkpoint["image_size"], augment=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    save_prediction_figure(
        model,
        test_loader,
        device,
        checkpoint["threshold"],
        run / "test_predictions.png",
    )
    print(f"Saved: {run / 'training_curves.png'}")
    print(f"Saved: {run / 'test_predictions.png'}")


if __name__ == "__main__":
    main()
