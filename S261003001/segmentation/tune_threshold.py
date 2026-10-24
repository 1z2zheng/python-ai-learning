"""Tune a binary segmentation threshold on the validation split."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_unet import LesionDataset, UNet, collect_pairs, load_or_create_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def scores(probability: torch.Tensor, truth: torch.Tensor, threshold: float):
    prediction = probability >= threshold
    truth = truth >= 0.5
    dims = (1, 2, 3)
    intersection = (prediction & truth).sum(dims).float()
    pred_sum = prediction.sum(dims).float()
    truth_sum = truth.sum(dims).float()
    union = (prediction | truth).sum(dims).float()
    dice = (2 * intersection + 1e-7) / (pred_sum + truth_sum + 1e-7)
    iou = (intersection + 1e-7) / (union + 1e-7)
    return dice.cpu().numpy(), iou.cpu().numpy()


def main() -> None:
    args = parse_args()
    data = args.data.resolve()
    run = args.run.resolve()
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    pairs = collect_pairs(data)
    splits, split_source = load_or_create_splits(
        pairs, data, data / "splits.json", seed=42
    )
    dataset = LesionDataset(splits["val"], checkpoint["image_size"], augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    batches = []
    with torch.no_grad():
        for images, masks, names in tqdm(loader, desc="Validation inference"):
            probabilities = torch.sigmoid(model(images.to(device))).cpu()
            batches.append((probabilities, masks, list(names)))

    thresholds = np.round(np.arange(0.20, 0.81, 0.02), 2)
    rows = []
    for threshold in thresholds:
        all_dice = []
        all_iou = []
        for probabilities, masks, _ in batches:
            dice, iou = scores(probabilities, masks, float(threshold))
            all_dice.extend(dice.tolist())
            all_iou.extend(iou.tolist())
        rows.append(
            {
                "threshold": float(threshold),
                "dice": float(np.mean(all_dice)),
                "iou": float(np.mean(all_iou)),
            }
        )

    best = max(rows, key=lambda row: row["dice"])
    per_class: dict[str, list[float]] = defaultdict(list)
    for probabilities, masks, names in batches:
        dice, _ = scores(probabilities, masks, best["threshold"])
        for name, value in zip(names, dice):
            per_class[name.split("__", 1)[0]].append(float(value))

    report = {
        "device": str(device),
        "split": "val",
        "split_source": split_source,
        "samples": len(dataset),
        "checkpoint_threshold": checkpoint["threshold"],
        "best": best,
        "at_0.5": next(row for row in rows if row["threshold"] == 0.5),
        "per_class_dice_at_best_threshold": {
            name: float(np.mean(values)) for name, values in sorted(per_class.items())
        },
        "thresholds": rows,
    }
    output = run / "threshold_analysis.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["best"], ensure_ascii=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
