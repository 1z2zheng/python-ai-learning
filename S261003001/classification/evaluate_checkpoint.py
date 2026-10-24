"""Evaluate a classification checkpoint on PlantVillage and PlantSeg test sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from .train import build_model
    from .train_mixed import (
        MixedDataset,
        evaluation_report,
        make_transforms,
        plantseg_records,
        plantvillage_records,
        run_epoch,
    )
except ImportError:
    from train import build_model
    from train_mixed import (
        MixedDataset,
        evaluation_report,
        make_transforms,
        plantseg_records,
        plantvillage_records,
        run_epoch,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plantvillage", type=Path, default=Path("dataset_classification"))
    parser.add_argument("--plantseg", type=Path, default=Path("dataset_segmentation_plantseg"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    model = build_model(checkpoint["model_name"], len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    _, eval_transform = make_transforms(int(checkpoint.get("image_size", 224)))
    roots = (args.plantvillage.resolve(), args.plantseg.resolve())
    datasets = {
        "plantvillage": MixedDataset(
            plantvillage_records(roots[0], "test", classes), eval_transform
        ),
        "plantseg": MixedDataset(plantseg_records(roots[1], "test", classes), eval_transform),
    }
    criterion = nn.CrossEntropyLoss()
    report = {"checkpoint": str(checkpoint_path), "model": checkpoint["model_name"], "device": str(device)}
    for name, dataset in datasets.items():
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.workers > 0,
        )
        metrics, targets, predictions = run_epoch(model, loader, criterion, device)
        report[name] = evaluation_report(metrics, targets, predictions, classes)
        print(
            f"{name}: accuracy={metrics['accuracy']:.4f}, "
            f"confidence={metrics['mean_confidence']:.4f}, ece={metrics['ece']:.4f}"
        )
    output = args.output.resolve() if args.output else checkpoint_path.parent / "external_evaluation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
