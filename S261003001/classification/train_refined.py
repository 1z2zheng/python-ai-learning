"""Second-stage domain and hard-class refinement for the mixed classifier.

This script keeps the official validation/test splits unchanged. It starts from
the existing mixed MobileNetV3-Large checkpoint, gives PlantSeg and weak
classes more sampling probability, and selects the best epoch by PlantSeg
macro-F1 while retaining PlantVillage accuracy.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from .train import build_model
    from .train_mixed import (
        MixedDataset,
        Record,
        evaluation_report,
        make_transforms,
        plantseg_records,
        plantvillage_records,
        read_classes,
        remove_cross_domain_leakage,
        run_epoch,
        seed_everything,
    )
except ImportError:
    from train import build_model
    from train_mixed import (
        MixedDataset,
        Record,
        evaluation_report,
        make_transforms,
        plantseg_records,
        plantvillage_records,
        read_classes,
        remove_cross_domain_leakage,
        run_epoch,
        seed_everything,
    )


DEFAULT_WEAK_BOOSTS = {
    "Apple_Black_rot": 1.5,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FocalCrossEntropy(nn.Module):
    """Cross entropy with focal modulation and optional label smoothing."""

    def __init__(self, gamma: float = 1.5, label_smoothing: float = 0.03) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = nn.functional.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        true_probability = logits.softmax(dim=1).gather(1, targets[:, None]).squeeze(1)
        focal_weight = (1.0 - true_probability).pow(self.gamma)
        return (focal_weight * cross_entropy).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plantvillage", type=Path, default=Path("dataset_classification"))
    parser.add_argument("--plantseg", type=Path, default=Path("dataset_segmentation_plantseg"))
    parser.add_argument("--plantseg-splits", type=Path, default=None,
                        help="Optional reviewed PlantSeg split manifest; should match the initial checkpoint")
    parser.add_argument("--extra", type=Path, default=Path("dataset_natural_extra"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/mobilenet_v3_large_mixed/best.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/mobilenet_v3_large_plantdoc"),
    )
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--plantseg-share", type=float, default=0.50)
    parser.add_argument("--plantdoc-share", "--extra-share", dest="plantdoc_share", type=float, default=0.30,
                        help="Sampling share of additional field data (PlantDoc/AppleLeaf9)")
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def extra_records(root: Path, classes: list[str]) -> list[Record]:
    class_to_index = {name: index for index, name in enumerate(classes)}
    records = []
    train_root = root / "train"
    if not train_root.is_dir():
        return records
    for directory in sorted(train_root.iterdir()):
        if not directory.is_dir():
            continue
        class_name = directory.name
        if class_name not in class_to_index:
            raise ValueError(f"Unknown extra-data class: {class_name}")
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(
                    Record(path, class_to_index[class_name], class_name, "plantdoc")
                )
    return records


def refined_sampler(
    records,
    plantseg_share: float,
    plantdoc_share: float,
    seed: int,
) -> WeightedRandomSampler:
    """Balance classes/domains, then deliberately emphasize weak classes."""
    if plantseg_share <= 0 or plantdoc_share < 0 or plantseg_share + plantdoc_share >= 1:
        raise ValueError("Domain shares must be positive and sum to less than 1")
    requested_mass = {
        "plantvillage": 1.0 - plantseg_share - plantdoc_share,
        "plantseg": plantseg_share,
        "plantdoc": plantdoc_share,
    }
    counts = Counter((record.class_name, record.domain) for record in records)
    domains_by_class: dict[str, set[str]] = {}
    for record in records:
        domains_by_class.setdefault(record.class_name, set()).add(record.domain)

    weights = []
    for record in records:
        domains = domains_by_class[record.class_name]
        available_mass = sum(requested_mass[domain] for domain in domains)
        domain_mass = requested_mass[record.domain] / available_mass
        weak_boost = DEFAULT_WEAK_BOOSTS.get(record.class_name, 1.0)
        weights.append(
            weak_boost * domain_mass / counts[(record.class_name, record.domain)]
        )

    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=len(records),
        replacement=True,
        generator=generator,
    )


def disease_macro_f1(
    targets: list[int], predictions: list[int], classes: list[str]
) -> float:
    present_labels = sorted(set(targets))
    return float(
        f1_score(
            targets,
            predictions,
            labels=present_labels,
            average="macro",
            zero_division=0,
        )
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    plantvillage = args.plantvillage.resolve()
    plantseg = args.plantseg.resolve()
    extra = args.extra.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing initial checkpoint: {checkpoint_path}")
    classes = read_classes(plantvillage)
    initial_checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    split_manifest_path = args.plantseg_splits.resolve() if args.plantseg_splits else None
    if split_manifest_path is not None and initial_checkpoint.get("plantseg_split_manifest") != str(split_manifest_path):
        raise ValueError(
            "Initial checkpoint was not trained with this reviewed PlantSeg split. "
            "Retrain the mixed checkpoint with --plantseg-splits first."
        )
    if initial_checkpoint["classes"] != classes:
        raise ValueError("Checkpoint classes do not match dataset classes")
    image_size = int(initial_checkpoint.get("image_size", 320))
    train_transform, eval_transform = make_transforms(image_size)

    pv = {
        split: plantvillage_records(plantvillage, split, classes)
        for split in ("train", "val", "test")
    }
    ps = {
        split: plantseg_records(plantseg, split, classes, split_manifest_path)
        for split in ("train", "val", "test")
    }
    filtered_pv, filtered_ps, deduplication = remove_cross_domain_leakage(
        pv["train"],
        ps["train"],
        pv["val"] + pv["test"],
        ps["val"] + ps["test"],
    )
    extra_train = extra_records(extra, classes)
    if not extra_train:
        raise RuntimeError(
            f"No additional natural-scene images found under {extra / 'train'}"
        )
    train_records = filtered_pv + filtered_ps + extra_train
    extra_source_counts = Counter(
        "AppleLeaf9" if record.path.name.startswith("appleleaf9_blackrot_")
        else "PlantDoc" if record.path.name.startswith("plantdoc_")
        else "Other reviewed field photos" for record in extra_train
    )
    datasets = {
        "train": MixedDataset(train_records, train_transform),
        "pv_val": MixedDataset(pv["val"], eval_transform),
        "ps_val": MixedDataset(ps["val"], eval_transform),
        "pv_test": MixedDataset(pv["test"], eval_transform),
        "ps_test": MixedDataset(ps["test"], eval_transform),
    }
    sampler = refined_sampler(
        train_records,
        args.plantseg_share,
        args.plantdoc_share,
        args.seed,
    )
    pin_memory = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=pin_memory,
            persistent_workers=args.workers > 0,
        ),
        **{
            name: DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=pin_memory,
                persistent_workers=args.workers > 0,
            )
            for name, dataset in datasets.items()
            if name != "train"
        },
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("mobilenet_v3_large", len(classes), pretrained=False).to(device)
    model.load_state_dict(initial_checkpoint["model_state"])
    criterion = FocalCrossEntropy(args.focal_gamma, args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=2e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    data_summary = {
        "initial_checkpoint": str(checkpoint_path),
        "plantseg_split_manifest": str(split_manifest_path or plantseg / "splits.json"),
        "image_size": image_size,
        "plantvillage": {split: len(records) for split, records in pv.items()},
        "plantseg": {split: len(records) for split, records in ps.items()},
        "plantdoc_extra_train": len(extra_train),
        "extra_train_by_source": dict(extra_source_counts),
        "plantdoc_extra_by_class": dict(
            Counter(record.class_name for record in extra_train)
        ),
        "mixed_train": len(train_records),
        "plantseg_share": args.plantseg_share,
        "plantdoc_share": args.plantdoc_share,
        "plantvillage_share": 1.0 - args.plantseg_share - args.plantdoc_share,
        "weak_class_boosts": DEFAULT_WEAK_BOOSTS,
        "deduplication": deduplication,
    }
    (output / "data_summary.json").write_text(
        json.dumps(data_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.dry_run:
        images, labels, domains, _ = next(iter(loaders["train"]))
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        print(
            f"Dry run ok: device={device}, images={tuple(images.shape)}, "
            f"loss={loss.item():.4f}, domains={dict(Counter(domains))}"
        )
        return

    history: list[dict[str, object]] = []
    best_score = -1.0
    stale_epochs = 0
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = run_epoch(
            model, loaders["train"], criterion, device, optimizer, scaler
        )
        pv_val, pv_targets, pv_predictions = run_epoch(
            model, loaders["pv_val"], criterion, device
        )
        ps_val, ps_targets, ps_predictions = run_epoch(
            model, loaders["ps_val"], criterion, device
        )
        scheduler.step()
        ps_macro_f1 = disease_macro_f1(ps_targets, ps_predictions, classes)
        # Natural-scene performance dominates, while the clean-domain term
        # prevents catastrophic forgetting of PlantVillage.
        selection_score = 0.75 * ps_macro_f1 + 0.25 * pv_val["accuracy"]
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "plantvillage_val": pv_val,
            "plantseg_val": ps_val,
            "plantseg_macro_f1": ps_macro_f1,
            "selection_score": selection_score,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}: train={train_metrics['accuracy']:.4f} | "
            f"PV={pv_val['accuracy']:.4f} | PS={ps_val['accuracy']:.4f} "
            f"macro_f1={ps_macro_f1:.4f} | score={selection_score:.4f}"
        )
        if selection_score > best_score:
            best_score = selection_score
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": "mobilenet_v3_large",
                    "classes": classes,
                    "image_size": image_size,
                    "best_selection_score": best_score,
                    "parameter_count": parameter_count,
                    "training_domains": ["PlantVillage", "PlantSeg", *extra_source_counts.keys()],
                    "plantseg_split_manifest": str(split_manifest_path or plantseg / "splits.json"),
                    "refinement": {
                        "initial_checkpoint": str(checkpoint_path),
                        "plantseg_share": args.plantseg_share,
                        "plantdoc_share": args.plantdoc_share,
                        "weak_class_boosts": DEFAULT_WEAK_BOOSTS,
                        "focal_gamma": args.focal_gamma,
                    },
                },
                output / "best.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    best_checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    pv_metrics, pv_targets, pv_predictions = run_epoch(
        model, loaders["pv_test"], criterion, device
    )
    ps_metrics, ps_targets, ps_predictions = run_epoch(
        model, loaders["ps_test"], criterion, device
    )
    metrics = {
        "device": str(device),
        "model": "mobilenet_v3_large_refined",
        "classes": classes,
        "parameter_count": parameter_count,
        "best_selection_score": best_score,
        "data": data_summary,
        "plantvillage_test": evaluation_report(
            pv_metrics, pv_targets, pv_predictions, classes
        ),
        "plantseg_test": {
            **evaluation_report(ps_metrics, ps_targets, ps_predictions, classes),
            "disease_macro_f1": disease_macro_f1(
                ps_targets, ps_predictions, classes
            ),
        },
        "history": history,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "classes.json").write_text(
        json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    baseline_metrics_path = checkpoint_path.parent / "metrics.json"
    if baseline_metrics_path.is_file():
        baseline = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
        baseline_report = baseline["plantseg_test"]["classification_report"]
        baseline_disease_f1 = float(
            np.mean(
                [
                    baseline_report[name]["f1-score"]
                    for name in classes
                    if baseline_report[name]["support"] > 0
                ]
            )
        )
        comparison = {
            "baseline": {
                "checkpoint": str(checkpoint_path),
                "plantvillage_accuracy": baseline["plantvillage_test"]["accuracy"],
                "plantseg_accuracy": baseline["plantseg_test"]["accuracy"],
                "plantseg_disease_macro_f1": baseline_disease_f1,
            },
            "refined": {
                "checkpoint": str(output / "best.pt"),
                "plantvillage_accuracy": pv_metrics["accuracy"],
                "plantseg_accuracy": ps_metrics["accuracy"],
                "plantseg_disease_macro_f1": metrics["plantseg_test"][
                    "disease_macro_f1"
                ],
            },
        }
        comparison["change"] = {
            "plantvillage_accuracy": comparison["refined"]["plantvillage_accuracy"]
            - comparison["baseline"]["plantvillage_accuracy"],
            "plantseg_accuracy": comparison["refined"]["plantseg_accuracy"]
            - comparison["baseline"]["plantseg_accuracy"],
            "plantseg_disease_macro_f1": comparison["refined"][
                "plantseg_disease_macro_f1"
            ]
            - comparison["baseline"]["plantseg_disease_macro_f1"],
        }
        comparison["deployment_recommended"] = bool(
            comparison["change"]["plantseg_accuracy"] >= 0
            and comparison["change"]["plantseg_disease_macro_f1"] >= 0
            and comparison["refined"]["plantvillage_accuracy"]
            >= comparison["baseline"]["plantvillage_accuracy"] - 0.01
        )
        (output / "comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(
        epochs,
        [row["plantvillage_val"]["accuracy"] for row in history],
        label="PlantVillage accuracy",
    )
    axes[0].plot(
        epochs,
        [row["plantseg_val"]["accuracy"] for row in history],
        label="PlantSeg accuracy",
    )
    axes[0].set_title("Validation accuracy")
    axes[1].plot(
        epochs,
        [row["plantseg_macro_f1"] for row in history],
        label="PlantSeg macro-F1",
    )
    axes[1].plot(
        epochs,
        [row["selection_score"] for row in history],
        label="Selection score",
    )
    axes[1].set_title("Refinement objective")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=160)
    plt.close(figure)

    print(
        f"PlantVillage test: accuracy={pv_metrics['accuracy']:.4f}, "
        f"confidence={pv_metrics['mean_confidence']:.4f}"
    )
    print(
        f"PlantSeg test: accuracy={ps_metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['plantseg_test']['disease_macro_f1']:.4f}, "
        f"confidence={ps_metrics['mean_confidence']:.4f}"
    )
    if baseline_metrics_path.is_file():
        print(
            "Deployment recommendation: "
            + ("use refined checkpoint" if comparison["deployment_recommended"] else "keep current checkpoint")
        )
    print(f"Saved outputs to {output}")


if __name__ == "__main__":
    main()
