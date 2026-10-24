"""Fine-tune MobileNetV3-Large on mixed PlantVillage and PlantSeg data."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

try:
    from .train import build_model
except ImportError:
    from train import build_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Record:
    path: Path
    label: int
    class_name: str
    domain: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plantvillage", type=Path, default=Path("dataset_classification"))
    parser.add_argument("--plantseg", type=Path, default=Path("dataset_segmentation_plantseg"))
    parser.add_argument("--plantseg-splits", type=Path, default=None,
                        help="Optional reviewed PlantSeg split manifest; original data is unchanged")
    parser.add_argument("--output", type=Path, default=Path("runs/mobilenet_v3_large_mixed"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--plantseg-share", type=float, default=0.50)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_classes(root: Path) -> list[str]:
    classes = json.loads((root / "classes.json").read_text(encoding="utf-8"))
    if not isinstance(classes, list) or len(classes) != 11:
        raise ValueError("Expected 11 classes in dataset_classification/classes.json")
    return classes


def plantvillage_records(root: Path, split: str, classes: list[str]) -> list[Record]:
    records = []
    for label, class_name in enumerate(classes):
        directory = root / split / class_name
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(Record(path, label, class_name, "plantvillage"))
    return records


def plantseg_records(root: Path, split: str, classes: list[str], split_manifest_path: Path | None = None) -> list[Record]:
    manifest_path = split_manifest_path or root / "splits.json"
    split_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    class_to_index = {name: index for index, name in enumerate(classes)}
    images = {path.stem: path for path in (root / "images").iterdir() if path.is_file()}
    records = []
    for stem in split_manifest[split]:
        class_name = stem.split("__", 1)[0]
        if class_name not in class_to_index:
            raise ValueError(f"Unknown PlantSeg class: {class_name}")
        if stem not in images:
            raise FileNotFoundError(f"PlantSeg image missing: {stem}")
        records.append(Record(images[stem], class_to_index[class_name], class_name, "plantseg"))
    return records


def difference_hash(path: Path) -> int:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L").resize(
            (9, 8), Image.Resampling.LANCZOS
        )
    values = np.asarray(image, dtype=np.int16)
    bits = (values[:, 1:] > values[:, :-1]).reshape(-1)
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def remove_cross_domain_leakage(
    pv_train: list[Record],
    ps_train: list[Record],
    pv_protected: list[Record],
    ps_protected: list[Record],
) -> tuple[list[Record], list[Record], dict[str, object]]:
    print("Checking cross-domain near-duplicates (dHash distance <= 2)...")
    protected_ps_hashes = [difference_hash(record.path) for record in ps_protected]
    protected_pv_hashes = [difference_hash(record.path) for record in pv_protected]

    removed_pv = []
    filtered_pv = []
    for record in tqdm(pv_train, desc="PlantVillage leakage check", leave=False):
        fingerprint = difference_hash(record.path)
        if any((fingerprint ^ known).bit_count() <= 2 for known in protected_ps_hashes):
            removed_pv.append(str(record.path))
        else:
            filtered_pv.append(record)

    removed_ps = []
    filtered_ps = []
    for record in tqdm(ps_train, desc="PlantSeg leakage check", leave=False):
        fingerprint = difference_hash(record.path)
        if any((fingerprint ^ known).bit_count() <= 2 for known in protected_pv_hashes):
            removed_ps.append(str(record.path))
        else:
            filtered_ps.append(record)

    report = {
        "method": "64-bit difference hash; remove cross-domain training image when Hamming distance <= 2 from validation/test",
        "plantvillage_train_removed": len(removed_pv),
        "plantseg_train_removed": len(removed_ps),
        "removed_plantvillage_paths": removed_pv,
        "removed_plantseg_paths": removed_ps,
    }
    return filtered_pv, filtered_ps, report


class MixedDataset(Dataset):
    def __init__(self, records: list[Record], transform) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        return self.transform(image), record.label, record.domain, record.path.stem


def make_transforms(size: int):
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.60, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.15),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.28, contrast=0.25, saturation=0.22, hue=0.04),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.15),
            transforms.RandomAutocontrast(p=0.15),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.12, scale=(0.02, 0.12), ratio=(0.5, 2.0)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(size * 1.14)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def balanced_domain_sampler(
    records: list[Record], plantseg_share: float, seed: int
) -> WeightedRandomSampler:
    if not 0.0 < plantseg_share < 1.0:
        raise ValueError("--plantseg-share must be between 0 and 1")
    counts = Counter((record.class_name, record.domain) for record in records)
    domains_by_class: dict[str, set[str]] = {}
    for record in records:
        domains_by_class.setdefault(record.class_name, set()).add(record.domain)
    weights = []
    for record in records:
        domains = domains_by_class[record.class_name]
        if domains == {"plantvillage", "plantseg"}:
            domain_mass = plantseg_share if record.domain == "plantseg" else 1.0 - plantseg_share
        else:
            domain_mass = 1.0
        weights.append(domain_mass / counts[(record.class_name, record.domain)])
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=len(records),
        replacement=True,
        generator=generator,
    )


def expected_calibration_error(
    confidences: list[float], correct: list[bool], bins: int = 10
) -> float:
    confidence_array = np.asarray(confidences)
    correct_array = np.asarray(correct, dtype=np.float32)
    error = 0.0
    for lower in np.linspace(0, 1, bins, endpoint=False):
        upper = lower + 1.0 / bins
        selected = (confidence_array > lower) & (confidence_array <= upper)
        if selected.any():
            error += selected.mean() * abs(
                correct_array[selected].mean() - confidence_array[selected].mean()
            )
    return float(error)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct_count = total = 0
    targets, predictions, confidences, correctness = [], [], [], []
    for images, labels, _, _ in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                assert optimizer is not None and scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        probabilities = logits.softmax(dim=1)
        confidence, prediction = probabilities.max(dim=1)
        is_correct = prediction == labels
        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct_count += is_correct.sum().item()
        total += batch_size
        targets.extend(labels.cpu().tolist())
        predictions.extend(prediction.cpu().tolist())
        confidences.extend(confidence.detach().cpu().tolist())
        correctness.extend(is_correct.detach().cpu().tolist())
    metrics = {
        "loss": loss_sum / total,
        "accuracy": correct_count / total,
        "mean_confidence": float(np.mean(confidences)),
        "correct_mean_confidence": float(
            np.mean([value for value, ok in zip(confidences, correctness) if ok])
        ),
        "ece": expected_calibration_error(confidences, correctness),
    }
    return metrics, targets, predictions


def evaluation_report(
    metrics: dict[str, float], targets: list[int], predictions: list[int], classes: list[str]
) -> dict[str, object]:
    labels = list(range(len(classes)))
    return {
        **metrics,
        "classification_report": classification_report(
            targets,
            predictions,
            labels=labels,
            target_names=classes,
            digits=4,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    plantvillage = args.plantvillage.resolve()
    plantseg = args.plantseg.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    classes = read_classes(plantvillage)
    train_transform, eval_transform = make_transforms(args.image_size)

    pv = {split: plantvillage_records(plantvillage, split, classes) for split in ("train", "val", "test")}
    split_manifest_path = args.plantseg_splits.resolve() if args.plantseg_splits else None
    ps = {split: plantseg_records(plantseg, split, classes, split_manifest_path) for split in ("train", "val", "test")}
    filtered_pv_train, filtered_ps_train, dedup_report = remove_cross_domain_leakage(
        pv["train"],
        ps["train"],
        pv["val"] + pv["test"],
        ps["val"] + ps["test"],
    )
    train_records = filtered_pv_train + filtered_ps_train
    datasets = {
        "train": MixedDataset(train_records, train_transform),
        "pv_val": MixedDataset(pv["val"], eval_transform),
        "ps_val": MixedDataset(ps["val"], eval_transform),
        "pv_test": MixedDataset(pv["test"], eval_transform),
        "ps_test": MixedDataset(ps["test"], eval_transform),
    }
    sampler = balanced_domain_sampler(train_records, args.plantseg_share, args.seed)
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
    data_summary = {
        "plantseg_split_manifest": str(split_manifest_path or plantseg / "splits.json"),
        "plantvillage": {split: len(records) for split, records in pv.items()},
        "plantseg": {split: len(records) for split, records in ps.items()},
        "mixed_train": len(train_records),
        "deduplication": dedup_report,
        "plantseg_sampling_share_for_disease_classes": args.plantseg_share,
    }
    (output / "data_summary.json").write_text(
        json.dumps(data_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(data_summary, ensure_ascii=False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("mobilenet_v3_large", len(classes), not args.no_pretrained).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.dry_run:
        images, labels, domains, _ = next(iter(loaders["train"]))
        images, labels = images.to(device), labels.to(device)
        dry_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        dry_scaler.scale(loss).backward()
        print(
            f"Dry run forward/backward ok: device={device}, image={tuple(images.shape)}, "
            f"logits={tuple(logits.shape)}, loss={loss.item():.4f}, "
            f"domains={dict(Counter(domains))}, parameters={parameter_count:,}"
        )
        return

    for parameter in model.features.parameters():
        parameter.requires_grad = args.freeze_epochs == 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_score = -1.0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            for parameter in model.features.parameters():
                parameter.requires_grad = True
            print("Unfroze MobileNetV3 feature encoder")
        train_metrics, _, _ = run_epoch(model, loaders["train"], criterion, device, optimizer, scaler)
        pv_val, _, _ = run_epoch(model, loaders["pv_val"], criterion, device)
        ps_val, _, _ = run_epoch(model, loaders["ps_val"], criterion, device)
        scheduler.step()
        domain_score = 0.5 * (pv_val["accuracy"] + ps_val["accuracy"])
        history.append(
            {"epoch": epoch, "train": train_metrics, "plantvillage_val": pv_val, "plantseg_val": ps_val, "domain_score": domain_score}
        )
        print(
            f"Epoch {epoch:02d}: train_acc={train_metrics['accuracy']:.4f} | "
            f"PV_val={pv_val['accuracy']:.4f} conf={pv_val['mean_confidence']:.4f} | "
            f"PS_val={ps_val['accuracy']:.4f} conf={ps_val['mean_confidence']:.4f} | "
            f"score={domain_score:.4f}"
        )
        if domain_score > best_score:
            best_score = domain_score
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": "mobilenet_v3_large",
                    "classes": classes,
                    "image_size": args.image_size,
                    "best_domain_score": best_score,
                    "parameter_count": parameter_count,
                    "training_domains": ["PlantVillage", "PlantSeg"],
                    "plantseg_split_manifest": str(split_manifest_path or plantseg / "splits.json"),
                },
                output / "best.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    pv_metrics, pv_targets, pv_predictions = run_epoch(model, loaders["pv_test"], criterion, device)
    ps_metrics, ps_targets, ps_predictions = run_epoch(model, loaders["ps_test"], criterion, device)
    metrics = {
        "device": str(device),
        "model": "mobilenet_v3_large",
        "classes": classes,
        "parameter_count": parameter_count,
        "best_domain_score": best_score,
        "data": data_summary,
        "plantvillage_test": evaluation_report(pv_metrics, pv_targets, pv_predictions, classes),
        "plantseg_test": evaluation_report(ps_metrics, ps_targets, ps_predictions, classes),
        "history": history,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "classes.json").write_text(
        json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train"]["accuracy"] for row in history], label="train")
    axes[0].plot(epochs, [row["plantvillage_val"]["accuracy"] for row in history], label="PlantVillage val")
    axes[0].plot(epochs, [row["plantseg_val"]["accuracy"] for row in history], label="PlantSeg val")
    axes[0].set_title("Accuracy")
    axes[1].plot(epochs, [row["plantvillage_val"]["mean_confidence"] for row in history], label="PlantVillage val")
    axes[1].plot(epochs, [row["plantseg_val"]["mean_confidence"] for row in history], label="PlantSeg val")
    axes[1].set_title("Mean confidence")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=160)
    plt.close(figure)
    print(
        f"PlantVillage test: accuracy={pv_metrics['accuracy']:.4f}, confidence={pv_metrics['mean_confidence']:.4f}"
    )
    print(
        f"PlantSeg test: accuracy={ps_metrics['accuracy']:.4f}, confidence={ps_metrics['mean_confidence']:.4f}"
    )
    print(f"Saved outputs to {output}")


if __name__ == "__main__":
    main()
