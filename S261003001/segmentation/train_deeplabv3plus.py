"""Train DeepLabV3+ with a pretrained MobileNetV3-Large encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.transforms import functional as TF
from tqdm import tqdm

try:
    from .train_unet import LesionDataset, batch_metrics, collect_pairs, load_or_create_splits
except ImportError:
    from train_unet import LesionDataset, batch_metrics, collect_pairs, load_or_create_splits


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CLASS_NAMES = (
    "Apple_Apple_scab",
    "Apple_Black_rot",
    "Apple_Cedar_apple_rust",
    "Potato_Early_blight",
    "Potato_Late_blight",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset_segmentation_plantseg"))
    parser.add_argument("--extra-train-data", type=Path, default=Path("dataset_segmentation"))
    parser.add_argument("--output", type=Path, default=Path("runs/deeplabv3plus_plantseg"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_name(stem: str) -> str:
    if "__" in stem:
        return stem.split("__", 1)[0]
    for name in CLASS_NAMES:
        if stem.startswith(name):
            return name
    return "manual_other"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(path: Path) -> int:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    values = np.asarray(image, dtype=np.int16)
    bits = (values[:, 1:] > values[:, :-1]).reshape(-1)
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def add_unique_manual_pairs(
    base_pairs: list[tuple[Path, Path]], extra_data: Path | None
) -> tuple[list[tuple[Path, Path]], dict[str, object]]:
    if extra_data is None or not extra_data.is_dir():
        return [], {"available": 0, "added": 0, "duplicates": []}
    extra_pairs = collect_pairs(extra_data)
    comparison_pairs = [pair for pair in base_pairs if class_name(pair[0].stem) == "Apple_Apple_scab"]
    base_hashes = {file_sha256(image) for image, _ in comparison_pairs}
    base_dhashes = [difference_hash(image) for image, _ in comparison_pairs]
    added = []
    duplicates = []
    for pair in extra_pairs:
        image = pair[0]
        exact_duplicate = file_sha256(image) in base_hashes
        fingerprint = difference_hash(image)
        near_duplicate = any((fingerprint ^ known).bit_count() <= 2 for known in base_dhashes)
        if exact_duplicate or near_duplicate:
            duplicates.append(image.stem)
        else:
            added.append(pair)
    return added, {
        "available": len(extra_pairs),
        "added": len(added),
        "duplicates": duplicates,
        "rule": "SHA-256 match or dHash Hamming distance <= 2 against all PlantSeg Apple scab images",
    }


class NormalizedLesionDataset(LesionDataset):
    def __getitem__(self, index: int):
        image, mask, name = super().__getitem__(index)
        return TF.normalize(image, MEAN, STD), mask, name


class ConvGNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        padding = dilation if kernel_size == 3 else 0
        groups = min(32, out_channels)
        while out_channels % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 128):
        super().__init__()
        self.branches = nn.ModuleList(
            [ConvGNReLU(in_channels, out_channels, 1)]
            + [ConvGNReLU(in_channels, out_channels, 3, rate) for rate in (6, 12, 18)]
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvGNReLU(in_channels, out_channels, 1))
        self.project = nn.Sequential(
            ConvGNReLU(out_channels * 5, out_channels, 1),
            nn.Dropout(0.15),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        pooled = F.interpolate(self.pool(x), size=size, mode="bilinear", align_corners=False)
        return self.project(torch.cat([*(branch(x) for branch in self.branches), pooled], dim=1))


class DeepLabV3PlusMobileNetV3(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.encoder = mobilenet_v3_large(weights=weights, dilated=True).features
        self.aspp = ASPP(960, 128)
        self.low_projection = ConvGNReLU(24, 48, 1)
        self.decoder = nn.Sequential(
            ConvGNReLU(176, 128),
            nn.Dropout(0.10),
            ConvGNReLU(128, 128),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        low = None
        features = x
        for index, layer in enumerate(self.encoder):
            features = layer(features)
            if index == 3:
                low = features
        assert low is not None
        high = self.aspp(features)
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        output = self.decoder(torch.cat([high, self.low_projection(low)], dim=1))
        return F.interpolate(output, size=input_size, mode="bilinear", align_corners=False)


class FocalDiceLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, smooth: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probability = torch.sigmoid(logits)
        pt = probability * target + (1 - probability) * (1 - target)
        focal = ((1 - pt).pow(self.gamma) * bce).mean()
        dims = (1, 2, 3)
        intersection = (probability * target).sum(dims)
        denominator = probability.sum(dims) + target.sum(dims)
        dice = 1 - ((2 * intersection + self.smooth) / (denominator + self.smooth)).mean()
        return 0.45 * focal + 0.55 * dice


def make_balanced_sampler(pairs: list[tuple[Path, Path]], seed: int) -> WeightedRandomSampler:
    labels = [class_name(image.stem) for image, _ in pairs]
    counts = Counter(labels)
    weights = torch.tensor([1.0 / counts[label] for label in labels], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    accumulate: int = 1,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "pixel_accuracy": 0.0}
    samples = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, (images, masks, _) in enumerate(tqdm(loader, leave=False), start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        batch_size = images.size(0)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            if training:
                assert scaler is not None and optimizer is not None
                scaler.scale(loss / accumulate).backward()
                if step % accumulate == 0 or step == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        dice, iou, accuracy = batch_metrics(logits.detach(), masks, threshold)
        totals["loss"] += loss.item() * batch_size
        totals["dice"] += dice * batch_size
        totals["iou"] += iou * batch_size
        totals["pixel_accuracy"] += accuracy * batch_size
        samples += batch_size
    return {key: value / samples for key, value in totals.items()}


@torch.no_grad()
def collect_probabilities(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    batches = []
    for images, masks, names in tqdm(loader, leave=False, desc="Threshold tuning"):
        probability = torch.sigmoid(model(images.to(device))).cpu()
        batches.append((probability, masks, list(names)))
    return batches


def metrics_from_batches(batches, threshold: float) -> dict[str, float]:
    dice_values, iou_values, accuracy_values = [], [], []
    for probabilities, masks, _ in batches:
        prediction = probabilities >= threshold
        truth = masks >= 0.5
        dims = (1, 2, 3)
        intersection = (prediction & truth).sum(dims).float()
        union = (prediction | truth).sum(dims).float()
        pred_sum = prediction.sum(dims).float()
        truth_sum = truth.sum(dims).float()
        dice_values.extend(((2 * intersection + 1e-7) / (pred_sum + truth_sum + 1e-7)).tolist())
        iou_values.extend(((intersection + 1e-7) / (union + 1e-7)).tolist())
        accuracy_values.extend((prediction == truth).float().mean((1, 2, 3)).tolist())
    return {
        "dice": float(np.mean(dice_values)),
        "iou": float(np.mean(iou_values)),
        "pixel_accuracy": float(np.mean(accuracy_values)),
    }


def save_curves(history: list[dict[str, object]], output: Path) -> None:
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
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    data = args.data.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pairs = collect_pairs(data)
    splits, split_source = load_or_create_splits(pairs, data, data / "splits.json", args.seed)
    extra_pairs, extra_report = add_unique_manual_pairs(
        pairs, args.extra_train_data.resolve() if args.extra_train_data else None
    )
    splits["train"] = splits["train"] + extra_pairs
    (output / "extra_train_report.json").write_text(
        json.dumps(extra_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "splits.json").write_text(
        json.dumps(
            {name: [image.stem for image, _ in subset] for name, subset in splits.items()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    datasets = {
        name: NormalizedLesionDataset(subset, args.image_size, augment=name == "train")
        for name, subset in splits.items()
    }
    pin_memory = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            sampler=make_balanced_sampler(splits["train"], args.seed),
            num_workers=args.workers,
            pin_memory=pin_memory,
            persistent_workers=args.workers > 0,
        ),
        **{
            name: DataLoader(
                datasets[name],
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=pin_memory,
                persistent_workers=args.workers > 0,
            )
            for name in ("val", "test")
        },
    }
    print(
        f"Pairs: base={len(pairs)}, manual_added={len(extra_pairs)} | "
        f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
    )
    print(f"Split: {split_source}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3PlusMobileNetV3(pretrained=not args.no_pretrained).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device} | parameters={parameter_count:,}")
    if args.dry_run:
        images, masks, names = next(iter(loaders["train"]))
        images = images.to(device)
        masks = masks.to(device)
        dry_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            dry_loss = FocalDiceLoss()(logits, masks)
        dry_scaler.scale(dry_loss).backward()
        print(
            f"Dry run forward/backward ok: image={tuple(images.shape)}, "
            f"mask={tuple(masks.shape)}, logits={tuple(logits.shape)}, "
            f"loss={dry_loss.item():.4f}, sample={names[0]}"
        )
        return

    criterion = FocalDiceLoss()
    encoder_parameters = list(model.encoder.parameters())
    head_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.lr * 0.25},
            {"params": head_parameters, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = []
    best_dice = -1.0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, loaders["train"], criterion, device, 0.5, optimizer, scaler, args.accumulate
        )
        val_metrics = run_epoch(model, loaders["val"], criterion, device, 0.5)
        scheduler.step()
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"Epoch {epoch:03d}: train_loss={train_metrics['loss']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} | val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f}"
        )
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": "deeplabv3plus_mobilenetv3_large",
                    "image_size": args.image_size,
                    "threshold": 0.5,
                    "best_val_dice": best_dice,
                    "parameter_count": parameter_count,
                    "pretrained_encoder": not args.no_pretrained,
                    "task": "binary_lesion_segmentation",
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
    val_batches = collect_probabilities(model, loaders["val"], device)
    candidates = np.round(np.arange(0.25, 0.76, 0.02), 2)
    threshold_rows = [
        {"threshold": float(value), **metrics_from_batches(val_batches, float(value))}
        for value in candidates
    ]
    best_threshold = max(threshold_rows, key=lambda row: row["dice"])["threshold"]
    checkpoint["threshold"] = best_threshold
    torch.save(checkpoint, output / "best.pt")
    test_batches = collect_probabilities(model, loaders["test"], device)
    test_metrics = metrics_from_batches(test_batches, best_threshold)

    per_class = defaultdict(list)
    for probabilities, masks, names in test_batches:
        prediction = probabilities >= best_threshold
        truth = masks >= 0.5
        dims = (1, 2, 3)
        intersection = (prediction & truth).sum(dims).float()
        dice = (2 * intersection + 1e-7) / (
            prediction.sum(dims).float() + truth.sum(dims).float() + 1e-7
        )
        for name, value in zip(names, dice.tolist()):
            per_class[class_name(name)].append(value)

    metrics = {
        "device": str(device),
        "base_pairs": len(pairs),
        "manual_annotations": extra_report,
        "split_sizes": {name: len(values) for name, values in splits.items()},
        "parameter_count": parameter_count,
        "best_val_dice_at_0.5": best_dice,
        "best_threshold": best_threshold,
        "test": test_metrics,
        "test_dice_by_class": {
            name: float(np.mean(values)) for name, values in sorted(per_class.items())
        },
        "threshold_search": threshold_rows,
        "history": history,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_curves(history, output / "training_curves.png")
    print(
        f"Test: dice={test_metrics['dice']:.4f}, iou={test_metrics['iou']:.4f}, "
        f"pixel_accuracy={test_metrics['pixel_accuracy']:.4f}, threshold={best_threshold:.2f}"
    )
    print(f"Saved outputs to {output}")


if __name__ == "__main__":
    main()
