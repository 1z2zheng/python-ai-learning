"""Train a compact binary U-Net on image/mask pairs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset_segmentation"))
    parser.add_argument("--output", type=Path, default=Path("runs/unet_apple_scab"))
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Optional JSON containing train/val/test image stems. "
        "Defaults to <data>/splits.json when present.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_pairs(data: Path) -> list[tuple[Path, Path]]:
    images_dir = data / "images"
    masks_dir = data / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError("Expected dataset_segmentation/images and masks")

    images: dict[str, Path] = {}
    for extension in IMAGE_EXTENSIONS:
        for path in images_dir.glob(f"*{extension}"):
            images[path.stem] = path
        for path in images_dir.glob(f"*{extension.upper()}"):
            images[path.stem] = path
    masks = {path.stem: path for path in masks_dir.glob("*.png")}
    missing_images = sorted(set(masks) - set(images))
    missing_masks = sorted(set(images) - set(masks))
    if missing_images or missing_masks:
        raise RuntimeError(
            f"Unpaired files: missing_images={missing_images[:5]}, "
            f"missing_masks={missing_masks[:5]}"
        )
    pairs = [(images[stem], masks[stem]) for stem in sorted(images)]
    if len(pairs) < 10:
        raise RuntimeError(f"At least 10 pairs are required, found {len(pairs)}")
    return pairs


def split_pairs(
    pairs: list[tuple[Path, Path]], seed: int
) -> dict[str, list[tuple[Path, Path]]]:
    shuffled = pairs.copy()
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    test_count = max(1, round(total * 0.10))
    val_count = max(1, round(total * 0.20))
    train_count = total - val_count - test_count
    if train_count < 1:
        raise RuntimeError("Dataset is too small for train/val/test split")
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def load_or_create_splits(
    pairs: list[tuple[Path, Path]], data: Path, split_file: Path | None, seed: int
) -> tuple[dict[str, list[tuple[Path, Path]]], str]:
    manifest_path = split_file.resolve() if split_file else data / "splits.json"
    if not manifest_path.is_file():
        return split_pairs(pairs, seed), "random 70/20/10 split"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest) != {"train", "val", "test"}:
        raise ValueError(f"Split file must contain train/val/test: {manifest_path}")

    pair_by_stem = {image.stem: (image, mask) for image, mask in pairs}
    assigned: list[str] = []
    splits: dict[str, list[tuple[Path, Path]]] = {}
    for split_name in ("train", "val", "test"):
        stems = manifest[split_name]
        if not isinstance(stems, list) or not stems:
            raise ValueError(f"Split '{split_name}' is empty or invalid")
        missing = sorted(set(stems) - set(pair_by_stem))
        if missing:
            raise ValueError(
                f"Split '{split_name}' references missing files: {missing[:5]}"
            )
        splits[split_name] = [pair_by_stem[stem] for stem in stems]
        assigned.extend(stems)

    duplicates = sorted({stem for stem in assigned if assigned.count(stem) > 1})
    unassigned = sorted(set(pair_by_stem) - set(assigned))
    if duplicates or unassigned or len(assigned) != len(pair_by_stem):
        raise ValueError(
            f"Invalid split coverage: duplicates={duplicates[:5]}, "
            f"unassigned={unassigned[:5]}"
        )
    return splits, f"official split from {manifest_path}"


class LesionDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        image_size: int,
        augment: bool,
    ) -> None:
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment
        self.color_jitter = ColorJitter(
            brightness=0.18, contrast=0.18, saturation=0.12, hue=0.03
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(mask_path) as source:
            mask = source.convert("L")

        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        if self.augment:
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() < 0.25:
                image, mask = TF.vflip(image), TF.vflip(mask)
            angle = random.choice((0, 90, 180, 270))
            if angle:
                image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
            image = self.color_jitter(image)

        image_tensor = TF.to_tensor(image)
        mask_tensor = (TF.pil_to_tensor(mask).float() / 255.0 >= 0.5).float()
        return image_tensor, mask_tensor, image_path.stem


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = TF.resize(x, list(skip.shape[-2:]), antialias=True)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self, base: int = 16) -> None:
        super().__init__()
        self.inc = DoubleConv(3, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 16)
        self.up1 = Up(base * 16, base * 8, base * 8)
        self.up2 = Up(base * 8, base * 4, base * 4)
        self.up3 = Up(base * 4, base * 2, base * 2)
        self.up4 = Up(base * 2, base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)


class BCEDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        probabilities = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probabilities * target).sum(dims)
        denominator = probabilities.sum(dims) + target.sum(dims)
        dice_loss = 1.0 - (
            (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        ).mean()
        return 0.5 * bce + 0.5 * dice_loss


def batch_metrics(
    logits: torch.Tensor, target: torch.Tensor, threshold: float
) -> tuple[float, float, float]:
    prediction = torch.sigmoid(logits) >= threshold
    truth = target >= 0.5
    dims = (1, 2, 3)
    intersection = (prediction & truth).sum(dims).float()
    pred_sum = prediction.sum(dims).float()
    truth_sum = truth.sum(dims).float()
    union = (prediction | truth).sum(dims).float()
    dice = ((2 * intersection + 1e-7) / (pred_sum + truth_sum + 1e-7)).mean()
    iou = ((intersection + 1e-7) / (union + 1e-7)).mean()
    accuracy = (prediction == truth).float().mean()
    return dice.item(), iou.item(), accuracy.item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "pixel_accuracy": 0.0}
    samples = 0

    for images, masks, _ in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        batch_size = images.size(0)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        dice, iou, accuracy = batch_metrics(logits.detach(), masks, threshold)
        totals["loss"] += loss.item() * batch_size
        totals["dice"] += dice * batch_size
        totals["iou"] += iou * batch_size
        totals["pixel_accuracy"] += accuracy * batch_size
        samples += batch_size
    return {key: value / samples for key, value in totals.items()}


@torch.no_grad()
def save_prediction_figure(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    output: Path,
) -> None:
    model.eval()
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    for images, masks, names in loader:
        probabilities = torch.sigmoid(model(images.to(device))).cpu()
        for image, mask, probability, name in zip(images, masks, probabilities, names):
            rows.append(
                (
                    image.permute(1, 2, 0).numpy(),
                    mask.squeeze(0).numpy(),
                    (probability.squeeze(0).numpy() >= threshold).astype(np.float32),
                    name,
                )
            )
            if len(rows) >= 4:
                break
        if len(rows) >= 4:
            break

    figure, axes = plt.subplots(len(rows), 4, figsize=(12, 3 * len(rows)), squeeze=False)
    for row_index, (image, truth, prediction, name) in enumerate(rows):
        overlay = image.copy()
        overlay[prediction > 0] = 0.55 * overlay[prediction > 0] + 0.45 * np.array([1.0, 0.0, 0.0])
        panels = (image, truth, prediction, overlay)
        titles = (name, "Ground truth", "Prediction", "Overlay")
        for column, (panel, title) in enumerate(zip(panels, titles)):
            axes[row_index, column].imshow(panel, cmap="gray" if panel.ndim == 2 else None)
            axes[row_index, column].set_title(title)
            axes[row_index, column].axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = collect_pairs(args.data.resolve())
    splits, split_source = load_or_create_splits(
        pairs, args.data.resolve(), args.split_file, args.seed
    )
    split_manifest = {
        name: [image.stem for image, _ in subset] for name, subset in splits.items()
    }
    (args.output / "splits.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    datasets = {
        name: LesionDataset(subset, args.image_size, augment=name == "train")
        for name, subset in splits.items()
    }
    pin_memory = torch.cuda.is_available()
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.workers,
            pin_memory=pin_memory,
            persistent_workers=args.workers > 0,
        )
        for name, dataset in datasets.items()
    }
    print(
        f"Pairs: {len(pairs)} | train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )
    print(f"Split: {split_source}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(args.base_channels).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device} | parameters={parameter_count:,}")

    if args.dry_run:
        images, masks, names = next(iter(loaders["train"]))
        with torch.no_grad():
            output = model(images[:1].to(device))
        print(
            f"Dry run ok: image={tuple(images[:1].shape)}, "
            f"mask={tuple(masks[:1].shape)}, logits={tuple(output.shape)}, "
            f"sample={names[0]}"
        )
        return

    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history: list[dict[str, object]] = []
    best_dice = -1.0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, loaders["train"], criterion, device, args.threshold, optimizer, scaler
        )
        val_metrics = run_epoch(
            model, loaders["val"], criterion, device, args.threshold
        )
        scheduler.step(val_metrics["dice"])
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f}"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": "unet",
                    "base_channels": args.base_channels,
                    "image_size": args.image_size,
                    "threshold": args.threshold,
                    "best_val_dice": best_dice,
                    "parameter_count": parameter_count,
                    "task": "binary_lesion_segmentation",
                    "scope": "PlantSeg three-crop binary lesion segmentation",
                },
                args.output / "best.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(
        args.output / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(
        model, loaders["test"], criterion, device, args.threshold
    )
    metrics = {
        "device": str(device),
        "pairs": len(pairs),
        "split_sizes": {name: len(values) for name, values in splits.items()},
        "parameter_count": parameter_count,
        "best_val_dice": best_dice,
        "test": test_metrics,
        "history": history,
        "split_source": split_source,
        "note": "PlantSeg binary lesion segmentation for Apple, Potato and Tomato.",
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

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
    figure.savefig(args.output / "training_curves.png", dpi=160)
    plt.close(figure)

    save_prediction_figure(
        model,
        loaders["test"],
        device,
        args.threshold,
        args.output / "test_predictions.png",
    )
    print(
        f"Test: loss={test_metrics['loss']:.4f}, "
        f"dice={test_metrics['dice']:.4f}, iou={test_metrics['iou']:.4f}, "
        f"pixel_accuracy={test_metrics['pixel_accuracy']:.4f}"
    )
    print(f"Saved outputs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
