"""Train MobileNetV3-Small or ResNet18 on an ImageFolder dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset_classification"))
    parser.add_argument("--output", type=Path, default=Path("runs/mobilenet_v3_small"))
    parser.add_argument("--model", choices=("mobilenet_v3_small", "resnet18"), default="mobilenet_v3_small")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--class-weighting", choices=("balanced", "none"), default="balanced",
        help="Use inverse-frequency class weights to reduce majority-class bias",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(data: Path, size: int, batch_size: int, workers: int):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.ToTensor(), normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(size * 1.14)), transforms.CenterCrop(size),
        transforms.ToTensor(), normalize,
    ])
    sets = {
        "train": datasets.ImageFolder(data / "train", train_tf),
        "val": datasets.ImageFolder(data / "val", eval_tf),
        "test": datasets.ImageFolder(data / "test", eval_tf),
    }
    if sets["train"].class_to_idx != sets["val"].class_to_idx or sets["train"].class_to_idx != sets["test"].class_to_idx:
        raise RuntimeError("Class folders differ across train/val/test")
    pin = torch.cuda.is_available()
    loaders = {
        split: DataLoader(ds, batch_size=batch_size, shuffle=split == "train", num_workers=workers,
                          pin_memory=pin, persistent_workers=workers > 0)
        for split, ds in sets.items()
    }
    return sets, loaders


def build_model(name: str, classes: int, pretrained: bool) -> nn.Module:
    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, classes)
    elif name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, classes)
    elif name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, classes)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = total = 0
    predictions, targets = [], []
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        pred = logits.argmax(1)
        loss_sum += loss.item() * labels.size(0)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        predictions.extend(pred.cpu().tolist())
        targets.extend(labels.cpu().tolist())
    return {"loss": loss_sum / total, "accuracy": correct / total}, targets, predictions


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    sets, loaders = make_loaders(args.data, args.image_size, args.batch_size, args.workers)
    classes = sets["train"].classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, len(classes), not args.no_pretrained).to(device)
    if args.class_weighting == "balanced":
        counts = np.bincount(sets["train"].targets, minlength=len(classes))
        class_weights = counts.sum() / (len(classes) * counts)
        loss_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        print("Class weights:", dict(zip(classes, class_weights.round(3).tolist())))
    else:
        loss_weights = None
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history, best_accuracy, stale = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_metrics, _, _ = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step()
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(f"Epoch {epoch:02d}: train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f}")
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy, stale = val_metrics["accuracy"], 0
            torch.save({
                "model_state": model.state_dict(), "model_name": args.model,
                "classes": classes, "image_size": args.image_size,
                "val_accuracy": best_accuracy,
            }, args.output / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, y_true, y_pred = run_epoch(model, loaders["test"], criterion, device)
    report = classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred).tolist()
    metrics = {"device": str(device), "model": args.model, "classes": classes,
               "class_weighting": args.class_weighting,
               "best_val_accuracy": best_accuracy, "test": test_metrics,
               "classification_report": report, "confusion_matrix": matrix, "history": history}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.figure(figsize=(8, 4))
    plt.plot([r["epoch"] for r in history], [r["train"]["accuracy"] for r in history], label="train")
    plt.plot([r["epoch"] for r in history], [r["val"]["accuracy"] for r in history], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(args.output / "accuracy.png", dpi=160); plt.close()
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Saved outputs to {args.output.resolve()}")


if __name__ == "__main__":
    main()

