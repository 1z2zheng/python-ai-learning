"""Train spring-barley growth and lodging classifiers from Zenodo 15497780."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


META_RE = re.compile(
    r"^(\d+) >>> (\S+) >>>\s+sl_nr:\s+(\S+) >>> (\d+) samples >>> (.*)$"
)


class PatchCNN(nn.Module):
    def __init__(self, in_channels: int = 5, classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(128, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def read_metadata(path: Path) -> dict[int, dict]:
    result = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = META_RE.match(line.strip())
        if match:
            column, code, source, count, description = match.groups()
            result[int(column)] = {
                "code": code,
                "source": source,
                "count": int(count),
                "description": description.strip().lower(),
            }
    return result


def task_columns(metadata: dict[int, dict], task: str) -> dict[int, int]:
    columns = {}
    for column, item in metadata.items():
        description = item["description"]
        is_spring_barley = item["code"].startswith("SG_")
        if not is_spring_barley or item["count"] < 2 or "empty" in description:
            continue
        if task == "growth":
            if "lodged" in description or "weed" in description or "bare soil" in description:
                continue
            if "dry vegetation" in description:
                columns[column] = 1
            elif "vital" in description:
                columns[column] = 0
        elif task == "lodging":
            if "weed" in description or "bare soil" in description:
                continue
            if "lodged areas" in description:
                columns[column] = 1
            elif any(token in description for token in ("vital vegetation", "flowering", "ripening", "dry vegetation")):
                columns[column] = 0
        else:
            raise ValueError(task)
    return columns


def load_samples(tiff_path: Path, metadata: dict[int, dict], task: str):
    image = tifffile.memmap(tiff_path)
    columns = task_columns(metadata, task)
    samples, labels, groups = [], [], []
    for column, label in columns.items():
        count = metadata[column]["count"]
        for row in range(count):
            # B1-B5 are the five directly sensed CIR channels.  Derived colour
            # spaces are deliberately excluded so deployment accepts a normal
            # five-band UAV TIFF instead of a proprietary 16-band feature cube.
            patch = image[row * 30 : (row + 1) * 30, column * 30 : (column + 1) * 30, 2:7]
            if not np.isfinite(patch).any():
                continue
            samples.append(np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
            labels.append(label)
            groups.append((column, row // 8))
    return np.stack(samples), np.asarray(labels, dtype=np.int64), groups, columns


def split_indices(labels: np.ndarray, groups: list[tuple[int, int]], seed: int):
    rng = random.Random(seed)
    split_names = ("train", "val", "test")
    result = {name: [] for name in split_names}
    for label in sorted(set(labels.tolist())):
        unique = sorted({group for index, group in enumerate(groups) if labels[index] == label})
        rng.shuffle(unique)
        for number, group in enumerate(unique):
            fraction = number / max(len(unique), 1)
            split = "train" if fraction < 0.70 else "val" if fraction < 0.85 else "test"
            result[split].extend(index for index, value in enumerate(groups) if value == group and labels[index] == label)
    return {name: np.asarray(sorted(values), dtype=np.int64) for name, values in result.items()}


def standardize(train: np.ndarray, arrays: list[np.ndarray]):
    mean = train.mean(axis=(0, 1, 2), keepdims=True)
    std = train.std(axis=(0, 1, 2), keepdims=True)
    std = np.maximum(std, 1e-6)
    return [(array - mean) / std for array in arrays], mean.reshape(-1), std.reshape(-1)


def confusion_metrics(confusion: np.ndarray) -> dict:
    tp = np.diag(confusion).astype(float)
    precision = tp / np.maximum(confusion.sum(0), 1)
    recall = tp / np.maximum(confusion.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": float(tp.sum() / max(confusion.sum(), 1)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": confusion.tolist(),
    }


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    confusion = np.zeros((2, 2), dtype=np.int64)
    for images, labels in loader:
        predictions = model(images.to(device)).argmax(1).cpu().numpy()
        for target, prediction in zip(labels.numpy(), predictions):
            confusion[target, prediction] += 1
    return confusion_metrics(confusion)


def run(args, task: str, seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    metadata = read_metadata(args.metadata)
    images, labels, groups, columns = load_samples(args.tiff, metadata, task)
    indices = split_indices(labels, groups, seed)
    arrays = [images[indices[name]] for name in ("train", "val", "test")]
    arrays, mean, std = standardize(arrays[0], arrays)
    loaders = {}
    for name, array in zip(("train", "val", "test"), arrays):
        x = torch.from_numpy(np.ascontiguousarray(array)).permute(0, 3, 1, 2)
        y = torch.from_numpy(labels[indices[name]])
        loaders[name] = DataLoader(
            TensorDataset(x, y), batch_size=args.batch_size, shuffle=name == "train", num_workers=0
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchCNN(in_channels=images.shape[-1]).to(device)
    train_counts = Counter(labels[indices["train"]].tolist())
    weights = torch.tensor(
        [len(indices["train"]) / max(2 * train_counts[i], 1) for i in range(2)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_f1 = -1.0
    output = args.output / task / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch, target in loaders["train"]:
            batch, target = batch.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), target)
            loss.backward()
            optimizer.step()
        scheduler.step()
        val = evaluate(model, loaders["val"], device)
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "task": task,
                    "classes": ["normal_vital", "dry"] if task == "growth" else ["standing", "lodged"],
                    "input_bands": ["B1", "B2", "B3", "B4", "B5"],
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "seed": seed,
                },
                output / "best.pt",
            )
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"task={task} seed={seed} epoch={epoch} val_f1={val['macro_f1']:.4f}")

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result = {
        "task": task,
        "seed": seed,
        "classes": checkpoint["classes"],
        "source_columns": {str(key): metadata[key] for key in columns},
        "split_counts": {
            name: dict(Counter(map(str, labels[index].tolist()))) for name, index in indices.items()
        },
        "best_val_macro_f1": best_f1,
        "test": evaluate(model, loaders["test"], device),
    }
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiff", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=("growth", "lodging"), default=("growth", "lodging"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 123, 2026))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    results = [run(args, task, seed) for task in args.tasks for seed in args.seeds]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for result in results:
        print(result["task"], result["seed"], result["test"])


if __name__ == "__main__":
    main()
