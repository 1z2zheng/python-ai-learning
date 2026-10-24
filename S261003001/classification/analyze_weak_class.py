"""Audit one weak class and compare two checkpoints on a fixed PlantSeg split.

This script only reads datasets and checkpoints. It writes a per-image CSV,
summary JSON, and a contact sheet to an output directory. Use validation for
model development; test output is descriptive and must not guide tuning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from classification.train_mixed import (  # noqa: E402
    MixedDataset,
    difference_hash,
    make_transforms,
    plantseg_records,
)
from inference.predict import DEFAULT_CLASSIFIER, load_classifier  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "dataset_segmentation_plantseg")
    parser.add_argument("--plantvillage", type=Path, default=PROJECT / "dataset_classification")
    parser.add_argument("--extra", type=Path, default=PROJECT / "dataset_natural_extra")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--target", default="Apple_Black_rot")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--baseline", type=Path, default=PROJECT / "runs/mobilenet_v3_large_mixed/best.pt")
    parser.add_argument("--output", type=Path, default=PROJECT / "runs/weak_class_audit")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def predict(checkpoint_path: Path, records, device: torch.device, batch_size: int):
    model, classes, _transform, checkpoint = load_classifier(checkpoint_path, device)
    _, eval_transform = make_transforms(int(checkpoint.get("image_size", 224)))
    loader = DataLoader(MixedDataset(records, eval_transform), batch_size=batch_size, shuffle=False)
    results = []
    with torch.inference_mode():
        for images, _labels, _domains, _stems in loader:
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images.to(device))
            probabilities = logits.softmax(dim=1).cpu()
            values, indices = probabilities.topk(3, dim=1)
            for scores, positions in zip(values.tolist(), indices.tolist()):
                results.append(
                    {
                        "class": classes[positions[0]],
                        "confidence": scores[0],
                        "top3": [
                            {"class": classes[position], "confidence": score}
                            for score, position in zip(scores, positions)
                        ],
                    }
                )
    return results


def target_paths(root: Path, plantvillage: Path, extra: Path, split: str, target: str):
    image_root = root / "images"
    stems_by_split = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    images = {path.stem: path for path in image_root.iterdir() if path.is_file()}
    paths = {}
    for part in ("train", "val", "test"):
        paths[f"PlantSeg {part}"] = [
            images[stem] for stem in stems_by_split[part]
            if stem.startswith(target + "__")
        ]
        paths[f"PlantVillage {part}"] = sorted(
            path for path in (plantvillage / part / target).iterdir() if path.is_file()
        )
    extra_dir = extra / "train" / target
    paths["PlantDoc train"] = sorted(path for path in extra_dir.iterdir() if path.is_file()) if extra_dir.is_dir() else []
    return paths


def audit_images(groups: dict[str, list[Path]]) -> dict:
    digest_groups: dict[str, list[str]] = {}
    unreadable = []
    dimensions: dict[str, list[tuple[int, int]]] = {}
    for group_name, paths in groups.items():
        dimensions[group_name] = []
        for path in paths:
            try:
                with Image.open(path) as source:
                    source.verify()
                with Image.open(path) as source:
                    dimensions[group_name].append(source.size)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                digest_groups.setdefault(digest, []).append(f"{group_name}: {path.name}")
            except (OSError, ValueError) as error:
                unreadable.append({"path": str(path), "error": str(error)})

    duplicates = [items for items in digest_groups.values() if len(items) > 1]
    return {
        "counts": {name: len(paths) for name, paths in groups.items()},
        "image_sizes": {
            name: {
                "minimum": list(map(int, min(sizes, key=lambda size: size[0] * size[1]))),
                "maximum": list(map(int, max(sizes, key=lambda size: size[0] * size[1]))),
            } if sizes else None
            for name, sizes in dimensions.items()
        },
        "unreadable": unreadable,
        "exact_duplicate_groups": duplicates,
    }


def near_duplicates(groups: dict[str, list[Path]]) -> list[dict]:
    """Flag, never remove, dHash neighbors across train and held-out images."""
    train = [path for name, paths in groups.items() if name.endswith("train") for path in paths]
    held_out = [path for name, paths in groups.items() if name.endswith(("val", "test")) for path in paths]
    train_hashes = [(path, difference_hash(path)) for path in train]
    result = []
    for held_path in held_out:
        fingerprint = difference_hash(held_path)
        neighbors = sorted(
            ((fingerprint ^ candidate_hash).bit_count(), candidate)
            for candidate, candidate_hash in train_hashes
        )
        if neighbors and neighbors[0][0] <= 2:
            result.append({
                "held_out": str(held_path),
                "train_neighbor": str(neighbors[0][1]),
                "hamming_distance": neighbors[0][0],
            })
    return result


def contact_sheet(rows: list[dict], output: Path, target: str) -> None:
    selected = [row for row in rows if row["true_class"] == target]
    if not selected:
        return
    tile_width, tile_height, columns = 350, 290, 3
    lines = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, lines * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 15) if font_path.is_file() else ImageFont.load_default()
    for index, row in enumerate(selected):
        x = index % columns * tile_width
        y = index // columns * tile_height
        with Image.open(row["path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((tile_width - 16, 215))
        sheet.paste(image, (x + (tile_width - image.width) // 2, y + 4))
        color = "#155c36" if row["current_correct"] else "#bd3030"
        draw.text((x + 7, y + 222), Path(row["path"]).stem[-39:], fill="#222222", font=font)
        draw.text((x + 7, y + 242), f"Current: {row['current_class']} {row['current_confidence']:.0%}", fill=color, font=font)
        draw.text((x + 7, y + 262), f"Previous: {row['baseline_class']} {row['baseline_confidence']:.0%}", fill="#333333", font=font)
    sheet.save(output)


def training_contact_sheet(paths: list[Path], output: Path) -> None:
    if not paths:
        return
    tile_width, tile_height, columns = 210, 170, 5
    lines = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, lines * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 11) if font_path.is_file() else ImageFont.load_default()
    for index, path in enumerate(paths):
        x = index % columns * tile_width
        y = index // columns * tile_height
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((tile_width - 8, 142))
        sheet.paste(image, (x + (tile_width - image.width) // 2, y + 2))
        draw.text((x + 4, y + 146), path.stem[-31:], fill="#222222", font=font)
    sheet.save(output)


def main() -> None:
    args = arguments()
    output = args.output.resolve() / args.split
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    if args.target not in classes:
        raise ValueError(f"Target class is not in checkpoint: {args.target}")
    records = plantseg_records(args.data.resolve(), args.split, classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current = predict(args.checkpoint.resolve(), records, device, args.batch_size)
    baseline = predict(args.baseline.resolve(), records, device, args.batch_size)
    rows = []
    for record, now, before in zip(records, current, baseline, strict=True):
        rows.append({
            "path": str(record.path),
            "true_class": record.class_name,
            "current_class": now["class"],
            "current_confidence": now["confidence"],
            "current_correct": now["class"] == record.class_name,
            "current_top3": now["top3"],
            "baseline_class": before["class"],
            "baseline_confidence": before["confidence"],
            "baseline_correct": before["class"] == record.class_name,
        })
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "path", "true_class", "current_class", "current_confidence", "current_correct",
            "current_top3", "baseline_class", "baseline_confidence", "baseline_correct",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "current_top3": json.dumps(row["current_top3"])})

    target_rows = [row for row in rows if row["true_class"] == args.target]
    false_positives = [row for row in rows if row["current_class"] == args.target and row["true_class"] != args.target]
    groups = target_paths(args.data.resolve(), args.plantvillage.resolve(), args.extra.resolve(), args.split, args.target)
    inventory = audit_images(groups)
    inventory["near_duplicate_candidates"] = near_duplicates(groups)
    summary = {
        "split": args.split,
        "target": args.target,
        "split_samples": len(rows),
        "target_samples": len(target_rows),
        "current_correct": sum(row["current_correct"] for row in target_rows),
        "baseline_correct": sum(row["baseline_correct"] for row in target_rows),
        "current_confusions": dict(Counter(row["current_class"] for row in target_rows if not row["current_correct"])),
        "baseline_confusions": dict(Counter(row["baseline_class"] for row in target_rows if not row["baseline_correct"])),
        "false_positive_sources": dict(Counter(row["true_class"] for row in false_positives)),
        "inventory": inventory,
        "note": "Validation is for diagnosis and model selection. Test is descriptive only after model choices are frozen.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    contact_sheet(rows, output / "target_contact_sheet.jpg", args.target)
    training_contact_sheet(groups["PlantSeg train"], output / "plantseg_training_contact_sheet.jpg")
    print(json.dumps({key: value for key, value in summary.items() if key != "inventory"} | {"inventory_counts": inventory["counts"], "near_duplicate_candidates": len(inventory["near_duplicate_candidates"])}, ensure_ascii=False, indent=2))
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
