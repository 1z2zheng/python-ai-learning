"""Extract selected PlantVillage classes and create reproducible splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path


CLASS_ALIASES = {
    "Tomato___healthy": "Tomato_healthy",
    "Tomato___Early_blight": "Tomato_Early_blight",
    "Tomato___Late_blight": "Tomato_Late_blight",
    "Tomato___Leaf_Mold": "Tomato_Leaf_Mold",
    "Potato___healthy": "Potato_healthy",
    "Potato___Early_blight": "Potato_Early_blight",
    "Potato___Late_blight": "Potato_Late_blight",
    "Apple___healthy": "Apple_healthy",
    "Apple___Apple_scab": "Apple_Apple_scab",
    "Apple___Black_rot": "Apple_Black_rot",
    "Apple___Cedar_apple_rust": "Apple_Cedar_apple_rust",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="PlantVillage root")
    parser.add_argument("--output", type=Path, default=Path("dataset_classification"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--mode", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_class_dirs(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_dir() and path.name in CLASS_ALIASES:
            if path.name in found:
                raise RuntimeError(f"Duplicate class directory: {path.name}")
            found[path.name] = path
    missing = sorted(set(CLASS_ALIASES) - set(found))
    if missing:
        raise FileNotFoundError("Missing PlantVillage classes:\n  " + "\n  ".join(missing))
    return found


def split_items(items: list[Path], train_ratio: float, val_ratio: float) -> dict[str, list[Path]]:
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": items[:n_train],
        "val": items[n_train : n_train + n_val],
        "test": items[n_train + n_val :],
    }


def deduplicate_images(images: list[Path]) -> tuple[list[Path], int]:
    """Keep one file for each exact image content within a class."""
    seen: set[str] = set()
    unique: list[Path] = []
    for path in images:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(path)
    return unique, len(images) - len(unique)


def main() -> None:
    args = parse_args()
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Ratios must be positive and train + val must be less than 1")
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}; use --overwrite to replace it")
        shutil.rmtree(output)

    class_dirs = find_class_dirs(source)
    rng = random.Random(args.seed)
    counts: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()
    manifest: list[dict[str, str]] = []

    for source_name, target_name in CLASS_ALIASES.items():
        images = sorted(
            p for p in class_dirs[source_name].iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        images, duplicate_counts[target_name] = deduplicate_images(images)
        if len(images) < 3:
            raise RuntimeError(f"Class {source_name} contains fewer than 3 images")
        rng.shuffle(images)
        for split, split_images in split_items(images, args.train_ratio, args.val_ratio).items():
            target_dir = output / split / target_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for index, src in enumerate(split_images):
                # Prefixing prevents collisions if a source dataset contains repeated names.
                dst = target_dir / f"{index:05d}_{src.name}"
                if args.mode == "hardlink":
                    dst.hardlink_to(src)
                else:
                    shutil.copy2(src, dst)
                counts[f"{split}/{target_name}"] += 1
                manifest.append({"split": split, "class": target_name, "source": str(src), "file": str(dst)})

    (output / "classes.json").write_text(
        json.dumps(list(CLASS_ALIASES.values()), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Created dataset at {output}")
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")
    removed = sum(duplicate_counts.values())
    print(f"Removed {removed} exact duplicate image(s) before splitting")
    for class_name, count in sorted(duplicate_counts.items()):
        if count:
            print(f"duplicate/{class_name}: {count}")


if __name__ == "__main__":
    main()

