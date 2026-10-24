"""Select a balanced set of diseased PlantVillage images for Labelme annotation."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


DISEASE_CLASSES = (
    "Apple_Apple_scab",
    "Apple_Black_rot",
    "Apple_Cedar_apple_rust",
    "Potato_Early_blight",
    "Potato_Late_blight",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-data", type=Path, default=Path("dataset_classification"))
    parser.add_argument("--output", type=Path, default=Path("dataset_segmentation"))
    parser.add_argument("--samples-per-class", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_class <= 0:
        raise ValueError("--samples-per-class must be positive")
    source = args.classification_data.resolve()
    output = args.output.resolve()
    labelme_dir = output / "labelme"

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}; use --overwrite to replace it")
        shutil.rmtree(output)
    labelme_dir.mkdir(parents=True)

    rng = random.Random(args.seed)
    rows: list[dict[str, str]] = []
    for class_name in DISEASE_CLASSES:
        candidates: list[Path] = []
        for split in ("train", "val", "test"):
            class_dir = source / split / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            candidates.extend(
                path for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        if len(candidates) < args.samples_per_class:
            raise RuntimeError(
                f"{class_name} has {len(candidates)} images, fewer than requested "
                f"{args.samples_per_class}"
            )
        candidates.sort()
        selected = rng.sample(candidates, args.samples_per_class)
        for index, src in enumerate(selected, start=1):
            filename = f"{class_name}_{index:04d}{src.suffix.lower()}"
            dst = labelme_dir / filename
            shutil.copy2(src, dst)
            rows.append({
                "image": filename,
                "disease_class": class_name,
                "source": str(src),
                "annotation_status": "pending",
            })

    with (output / "annotation_manifest.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Prepared {len(rows)} images in {labelme_dir}")
    print("Open this directory in Labelme and label every lesion polygon as: lesion")


if __name__ == "__main__":
    main()

