"""Extract the eight project disease classes from PlantSeg v7."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps


PROJECT = Path(__file__).resolve().parents[1]
ARCHIVE = PROJECT / "downloads" / "plantseg-v7.zip"
OUTPUT = PROJECT / "dataset_segmentation_plantseg"

CLASS_MAP = {
    "apple_scab": "Apple_Apple_scab",
    "apple_black_rot": "Apple_Black_rot",
    "apple_rust": "Apple_Cedar_apple_rust",
    "potato_early_blight": "Potato_Early_blight",
    "potato_late_blight": "Potato_Late_blight",
    "tomato_early_blight": "Tomato_Early_blight",
    "tomato_late_blight": "Tomato_Late_blight",
    "tomato_leaf_mold": "Tomato_Leaf_Mold",
}
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def match_class(filename: str) -> tuple[str, str] | None:
    lower = filename.lower()
    for source_class in sorted(CLASS_MAP, key=len, reverse=True):
        if lower.startswith(source_class + "_"):
            return source_class, CLASS_MAP[source_class]
    return None


def main() -> None:
    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)
    images_dir = OUTPUT / "images"
    masks_dir = OUTPUT / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    split_names: dict[str, list[str]] = {split: [] for split in SPLITS}
    counts: Counter[tuple[str, str]] = Counter()

    with zipfile.ZipFile(ARCHIVE) as archive:
        members = archive.namelist()
        image_index: dict[tuple[str, str], str] = {}
        for member in members:
            path = Path(member)
            if len(path.parts) != 4 or path.parts[1] != "images":
                continue
            split = path.parts[2]
            if split not in SPLITS or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image_index[(split, path.stem)] = member

        annotation_members = []
        for member in members:
            path = Path(member)
            if (
                len(path.parts) == 4
                and path.parts[1] == "annotations"
                and path.parts[2] in SPLITS
                and path.suffix.lower() == ".png"
                and match_class(path.name) is not None
            ):
                annotation_members.append(member)

        for annotation_member in sorted(annotation_members):
            annotation_path = Path(annotation_member)
            split = annotation_path.parts[2]
            source_class, target_class = match_class(annotation_path.name)  # type: ignore[misc]
            image_member = image_index.get((split, annotation_path.stem))
            if image_member is None:
                raise FileNotFoundError(
                    f"No image for {annotation_member} in split {split}"
                )
            image_suffix = Path(image_member).suffix.lower()
            output_stem = f"{target_class}__{annotation_path.stem}"
            output_image = images_dir / f"{output_stem}{image_suffix}"
            output_mask = masks_dir / f"{output_stem}.png"

            with Image.open(io.BytesIO(archive.read(image_member))) as source_image:
                normalized_image = ImageOps.exif_transpose(source_image).convert("RGB")
                normalized_image.save(output_image, quality=95)

            with Image.open(io.BytesIO(archive.read(annotation_member))) as mask:
                grayscale = mask.convert("L")
                binary = grayscale.point(lambda value: 255 if value != 0 else 0)
                binary.save(output_mask)

            split_names[split].append(output_stem)
            counts[(target_class, split)] += 1
            records.append(
                {
                    "name": output_stem,
                    "target_class": target_class,
                    "source_class": source_class,
                    "split": split,
                    "image": output_image.name,
                    "mask": output_mask.name,
                    "source_image": image_member,
                    "source_annotation": annotation_member,
                }
            )

    with (OUTPUT / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (OUTPUT / "splits.json").write_text(
        json.dumps(split_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "source": "PlantSeg v7, Zenodo record 17719108",
        "archive": str(ARCHIVE),
        "mask_encoding": {"background": 0, "lesion": 255},
        "total_pairs": len(records),
        "classes": {
            target: {
                split: counts[(target, split)] for split in SPLITS
            }
            for target in CLASS_MAP.values()
        },
        "note": "PlantSeg apple_rust is mapped to Apple_Cedar_apple_rust for this project.",
    }
    (OUTPUT / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Extracted {len(records)} image/mask pairs to {OUTPUT}")
    for target in CLASS_MAP.values():
        values = [counts[(target, split)] for split in SPLITS]
        print(f"{target}: train={values[0]}, val={values[1]}, test={values[2]}, total={sum(values)}")


if __name__ == "__main__":
    main()
