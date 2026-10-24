"""Validate an ImageFolder dataset and write reproducible summary reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset_classification"))
    parser.add_argument("--skip-hash", action="store_true", help="Skip cross-split content duplicate detection")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    data = args.data.resolve()
    class_sets: dict[str, set[str]] = {}
    counts: dict[str, dict[str, int]] = {}
    invalid: list[dict[str, str]] = []
    content_locations: dict[str, list[str]] = defaultdict(list)

    for split in SPLITS:
        split_dir = data / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        classes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
        class_sets[split] = set(classes)
        counts[split] = {}
        for class_name in classes:
            images = sorted(
                path for path in (split_dir / class_name).iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            counts[split][class_name] = len(images)
            for path in images:
                try:
                    with Image.open(path) as image:
                        image.verify()
                except Exception as exc:  # report every unreadable image instead of stopping early
                    invalid.append({"file": str(path), "error": str(exc)})
                if not args.skip_hash:
                    content_locations[sha256(path)].append(str(path.relative_to(data)))

    if not (class_sets["train"] == class_sets["val"] == class_sets["test"]):
        raise RuntimeError(f"Class mismatch across splits: {class_sets}")

    classes = sorted(class_sets["train"])
    rows = []
    for class_name in classes:
        train = counts["train"][class_name]
        val = counts["val"][class_name]
        test = counts["test"][class_name]
        total = train + val + test
        rows.append({
            "class": class_name, "train": train, "val": val, "test": test, "total": total,
            "train_ratio": round(train / total, 4),
            "val_ratio": round(val / total, 4),
            "test_ratio": round(test / total, 4),
        })

    duplicates = [locations for locations in content_locations.values() if len(locations) > 1]
    cross_split_duplicates = [
        locations for locations in duplicates
        if len({location.split("\\")[0].split("/")[0] for location in locations}) > 1
    ]
    split_totals = {split: sum(counts[split].values()) for split in SPLITS}
    grand_total = sum(split_totals.values())
    report = {
        "dataset": str(data),
        "classes": classes,
        "class_count": len(classes),
        "split_totals": split_totals,
        "total_images": grand_total,
        "split_ratios": {split: round(split_totals[split] / grand_total, 4) for split in SPLITS},
        "invalid_images": invalid,
        "cross_split_duplicate_groups": cross_split_duplicates,
        "status": "ok" if not invalid and not cross_split_duplicates else "failed",
    }

    with (data / "dataset_counts.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (data / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
