"""Validate and quarantine extra images that could leak into protected splits."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

try:
    from .train_mixed import (
        difference_hash,
        plantseg_records,
        plantvillage_records,
        read_classes,
    )
except ImportError:
    from train_mixed import (
        difference_hash,
        plantseg_records,
        plantvillage_records,
        read_classes,
    )


PROJECT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extra", type=Path, default=PROJECT / "dataset_natural_extra"
    )
    parser.add_argument(
        "--plantvillage", type=Path, default=PROJECT / "dataset_classification"
    )
    parser.add_argument(
        "--plantseg", type=Path, default=PROJECT / "dataset_segmentation_plantseg"
    )
    parser.add_argument("--max-hamming-distance", type=int, default=2)
    return parser.parse_args()


def unique_destination(root: Path, class_name: str, filename: str) -> Path:
    destination = root / class_name / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        return destination
    return destination.with_name(f"{destination.stem}_copy{destination.suffix}")


def main() -> None:
    args = parse_args()
    extra = args.extra.resolve()
    train_root = extra / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(train_root)
    classes = read_classes(args.plantvillage.resolve())
    protected_records = []
    for split in ("val", "test"):
        protected_records.extend(
            plantvillage_records(args.plantvillage.resolve(), split, classes)
        )
        protected_records.extend(
            plantseg_records(args.plantseg.resolve(), split, classes)
        )
    print(f"Hashing {len(protected_records)} protected validation/test images...")
    protected_hashes = [
        (difference_hash(record.path), str(record.path))
        for record in tqdm(protected_records, leave=False)
    ]

    candidates = sorted(
        path
        for path in train_root.glob("*/*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    accepted_hashes: list[tuple[int, str]] = []
    accepted = []
    quarantined = []
    invalid = []
    for path in tqdm(candidates, desc="Checking extra images"):
        class_name = path.parent.name
        try:
            with Image.open(path) as image:
                image.verify()
            fingerprint = difference_hash(path)
        except Exception as error:
            destination = unique_destination(
                extra / "rejected" / "invalid", class_name, path.name
            )
            shutil.move(str(path), destination)
            invalid.append(
                {
                    "file": str(path),
                    "moved_to": str(destination),
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            continue

        match = next(
            (
                source
                for known_hash, source in protected_hashes
                if (fingerprint ^ known_hash).bit_count()
                <= args.max_hamming_distance
            ),
            None,
        )
        duplicate_kind = "protected_split" if match else None
        if match is None:
            match = next(
                (
                    source
                    for known_hash, source in accepted_hashes
                    if (fingerprint ^ known_hash).bit_count()
                    <= args.max_hamming_distance
                ),
                None,
            )
            duplicate_kind = "within_extra" if match else None

        if match is not None:
            destination = unique_destination(
                extra / "rejected" / "near_duplicate", class_name, path.name
            )
            shutil.move(str(path), destination)
            quarantined.append(
                {
                    "file": str(path),
                    "moved_to": str(destination),
                    "duplicate_of": match,
                    "duplicate_kind": duplicate_kind,
                }
            )
        else:
            accepted_hashes.append((fingerprint, str(path)))
            accepted.append(str(path))

    counts = {}
    for directory in sorted(train_root.iterdir()):
        if directory.is_dir():
            counts[directory.name] = sum(
                path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                for path in directory.iterdir()
            )
    report = {
        "method": (
            "64-bit difference hash; quarantine when Hamming distance is at most "
            f"{args.max_hamming_distance}"
        ),
        "protected_images": len(protected_records),
        "candidates": len(candidates),
        "accepted": len(accepted),
        "quarantined_near_duplicates": len(quarantined),
        "invalid": len(invalid),
        "accepted_class_counts": counts,
        "quarantined": quarantined,
        "invalid_files": invalid,
    }
    report_path = extra / "dedup_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key not in {"quarantined", "invalid_files"}}, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
