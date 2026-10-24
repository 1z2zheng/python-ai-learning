"""Import *manually verified* field photos of apple black rot into training only.

This does not infer a disease label from a filename or a model prediction.
It rejects near-duplicates of every existing PlantVillage, PlantSeg, and extra
training/validation/test image before copying anything into the extra train set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps

try:
    from .train_mixed import difference_hash, plantseg_records, plantvillage_records, read_classes
except ImportError:
    from train_mixed import difference_hash, plantseg_records, plantvillage_records, read_classes


PROJECT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_NAME = "Apple_Black_rot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Directory containing only visually reviewed Apple black-rot field photos")
    parser.add_argument("--source-url", required=True,
                        help="Original collection URL or a description of self-collected photos")
    parser.add_argument("--license", required=True,
                        help="License/permission statement for this image collection")
    parser.add_argument("--confirmed-labels", action="store_true",
                        help="Confirm a human checked that every input image depicts Apple black rot")
    parser.add_argument("--plantvillage", type=Path, default=PROJECT / "dataset_classification")
    parser.add_argument("--plantseg", type=Path, default=PROJECT / "dataset_segmentation_plantseg")
    parser.add_argument("--extra", type=Path, default=PROJECT / "dataset_natural_extra")
    parser.add_argument("--max-hamming-distance", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def existing_images(plantvillage: Path, plantseg: Path, extra: Path) -> list[Path]:
    classes = read_classes(plantvillage)
    paths = []
    for split in ("train", "val", "test"):
        paths.extend(record.path for record in plantvillage_records(plantvillage, split, classes))
        paths.extend(record.path for record in plantseg_records(plantseg, split, classes))
    train_root = extra / "train"
    if train_root.exists():
        paths.extend(path for path in train_root.glob("*/*")
                     if path.is_file() and path.suffix.lower() in EXTENSIONS)
    return paths


def main() -> None:
    args = parse_args()
    if not args.confirmed_labels:
        raise SystemExit("Refusing automatic labeling. Review all images first, then pass --confirmed-labels.")
    if not 0 <= args.max_hamming_distance <= 8:
        raise ValueError("--max-hamming-distance must be in [0, 8]")
    source = args.source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    candidates = sorted(path for path in source.iterdir()
                        if path.is_file() and path.suffix.lower() in EXTENSIONS)
    if not candidates:
        raise RuntimeError(f"No images found directly inside {source}")

    existing = existing_images(args.plantvillage.resolve(), args.plantseg.resolve(), args.extra.resolve())
    print(f"Hashing {len(existing)} existing images...")
    known: list[tuple[int, str]] = []
    for index, path in enumerate(existing, 1):
        known.append((difference_hash(path), str(path)))
        if index % 2000 == 0:
            print(f"  {index}/{len(existing)}")

    destination_dir = args.extra.resolve() / "train" / CLASS_NAME
    rows = []
    accepted = duplicates = invalid = 0
    for path in candidates:
        row = {"source_file": str(path), "source_url": args.source_url,
               "license": args.license, "output_file": "", "sha256": "",
               "status": "", "matched_existing": "", "width": "", "height": ""}
        try:
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            with Image.open(path) as source_image:
                source_image.verify()
            with Image.open(path) as source_image:
                image = ImageOps.exif_transpose(source_image).convert("RGB")
            if min(image.size) < 128:
                raise ValueError(f"Image too small: {image.size}")
            fingerprint = difference_hash(path)
            match = next((name for known_hash, name in known
                          if (fingerprint ^ known_hash).bit_count() <= args.max_hamming_distance), None)
            row.update(sha256=digest, width=image.width, height=image.height)
            if match:
                row.update(status="near_duplicate", matched_existing=match)
                duplicates += 1
            else:
                filename = f"field_blackrot_{digest[:20]}.jpg"
                destination = destination_dir / filename
                row.update(status="accepted", output_file=str(destination))
                if not args.dry_run:
                    destination_dir.mkdir(parents=True, exist_ok=True)
                    if not destination.exists():
                        image.save(destination, format="JPEG", quality=94, optimize=True)
                known.append((fingerprint, str(destination)))
                accepted += 1
        except Exception as error:
            row.update(status="invalid", matched_existing=f"{type(error).__name__}: {error}")
            invalid += 1
        rows.append(row)

    report = {"source": str(source), "source_url": args.source_url, "license": args.license,
              "class": CLASS_NAME, "destination_split": "train", "dry_run": args.dry_run,
              "existing_images_checked": len(existing), "candidates": len(candidates),
              "accepted": accepted, "near_duplicates": duplicates, "invalid": invalid,
              "max_hamming_distance": args.max_hamming_distance}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    output = args.extra.resolve()
    manifest = output / "field_blackrot_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "field_blackrot_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
