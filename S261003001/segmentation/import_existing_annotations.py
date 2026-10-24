"""Import Labelme annotations accidentally saved in the classification dataset."""

from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "dataset_classification" / "train" / "Apple_Apple_scab"
DESTINATION = PROJECT / "dataset_segmentation" / "labelme"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    json_files = sorted(SOURCE.glob("*.json"))
    imported = 0
    skipped = 0
    existing_hashes = {
        sha256(path)
        for path in DESTINATION.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    next_index = 1

    for json_path in json_files:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(data.get("shapes"), list) or not data.get("imagePath"):
            continue
        labels = {shape.get("label") for shape in data["shapes"]}
        if labels != {"lesion"}:
            raise ValueError(f"Unexpected labels in {json_path}: {sorted(labels)}")

        source_image = json_path.parent / data["imagePath"]
        if not source_image.is_file():
            source_image = json_path.with_suffix(".JPG")
        if not source_image.is_file():
            source_image = json_path.with_suffix(".jpg")
        if not source_image.is_file():
            raise FileNotFoundError(f"Image not found for {json_path}")

        image_hash = sha256(source_image)
        if image_hash in existing_hashes:
            skipped += 1
            continue

        while (DESTINATION / f"Apple_Apple_scab_extra_{next_index:04d}.json").exists():
            next_index += 1

        stem = f"Apple_Apple_scab_extra_{next_index:04d}"
        target_image = DESTINATION / f"{stem}.jpg"
        target_json = DESTINATION / f"{stem}.json"

        shutil.copy2(source_image, target_image)
        data["imagePath"] = target_image.name
        target_json.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        imported += 1
        existing_hashes.add(image_hash)
        next_index += 1

    print(f"Imported {imported} annotations")
    print(f"Skipped {skipped} annotations already present")
    print(f"Destination: {DESTINATION}")


if __name__ == "__main__":
    main()
