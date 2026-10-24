"""Check image/mask pairing, dimensions, and binary mask values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def flattened_pixels(image: Image.Image):
    """Return pixels without Pillow 14's deprecated getdata() warning."""
    if hasattr(image, "get_flattened_data"):
        return image.get_flattened_data()
    return image.getdata()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset_segmentation"))
    args = parser.parse_args()
    images_dir = args.data.resolve() / "images"
    masks_dir = args.data.resolve() / "masks"
    images = {p.stem: p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
    masks = {p.stem: p for p in masks_dir.glob("*.png")}

    errors: list[str] = []
    stats: list[dict[str, object]] = []
    for stem in sorted(set(images) | set(masks)):
        if stem not in images:
            errors.append(f"Missing image for mask: {masks[stem].name}")
            continue
        if stem not in masks:
            errors.append(f"Missing mask for image: {images[stem].name}")
            continue
        with Image.open(images[stem]) as image, Image.open(masks[stem]) as mask:
            if image.size != mask.size:
                errors.append(f"Size mismatch for {stem}: image={image.size}, mask={mask.size}")
            grayscale_mask = mask.convert("L")
            pixels = list(flattened_pixels(grayscale_mask))
            values = set(pixels)
            if not values.issubset({0, 255}):
                errors.append(f"Non-binary mask values for {stem}: {sorted(values)[:20]}")
            lesion_pixels = sum(value == 255 for value in pixels)
            ratio = lesion_pixels / (mask.width * mask.height)
            if lesion_pixels == 0:
                errors.append(f"Empty lesion mask: {stem}")
            stats.append({"name": stem, "width": mask.width, "height": mask.height, "lesion_ratio": round(ratio, 6)})

    report = {
        "images": len(images), "masks": len(masks), "valid_pairs": len(stats),
        "errors": errors, "status": "ok" if not errors else "failed", "mask_stats": stats,
    }
    report_path = args.data.resolve() / "mask_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Images: {len(images)}, masks: {len(masks)}, valid pairs: {len(stats)}")
    print(f"Status: {report['status']}")
    print(f"Report: {report_path}")
    if errors:
        print("\n".join(errors[:50]))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

