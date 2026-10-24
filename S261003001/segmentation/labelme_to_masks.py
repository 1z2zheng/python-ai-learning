"""Convert Labelme lesion annotations to binary masks (0 background, 255 lesion)."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labelme-dir", type=Path, default=Path("dataset_segmentation/labelme"))
    parser.add_argument("--output", type=Path, default=Path("dataset_segmentation"))
    parser.add_argument("--label", default="lesion", help="Positive Labelme shape label")
    return parser.parse_args()


def resolve_image(json_path: Path, data: dict) -> Path:
    image_path = Path(data.get("imagePath", ""))
    candidates = [json_path.parent / image_path, json_path.with_suffix(".jpg"), json_path.with_suffix(".jpeg"), json_path.with_suffix(".png")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find image referenced by {json_path}")


def draw_shape(draw: ImageDraw.ImageDraw, shape: dict) -> None:
    points = [tuple(map(float, point)) for point in shape.get("points", [])]
    shape_type = shape.get("shape_type", "polygon")
    if shape_type == "polygon" and len(points) >= 3:
        draw.polygon(points, fill=255)
    elif shape_type == "rectangle" and len(points) == 2:
        draw.rectangle([points[0], points[1]], fill=255)
    elif shape_type == "circle" and len(points) == 2:
        cx, cy = points[0]
        radius = math.dist(points[0], points[1])
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=255)
    else:
        raise ValueError(f"Unsupported or invalid shape type: {shape_type}")


def main() -> None:
    args = parse_args()
    labelme_dir = args.labelme_dir.resolve()
    output = args.output.resolve()
    images_dir = output / "images"
    masks_dir = output / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(labelme_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No Labelme JSON files found in {labelme_dir}")

    converted = 0
    for json_path in json_files:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        image_path = resolve_image(json_path, data)
        with Image.open(image_path) as image:
            width, height = image.size
        expected = (int(data.get("imageWidth", width)), int(data.get("imageHeight", height)))
        if expected != (width, height):
            raise ValueError(f"Image dimensions disagree with JSON: {json_path}")

        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        lesion_shapes = [shape for shape in data.get("shapes", []) if shape.get("label") == args.label]
        unknown_labels = sorted({shape.get("label") for shape in data.get("shapes", []) if shape.get("label") != args.label})
        if unknown_labels:
            raise ValueError(f"Unexpected labels in {json_path.name}: {unknown_labels}; expected only '{args.label}'")
        if not lesion_shapes:
            raise ValueError(f"No '{args.label}' shapes in {json_path.name}")
        for shape in lesion_shapes:
            draw_shape(draw, shape)

        dst_image = images_dir / image_path.name
        shutil.copy2(image_path, dst_image)
        mask.save(masks_dir / f"{image_path.stem}.png")
        converted += 1

    print(f"Converted {converted} annotations")
    print(f"Images: {images_dir}")
    print(f"Masks:  {masks_dir}")


if __name__ == "__main__":
    main()

