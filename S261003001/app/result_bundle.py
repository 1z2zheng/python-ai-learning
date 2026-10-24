"""Build downloadable files for one completed prediction."""

from __future__ import annotations

import io
import json
import zipfile

from PIL import Image


def image_bytes(image: Image.Image, image_format: str) -> bytes:
    buffer = io.BytesIO()
    options = {"quality": 95} if image_format.upper() in {"JPEG", "JPG"} else {}
    image.save(buffer, format=image_format, **options)
    return buffer.getvalue()


def result_archive(image: Image.Image, result: dict[str, object]) -> bytes:
    """Build one ZIP containing the input, report, and available visual results."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("input.jpg", image_bytes(image, "JPEG"))
        archive.writestr(
            "result.json",
            json.dumps(result["report"], ensure_ascii=False, indent=2).encode("utf-8"),
        )
        if result["mask"] is not None:
            archive.writestr("lesion_mask.png", image_bytes(result["mask"], "PNG"))
            archive.writestr("lesion_overlay.jpg", image_bytes(result["overlay"], "JPEG"))
        if result["heatmap"] is not None:
            archive.writestr("classification_crop.jpg", image_bytes(result["classification_crop"], "JPEG"))
            archive.writestr("classification_gradcam.jpg", image_bytes(result["heatmap"], "JPEG"))
    return buffer.getvalue()
