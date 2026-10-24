from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    pred_count = int(prediction.sum())
    target_count = int(target.sum())
    return {
        "iou": intersection / union if union else 1.0,
        "dice": (2 * intersection) / (pred_count + target_count)
        if pred_count + target_count
        else 1.0,
        "precision": intersection / pred_count if pred_count else 0.0,
        "recall": intersection / target_count if target_count else 0.0,
    }

def overlay_comparison(image: Image.Image, manual: np.ndarray, automatic: np.ndarray) -> Image.Image:
    array = np.asarray(image, dtype=np.float32).copy()
    only_manual = (manual == 1) & (automatic == 0)
    only_auto = (manual == 0) & (automatic == 1)
    agreement = (manual == 1) & (automatic == 1)
    array[only_manual] = array[only_manual] * 0.45 + np.array([255, 30, 30]) * 0.55
    array[only_auto] = array[only_auto] * 0.45 + np.array([30, 100, 255]) * 0.55
    array[agreement] = array[agreement] * 0.40 + np.array([30, 220, 80]) * 0.60
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

def contact_sheet(items: list[tuple[str, Image.Image]], output: Path) -> None:
    tile = 400
    label_height = 28
    margin = 14
    columns = 2
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * tile + (columns + 1) * margin, rows * (tile + label_height) + (rows + 1) * margin),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(items):
        row, column = divmod(index, columns)
        x = margin + column * (tile + margin)
        y = margin + row * (tile + label_height + margin)
        preview = image.copy()
        preview.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        draw.text((x, y + 4), label, fill="black")
        sheet.paste(preview, (x + (tile - preview.width) // 2, y + label_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
