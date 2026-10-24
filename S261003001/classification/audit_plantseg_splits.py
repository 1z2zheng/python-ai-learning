"""Find exact and visually similar images crossing PlantSeg splits.

Reports candidates without changing any image or split. dHash matches are
screening signals and must be checked visually before calling them duplicates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from classification.train_mixed import difference_hash  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "dataset_segmentation_plantseg")
    parser.add_argument("--output", type=Path, default=PROJECT / "runs/plantseg_split_audit")
    parser.add_argument("--max-distance", type=int, default=2)
    return parser.parse_args()


def build_records(data: Path):
    splits = json.loads((data / "splits.json").read_text(encoding="utf-8"))
    images = {path.stem: path for path in (data / "images").iterdir() if path.is_file()}
    records = []
    for split, stems in splits.items():
        for stem in stems:
            path = images[stem]
            records.append({
                "split": split,
                "class": stem.split("__", 1)[0],
                "path": path,
                "dhash": difference_hash(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    return records


def contact_sheet(pairs: list[dict], output: Path, limit: int = 30):
    if not pairs:
        return
    selected = pairs[:limit]
    width, height = 640, 190
    sheet = Image.new("RGB", (width, height * len(selected)), "white")
    draw = ImageDraw.Draw(sheet)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 13) if font_path.is_file() else ImageFont.load_default()
    for index, pair in enumerate(selected):
        top = index * height
        for column, side in enumerate(("a", "b")):
            with Image.open(pair[f"{side}_path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((300, 140))
            x = 5 + column * 320
            sheet.paste(image, (x + (300 - image.width) // 2, top + 2))
            draw.text((x, top + 143), pair[f"{side}_split"] + ": " + pair[f"{side}_class"], fill="#222222", font=font)
            draw.text((x, top + 161), Path(pair[f"{side}_path"]).stem[-35:], fill="#555555", font=font)
        draw.text((560, top + 143), "d=" + str(pair["distance"]), fill="#b52d2d", font=font)
    sheet.save(output)


def main():
    args = parse_args()
    data, output = args.data.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = build_records(data)
    pairs = []
    for index, a in enumerate(records):
        for b in records[index + 1:]:
            if a["split"] == b["split"]:
                continue
            distance = (a["dhash"] ^ b["dhash"]).bit_count()
            if distance > args.max_distance and a["sha256"] != b["sha256"]:
                continue
            pairs.append({
                "a_split": a["split"], "a_class": a["class"], "a_path": str(a["path"]),
                "b_split": b["split"], "b_class": b["class"], "b_path": str(b["path"]),
                "distance": distance, "exact_bytes": a["sha256"] == b["sha256"],
            })
    pairs.sort(key=lambda row: (not row["exact_bytes"], row["distance"], row["a_path"]))
    with (output / "cross_split_candidates.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(pairs[0]) if pairs else [
            "a_split", "a_class", "a_path", "b_split", "b_class", "b_path", "distance", "exact_bytes",
        ])
        writer.writeheader()
        writer.writerows(pairs)
    contact_sheet(pairs, output / "candidate_contact_sheet.jpg")
    summary = {
        "samples": dict(Counter(row["split"] for row in records)),
        "candidate_pairs": len(pairs),
        "exact_byte_pairs": sum(row["exact_bytes"] for row in pairs),
        "cross_class_candidates": sum(row["a_class"] != row["b_class"] for row in pairs),
        "candidates_by_class": dict(Counter(row["a_class"] for row in pairs)),
        "method": f"64-bit dHash Hamming distance <= {args.max_distance}; visually confirm candidates",
        "note": "No dataset files or splits were changed.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
