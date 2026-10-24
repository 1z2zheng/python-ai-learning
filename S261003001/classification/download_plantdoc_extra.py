"""Download selected PlantDoc natural-scene classes for classification training."""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_TREE = PROJECT / "external" / "plantdoc_tree.json"
DEFAULT_OUTPUT = PROJECT / "dataset_natural_extra"
RAW_ROOT = "https://raw.githubusercontent.com/pratikkayal/PlantDoc-Dataset/master/"

CLASS_MAPPING = {
    "Apple Scab Leaf": "Apple_Apple_scab",
    "Apple rust leaf": "Apple_Cedar_apple_rust",
    "Potato leaf early blight": "Potato_Early_blight",
    "Potato leaf late blight": "Potato_Late_blight",
    "Tomato Early blight leaf": "Tomato_Early_blight",
    "Tomato leaf late blight": "Tomato_Late_blight",
    "Tomato mold leaf": "Tomato_Leaf_Mold",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


def selected_entries(tree: dict) -> list[tuple[dict, str]]:
    selected = []
    for entry in tree["tree"]:
        if entry.get("type") != "blob":
            continue
        parts = entry["path"].split("/", 2)
        if len(parts) != 3 or parts[0] != "train":
            continue
        if parts[1] in CLASS_MAPPING:
            selected.append((entry, CLASS_MAPPING[parts[1]]))
    return selected


def destination_for(output: Path, entry: dict, class_name: str) -> Path:
    return output / "train" / class_name / f"plantdoc_{entry['sha'][:16]}.jpg"


def valid_existing(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def download_one(
    output: Path,
    entry: dict,
    class_name: str,
    retries: int,
    timeout: int,
) -> dict[str, object]:
    destination = destination_for(output, entry, class_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_path = entry["path"]
    url = RAW_ROOT + urllib.parse.quote(source_path, safe="/")
    if valid_existing(destination):
        with Image.open(destination) as existing:
            width, height = existing.size
        return {
            "class_name": class_name,
            "output_file": str(destination.relative_to(output)),
            "source_path": source_path,
            "source_sha": entry["sha"],
            "source_url": url,
            "width": width,
            "height": height,
            "status": "existing",
            "error": "",
        }

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "PlantDiseaseGraduationProject/1.0"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            with Image.open(io.BytesIO(payload)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            if image.width < 128 or image.height < 128:
                raise ValueError(f"image too small: {image.size}")
            image.save(destination, format="JPEG", quality=94, optimize=True)
            return {
                "class_name": class_name,
                "output_file": str(destination.relative_to(output)),
                "source_path": source_path,
                "source_sha": entry["sha"],
                "source_url": url,
                "width": image.width,
                "height": image.height,
                "status": "downloaded",
                "error": "",
            }
        except Exception as error:  # network and image-decoding errors are recorded
            last_error = f"{type(error).__name__}: {error}"
            if destination.exists():
                destination.unlink()
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    return {
        "class_name": class_name,
        "output_file": str(destination.relative_to(output)),
        "source_path": source_path,
        "source_sha": entry["sha"],
        "source_url": url,
        "width": "",
        "height": "",
        "status": "failed",
        "error": last_error,
    }


def main() -> None:
    args = parse_args()
    tree_path = args.tree.resolve()
    output = args.output.resolve()
    if not tree_path.is_file():
        raise FileNotFoundError(tree_path)
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    entries = selected_entries(tree)
    if not entries:
        raise RuntimeError("No mapped PlantDoc training images found in repository tree")
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                download_one,
                output,
                entry,
                class_name,
                args.retries,
                args.timeout,
            ): (entry, class_name)
            for entry, class_name in entries
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"[{index:03d}/{len(entries)}] {row['status']}: "
                f"{row['class_name']} / {Path(str(row['source_path'])).name}"
            )

    rows.sort(key=lambda row: (str(row["class_name"]), str(row["source_path"])))
    manifest = output / "plantdoc_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, dict[str, int]] = {}
    for class_name in sorted(set(CLASS_MAPPING.values())):
        class_rows = [row for row in rows if row["class_name"] == class_name]
        counts[class_name] = {
            "available": len(class_rows),
            "valid": sum(row["status"] != "failed" for row in class_rows),
            "failed": sum(row["status"] == "failed" for row in class_rows),
        }
    report = {
        "source": "PlantDoc-Dataset",
        "repository": "https://github.com/pratikkayal/PlantDoc-Dataset",
        "license": "CC BY 4.0",
        "source_split": "train only",
        "class_mapping": CLASS_MAPPING,
        "counts": counts,
        "total_valid": sum(row["status"] != "failed" for row in rows),
        "total_failed": sum(row["status"] == "failed" for row in rows),
    }
    (output / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "SOURCE.md").write_text(
        "# 数据来源\n\n"
        "本目录图片来自PlantDoc-Dataset的train划分，仅用于模型训练。\n\n"
        "- 仓库：https://github.com/pratikkayal/PlantDoc-Dataset\n"
        "- 论文：PlantDoc: A Dataset for Visual Plant Disease Detection\n"
        "- 许可：Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
        "- 类别映射与原始文件URL：见 `plantdoc_manifest.csv`\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
