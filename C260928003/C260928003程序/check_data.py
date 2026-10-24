from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def inspect_archives(root: Path) -> list[dict]:
    result = []
    for path in sorted(root.rglob("*.zip")):
        with ZipFile(path) as archive:
            files = [i for i in archive.infolist() if not i.is_dir()]
            suffixes = Counter(Path(i.filename).suffix.lower() or "<无扩展名>" for i in files)
            result.append(
                {
                    "path": str(path),
                    "file_count": len(files),
                    "uncompressed_bytes": sum(i.file_size for i in files),
                    "suffix_counts": dict(sorted(suffixes.items())),
                    "sample_names": [i.filename for i in files[:10]],
                }
            )
    return result


def inspect_gwhd(root: Path) -> dict | None:
    csv_path = next(iter(sorted(root.rglob("train.csv"))), None)
    if csv_path is None:
        return None
    required = {"image_id", "width", "height", "bbox", "source"}
    row_count = 0
    image_ids: set[str] = set()
    source_counts: Counter[str] = Counter()
    invalid_bbox = 0
    out_of_bounds_bbox = 0
    widths: Counter[int] = Counter()
    heights: Counter[int] = Counter()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            return {"path": str(csv_path), "error": f"缺少字段：{missing}"}
        for row in reader:
            row_count += 1
            image_ids.add(row["image_id"])
            source_counts[row["source"]] += 1
            try:
                width = int(float(row["width"]))
                height = int(float(row["height"]))
                x, y, box_width, box_height = [float(v) for v in ast.literal_eval(row["bbox"])]
                widths[width] += 1
                heights[height] += 1
                if box_width <= 0 or box_height <= 0 or x < 0 or y < 0:
                    invalid_bbox += 1
                if x + box_width > width + 1e-6 or y + box_height > height + 1e-6:
                    out_of_bounds_bbox += 1
            except (ValueError, TypeError, SyntaxError):
                invalid_bbox += 1
    return {
        "path": str(csv_path),
        "task_type": "object_detection",
        "annotation_format": "[x, y, width, height] bounding boxes",
        "annotation_rows": row_count,
        "unique_annotated_images": len(image_ids),
        "image_width_counts": dict(widths),
        "image_height_counts": dict(heights),
        "source_counts": dict(source_counts),
        "invalid_bbox_rows": invalid_bbox,
        "out_of_bounds_bbox_rows": out_of_bounds_bbox,
    }


def inspect_segmentation(root: Path, num_classes: int) -> dict | None:
    images_dir = root / "images"
    masks_dir = root / "masks"
    if not images_dir.exists() and not masks_dir.exists():
        return None
    images = {p.stem: p for p in image_files(images_dir)}
    masks = {p.stem: p for p in image_files(masks_dir)}
    paired = sorted(images.keys() & masks.keys())
    report = {
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "image_count": len(images),
        "mask_count": len(masks),
        "paired_count": len(paired),
        "images_without_masks": sorted(images.keys() - masks.keys())[:100],
        "masks_without_images": sorted(masks.keys() - images.keys())[:100],
        "size_mismatches": [],
        "invalid_masks": [],
        "class_pixel_counts": {str(i): 0 for i in range(num_classes)},
    }
    for stem in paired:
        try:
            with Image.open(images[stem]) as image, Image.open(masks[stem]) as mask:
                if image.size != mask.size:
                    report["size_mismatches"].append(
                        {"stem": stem, "image_size": image.size, "mask_size": mask.size}
                    )
                mask_array = np.asarray(mask.convert("L"), dtype=np.int64)
                values, counts = np.unique(mask_array, return_counts=True)
                invalid = [int(v) for v in values if v < 0 or v >= num_classes]
                if invalid:
                    report["invalid_masks"].append({"stem": stem, "invalid_values": invalid})
                for value, count in zip(values, counts):
                    if 0 <= value < num_classes:
                        report["class_pixel_counts"][str(int(value))] += int(count)
        except (OSError, ValueError) as exc:
            report["invalid_masks"].append({"stem": stem, "error": str(exc)})
    splits = {}
    for split in ("train", "val", "test"):
        path = root / "splits" / f"{split}.txt"
        if path.exists():
            items = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            splits[split] = {"count": len(items), "missing_pairs": sorted(set(items) - set(paired))[:100]}
        else:
            splits[split] = {"error": f"缺少 {path.name}"}
    report["splits"] = splits
    split_sets = {
        name: set(
            line.strip()
            for line in (root / "splits" / f"{name}.txt").read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
        for name in ("train", "val", "test")
        if (root / "splits" / f"{name}.txt").exists()
    }
    report["split_overlaps"] = {
        "train_val": sorted(split_sets.get("train", set()) & split_sets.get("val", set()))[:100],
        "train_test": sorted(split_sets.get("train", set()) & split_sets.get("test", set()))[:100],
        "val_test": sorted(split_sets.get("val", set()) & split_sets.get("test", set()))[:100],
    }
    return report


def build_summary(gwhd: dict | None, segmentation: dict | None) -> tuple[list[str], list[str]]:
    findings: list[str] = []
    warnings: list[str] = []
    if gwhd:
        findings.append(
            f"检测到目标检测数据：{gwhd.get('annotation_rows', 0)}条边界框，"
            f"{gwhd.get('unique_annotated_images', 0)}个有标注图像。"
        )
        warnings.append("GWHD边界框不能直接作为U-Net语义分割Mask。")
        if gwhd.get("invalid_bbox_rows") or gwhd.get("out_of_bounds_bbox_rows"):
            warnings.append("边界框中存在无效或越界记录，请查看JSON报告。")
    if segmentation:
        findings.append(f"检测到语义分割数据：{segmentation['paired_count']}对图像和Mask。")
        if segmentation["images_without_masks"] or segmentation["masks_without_images"]:
            warnings.append("存在未配对的图像或Mask。")
        if segmentation["size_mismatches"]:
            warnings.append("存在图像与Mask尺寸不一致。")
        if segmentation["invalid_masks"]:
            warnings.append("存在无法读取或类别值越界的Mask。")
        if any(segmentation["split_overlaps"].values()):
            warnings.append("训练、验证或测试划分存在重叠。")
    if not gwhd and not segmentation:
        warnings.append("未检测到GWHD train.csv或images/masks语义分割目录。")
    return findings, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="检查目标检测和语义分割数据")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.data_root.exists():
        raise SystemExit(f"数据目录不存在：{args.data_root}")
    gwhd = inspect_gwhd(args.data_root)
    segmentation = inspect_segmentation(args.data_root, args.num_classes)
    findings, warnings = build_summary(gwhd, segmentation)
    report = {
        "data_root": str(args.data_root.resolve()),
        "gwhd": gwhd,
        "segmentation": segmentation,
        "archives": inspect_archives(args.data_root),
        "findings": findings,
        "warnings": warnings,
    }
    print("数据检查结果")
    for item in findings:
        print(f"[信息] {item}")
    for item in warnings:
        print(f"[警告] {item}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已保存：{args.report}")


if __name__ == "__main__":
    main()

