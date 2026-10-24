"""Run a reproducible end-to-end acceptance test on held-out PlantSeg images."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from inference.predict import (  # noqa: E402
    DEFAULT_CLASSIFIER,
    DEFAULT_SEGMENTER,
    classify,
    load_classifier,
    load_segmenter,
    segment,
    select_device,
)


DEFAULT_DATA = PROJECT / "dataset_segmentation_plantseg"
DEFAULT_OUTPUT = PROJECT / "runs" / "system_acceptance"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--segmenter", type=Path, default=DEFAULT_SEGMENTER)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    parser.add_argument("--segmentation-threshold", type=float, default=0.65)
    parser.add_argument("--segmentation-size", type=int, default=512)
    parser.add_argument("--min-component-percent", type=float, default=0.03)
    return parser.parse_args()


def find_file(folder: Path, stem: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        candidate = folder / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No image found for {stem} in {folder}")


def true_class(stem: str) -> str:
    return stem.split("__", 1)[0]


def balanced_sample(stems: list[str], count: int, seed: int) -> list[str]:
    if count <= 0:
        raise ValueError("--samples must be positive")
    if count > len(stems):
        raise ValueError(f"Requested {count} samples, but split only has {len(stems)}")
    groups: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        groups[true_class(stem)].append(stem)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)

    selected: list[str] = []
    class_names = sorted(groups)
    while len(selected) < count:
        progressed = False
        for name in class_names:
            if groups[name] and len(selected) < count:
                selected.append(groups[name].pop())
                progressed = True
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


def binary_metrics(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    predicted = predicted.astype(bool)
    target = target.astype(bool)
    intersection = int(np.logical_and(predicted, target).sum())
    pred_sum = int(predicted.sum())
    target_sum = int(target.sum())
    union = int(np.logical_or(predicted, target).sum())
    dice = (2 * intersection + 1e-7) / (pred_sum + target_sum + 1e-7)
    iou = (intersection + 1e-7) / (union + 1e-7)
    return float(dice), float(iou)


def save_confusion_matrix(rows: list[dict[str, object]], output: Path) -> None:
    true_classes = sorted({str(row["true_class"]) for row in rows})
    predicted_classes = sorted(
        {str(row["predicted_class"]) for row in rows} | set(true_classes)
    )
    true_index = {name: position for position, name in enumerate(true_classes)}
    predicted_index = {
        name: position for position, name in enumerate(predicted_classes)
    }
    matrix = np.zeros((len(true_classes), len(predicted_classes)), dtype=int)
    for row in rows:
        actual = str(row["true_class"])
        predicted = str(row["predicted_class"])
        matrix[true_index[actual], predicted_index[predicted]] += 1
    totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals > 0)
    true_labels = [name.replace("Tomato_", "T_").replace("Potato_", "P_").replace("Apple_", "A_") for name in true_classes]
    predicted_labels = [name.replace("Tomato_", "T_").replace("Potato_", "P_").replace("Apple_", "A_") for name in predicted_classes]

    fig, ax = plt.subplots(figsize=(13, 9))
    image = ax.imshow(normalized, cmap="Greens", vmin=0, vmax=1)
    for row_index in range(len(true_classes)):
        for column_index in range(len(predicted_classes)):
            value = normalized[row_index, column_index]
            color = "white" if value >= 0.55 else "#173b31"
            ax.text(column_index, row_index, f"{value:.2f}\n({matrix[row_index, column_index]})", ha="center", va="center", color=color, fontsize=9)
    ax.set_xticks(range(len(predicted_classes)), predicted_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(true_classes)), true_labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("System acceptance test: normalized confusion matrix")
    fig.colorbar(image, ax=ax, label="Recall proportion")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_examples(rows: list[dict[str, object]], output: Path) -> None:
    failures = [row for row in rows if not bool(row["classification_correct"])]
    successes = [row for row in rows if bool(row["classification_correct"])]
    chosen = (failures[:4] + successes[:4])[:8]
    if not chosen:
        return
    fig, axes = plt.subplots(2, 4, figsize=(16, 11))
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, chosen):
        with Image.open(str(row["overlay_path"])) as source:
            ax.imshow(source.convert("RGB"))
        marker = "OK" if row["classification_correct"] else "ERROR"
        ax.set_title(
            f"{marker} | conf={float(row['confidence']):.1%}\n"
            f"GT: {row['true_class']}\nPred: {row['predicted_class']}\n"
            f"IoU={float(row['iou']):.3f}",
            fontsize=8,
            pad=5,
        )
        ax.axis("off")
    fig.suptitle("System acceptance examples (red = predicted lesion)", fontsize=15, y=0.97)
    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.04, top=0.90, wspace=0.08, hspace=0.42)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: dict[str, object], output: Path) -> None:
    class_rows = "\n".join(
        f"| {name} | {values['samples']} | {values['accuracy']:.2%} | {values['mean_confidence']:.2%} | {values['mean_iou']:.4f} |"
        for name, values in summary["per_class"].items()
    )
    text = f"""# 系统验收测试报告

## 1. 测试设置

- 数据来源：PlantSeg独立留出{summary['split']}集
- 抽样方式：按病害类别均衡抽样，随机种子为{summary['seed']}
- 测试图片：{summary['samples']}张
- 分类模型：MobileNetV3-Large混合微调模型
- 分割模型：DeepLabV3+ + MobileNetV3
- 分割阈值：{summary['segmentation_threshold']:.2f}
- 分割输入尺寸：{summary['segmentation_size']}×{summary['segmentation_size']}

## 2. 总体结果

| 指标 | 结果 |
|---|---:|
| 分类准确率 | {summary['classification_accuracy']:.2%} |
| 平均分类置信度 | {summary['mean_confidence']:.2%} |
| 低置信度样本数 | {summary['uncertain_samples']} |
| 平均Dice | {summary['mean_dice']:.4f} |
| 平均IoU | {summary['mean_iou']:.4f} |
| 平均单张推理耗时 | {summary['mean_elapsed_seconds']:.3f}秒 |

## 3. 各类别结果

| 类别 | 样本数 | 准确率 | 平均置信度 | 平均IoU |
|---|---:|---:|---:|---:|
{class_rows}

## 4. 验收判定

本项目将分类准确率不低于70%、平均IoU不低于0.50、GPU平均单张推理耗时不超过1秒作为系统验收参考线。

| 验收项目 | 参考线 | 实测值 | 结论 |
|---|---:|---:|---:|
| 分类准确率 | ≥70% | {summary['classification_accuracy']:.2%} | {'通过' if summary['criteria']['classification_pass'] else '未通过'} |
| 平均IoU | ≥0.50 | {summary['mean_iou']:.4f} | {'通过' if summary['criteria']['segmentation_pass'] else '未通过'} |
| GPU单张推理耗时 | ≤1秒 | {summary['mean_elapsed_seconds']:.3f}秒 | {'通过' if summary['criteria']['speed_pass'] else '未通过'} |

综合结论：**{summary['acceptance_result']}**。

## 5. 结果说明

本次测试使用未参与训练的PlantSeg留出测试集，覆盖苹果、马铃薯和番茄三种作物的8种病害。详细逐图结果见 `system_acceptance_results.csv`，分类错误和正确案例见 `system_acceptance_examples.png`。
"""
    output.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = args.data.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = json.loads((data_dir / "splits.json").read_text(encoding="utf-8"))
    selected = balanced_sample(list(splits[args.split]), args.samples, args.seed)
    device = select_device(args.device)

    classifier, classes, classifier_transform, _ = load_classifier(args.classifier.resolve(), device)
    segmenter, segmentation_checkpoint = load_segmenter(args.segmenter.resolve(), device)
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(exist_ok=True)

    rows: list[dict[str, object]] = []
    for number, stem in enumerate(selected, start=1):
        image_path = find_file(data_dir / "images", stem)
        mask_path = find_file(data_dir / "masks", stem)
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        with Image.open(mask_path) as source:
            target_mask = source.convert("L").resize(image.size, Image.Resampling.NEAREST)

        started = time.perf_counter()
        predictions = classify(image, classifier, classes, classifier_transform, device, top_k=3)
        predicted_mask, overlay, lesion_ratio = segment(
            image,
            segmenter,
            segmentation_checkpoint,
            device,
            threshold_override=args.segmentation_threshold,
            image_size_override=args.segmentation_size,
            min_component_percent=args.min_component_percent,
        )
        elapsed = time.perf_counter() - started
        predicted_class = str(predictions[0]["class"])
        confidence = float(predictions[0]["confidence"])
        expected_class = true_class(stem)
        dice, iou = binary_metrics(np.asarray(predicted_mask) > 0, np.asarray(target_mask) > 0)
        overlay_path = overlays_dir / f"{number:03d}_{stem}.jpg"
        overlay.save(overlay_path, quality=92)
        rows.append(
            {
                "number": number,
                "image_file": image_path.name,
                "crop": expected_class.split("_", 1)[0],
                "true_class": expected_class,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "classification_correct": predicted_class == expected_class,
                "uncertain": confidence < args.confidence_threshold,
                "lesion_ratio": lesion_ratio,
                "dice": dice,
                "iou": iou,
                "elapsed_seconds": elapsed,
                "background_type": "复杂",
                "subjective_segmentation_score_1_to_5": "",
                "notes": "",
                "overlay_path": str(overlay_path),
            }
        )
        print(f"[{number:02d}/{len(selected)}] {expected_class} -> {predicted_class} | conf={confidence:.1%} IoU={iou:.3f}")

    csv_path = output_dir / "system_acceptance_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_class: dict[str, dict[str, float | int]] = {}
    for name in sorted({str(row["true_class"]) for row in rows}):
        values = [row for row in rows if row["true_class"] == name]
        per_class[name] = {
            "samples": len(values),
            "accuracy": float(np.mean([row["classification_correct"] for row in values])),
            "mean_confidence": float(np.mean([row["confidence"] for row in values])),
            "mean_dice": float(np.mean([row["dice"] for row in values])),
            "mean_iou": float(np.mean([row["iou"] for row in values])),
        }
    summary = {
        "split": args.split,
        "seed": args.seed,
        "samples": len(rows),
        "sample_distribution": dict(Counter(str(row["true_class"]) for row in rows)),
        "classification_accuracy": float(np.mean([row["classification_correct"] for row in rows])),
        "mean_confidence": float(np.mean([row["confidence"] for row in rows])),
        "uncertain_samples": int(sum(bool(row["uncertain"]) for row in rows)),
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
        "mean_iou": float(np.mean([row["iou"] for row in rows])),
        "mean_elapsed_seconds": float(np.mean([row["elapsed_seconds"] for row in rows])),
        "segmentation_threshold": args.segmentation_threshold,
        "segmentation_size": args.segmentation_size,
        "min_component_percent": args.min_component_percent,
        "device": str(device),
        "per_class": per_class,
    }
    summary["criteria"] = {
        "classification_accuracy_min": 0.70,
        "mean_iou_min": 0.50,
        "mean_elapsed_seconds_max": 1.0,
        "classification_pass": summary["classification_accuracy"] >= 0.70,
        "segmentation_pass": summary["mean_iou"] >= 0.50,
        "speed_pass": summary["mean_elapsed_seconds"] <= 1.0,
    }
    summary["acceptance_result"] = (
        "通过" if all(
            (
                summary["criteria"]["classification_pass"],
                summary["criteria"]["segmentation_pass"],
                summary["criteria"]["speed_pass"],
            )
        ) else "未通过"
    )
    (output_dir / "system_acceptance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_confusion_matrix(rows, output_dir / "system_acceptance_confusion_matrix.png")
    save_examples(rows, output_dir / "system_acceptance_examples.png")
    write_report(summary, output_dir / "system_acceptance_report.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()
