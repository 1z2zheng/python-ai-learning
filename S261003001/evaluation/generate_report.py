"""Generate thesis-ready experiment figures, tables and acceptance templates."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "runs" / "final_report"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def save_classification_comparison(old: dict, mixed: dict, refined: dict) -> None:
    names = ["PlantVillage accuracy", "PlantSeg accuracy", "PlantSeg confidence"]
    old_values = [
        old["plantvillage"]["accuracy"],
        old["plantseg"]["accuracy"],
        old["plantseg"]["mean_confidence"],
    ]
    mixed_values = [
        mixed["plantvillage_test"]["accuracy"],
        mixed["plantseg_test"]["accuracy"],
        mixed["plantseg_test"]["mean_confidence"],
    ]
    refined_values = [
        refined["plantvillage_test"]["accuracy"],
        refined["plantseg_test"]["accuracy"],
        refined["plantseg_test"]["mean_confidence"],
    ]
    x = np.arange(len(names))
    width = 0.25
    figure, axis = plt.subplots(figsize=(10, 5.4))
    old_bars = axis.bar(x - width, old_values, width, label="MobileNetV3-Small", color="#9db8ad")
    mixed_bars = axis.bar(x, mixed_values, width, label="Large: PV + PlantSeg", color="#4f9275")
    refined_bars = axis.bar(x + width, refined_values, width, label="Large: PV + PlantSeg + PlantDoc", color="#145a3d")
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Score")
    axis.set_xticks(x, names)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right")
    for bars in (old_bars, mixed_bars, refined_bars):
        axis.bar_label(bars, labels=[f"{value:.1%}" for value in bars.datavalues], padding=3)
    figure.tight_layout()
    figure.savefig(OUTPUT / "classification_model_comparison.png", dpi=200)
    plt.close(figure)


def supported_class_rows(test_metrics: dict, classes: list[str]):
    report = test_metrics["classification_report"]
    return [
        (name, report[name])
        for name in classes
        if name in report and int(report[name]["support"]) > 0
    ]


def save_plantseg_f1(new: dict) -> None:
    rows = supported_class_rows(new["plantseg_test"], new["classes"])
    names = [name.replace("_", "\n") for name, _ in rows]
    scores = [float(values["f1-score"]) for _, values in rows]
    order = np.argsort(scores)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    bars = axis.barh(np.asarray(names)[order], np.asarray(scores)[order], color="#3a8064")
    axis.set_xlim(0, 1.02)
    axis.set_xlabel("F1-score")
    axis.set_title("PlantSeg test performance by disease class")
    axis.grid(axis="x", alpha=0.25)
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in np.asarray(scores)[order]], padding=3)
    figure.tight_layout()
    figure.savefig(OUTPUT / "plantseg_class_f1.png", dpi=200)
    plt.close(figure)


def save_confusion_matrix(new: dict) -> None:
    classes = new["classes"]
    test_metrics = new["plantseg_test"]
    supported = [
        index
        for index, name in enumerate(classes)
        if name in test_metrics["classification_report"]
        and int(test_metrics["classification_report"][name]["support"]) > 0
    ]
    matrix = np.asarray(test_metrics["confusion_matrix"], dtype=float)
    matrix = matrix[np.ix_(supported, supported)]
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
    labels = [classes[index].replace("Apple_", "A_").replace("Potato_", "P_").replace("Tomato_", "T_") for index in supported]
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(normalized, cmap="Greens", vmin=0, vmax=1)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Recall proportion")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("PlantSeg normalized confusion matrix")
    for row in range(normalized.shape[0]):
        for column in range(normalized.shape[1]):
            value = normalized[row, column]
            color = "white" if value > 0.55 else "#17352b"
            axis.text(column, row, f"{value:.2f}\n({int(matrix[row, column])})", ha="center", va="center", color=color, fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT / "plantseg_confusion_matrix.png", dpi=200)
    plt.close(figure)


def save_segmentation_comparison(unet: dict, deeplab: dict) -> None:
    names = ["Dice", "IoU", "Pixel accuracy"]
    unet_values = [unet["test"]["dice"], unet["test"]["iou"], unet["test"]["pixel_accuracy"]]
    deeplab_values = [deeplab["test"]["dice"], deeplab["test"]["iou"], deeplab["test"]["pixel_accuracy"]]
    x = np.arange(len(names))
    width = 0.34
    figure, axis = plt.subplots(figsize=(8.5, 5))
    first = axis.bar(x - width / 2, unet_values, width, label="U-Net", color="#93b7be")
    second = axis.bar(x + width / 2, deeplab_values, width, label="DeepLabV3+ MobileNetV3", color="#205b73")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_xticks(x, names)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right")
    for bars in (first, second):
        axis.bar_label(bars, labels=[f"{value:.4f}" for value in bars.datavalues], padding=3)
    figure.tight_layout()
    figure.savefig(OUTPUT / "segmentation_model_comparison.png", dpi=200)
    plt.close(figure)


def write_summary_csv(old: dict, mixed: dict, refined: dict, unet: dict, deeplab: dict) -> None:
    rows = [
        ["Classification", "MobileNetV3-Small", "PlantVillage accuracy", old["plantvillage"]["accuracy"]],
        ["Classification", "MobileNetV3-Small", "PlantSeg accuracy", old["plantseg"]["accuracy"]],
        ["Classification", "MobileNetV3-Small", "PlantSeg mean confidence", old["plantseg"]["mean_confidence"]],
        ["Classification", "MobileNetV3-Large PV+PlantSeg", "PlantVillage accuracy", mixed["plantvillage_test"]["accuracy"]],
        ["Classification", "MobileNetV3-Large PV+PlantSeg", "PlantSeg accuracy", mixed["plantseg_test"]["accuracy"]],
        ["Classification", "MobileNetV3-Large PV+PlantSeg", "PlantSeg mean confidence", mixed["plantseg_test"]["mean_confidence"]],
        ["Classification", "MobileNetV3-Large PV+PlantSeg+PlantDoc", "PlantVillage accuracy", refined["plantvillage_test"]["accuracy"]],
        ["Classification", "MobileNetV3-Large PV+PlantSeg+PlantDoc", "PlantSeg accuracy", refined["plantseg_test"]["accuracy"]],
        ["Classification", "MobileNetV3-Large PV+PlantSeg+PlantDoc", "PlantSeg mean confidence", refined["plantseg_test"]["mean_confidence"]],
        ["Segmentation", "U-Net", "Dice", unet["test"]["dice"]],
        ["Segmentation", "U-Net", "IoU", unet["test"]["iou"]],
        ["Segmentation", "DeepLabV3+ MobileNetV3", "Dice", deeplab["test"]["dice"]],
        ["Segmentation", "DeepLabV3+ MobileNetV3", "IoU", deeplab["test"]["iou"]],
    ]
    with (OUTPUT / "experiment_summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["Task", "Model", "Metric", "Value"])
        writer.writerows(rows)


def write_acceptance_template() -> None:
    headers = [
        "编号",
        "图片文件名",
        "作物",
        "人工真实类别",
        "系统预测类别",
        "分类置信度",
        "分类是否正确",
        "病斑比例",
        "分割主观评分(1-5)",
        "背景类型",
        "备注",
    ]
    with (OUTPUT / "system_acceptance_template.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for index in range(1, 61):
            writer.writerow([index, "", "", "", "", "", "", "", "", "简单/复杂", ""])


def write_markdown(old: dict, mixed: dict, refined: dict, unet: dict, deeplab: dict) -> None:
    classification_gain = refined["plantseg_test"]["accuracy"] - old["plantseg"]["accuracy"]
    plantdoc_gain = refined["plantseg_test"]["accuracy"] - mixed["plantseg_test"]["accuracy"]
    confidence_gain = refined["plantseg_test"]["mean_confidence"] - old["plantseg"]["mean_confidence"]
    dice_gain = deeplab["test"]["dice"] - unet["test"]["dice"]
    iou_gain = deeplab["test"]["iou"] - unet["test"]["iou"]
    rows = supported_class_rows(refined["plantseg_test"], refined["classes"])
    class_table = "\n".join(
        f"| {name} | {values['precision']:.4f} | {values['recall']:.4f} | {values['f1-score']:.4f} | {int(values['support'])} |"
        for name, values in rows
    )
    markdown = f"""# 最终实验结果汇总

## 1. 分类模型对比

| 模型 | PlantVillage准确率 | PlantSeg准确率 | PlantSeg平均置信度 | PlantSeg ECE |
|---|---:|---:|---:|---:|
| MobileNetV3-Small | {percent(old['plantvillage']['accuracy'])} | {percent(old['plantseg']['accuracy'])} | {percent(old['plantseg']['mean_confidence'])} | {old['plantseg']['ece']:.4f} |
| MobileNetV3-Large（PlantVillage+PlantSeg） | {percent(mixed['plantvillage_test']['accuracy'])} | {percent(mixed['plantseg_test']['accuracy'])} | {percent(mixed['plantseg_test']['mean_confidence'])} | {mixed['plantseg_test']['ece']:.4f} |
| MobileNetV3-Large（加入PlantDoc） | {percent(refined['plantvillage_test']['accuracy'])} | {percent(refined['plantseg_test']['accuracy'])} | {percent(refined['plantseg_test']['mean_confidence'])} | {refined['plantseg_test']['ece']:.4f} |

最终模型相较初始基线使PlantSeg野外测试准确率提高了{classification_gain * 100:.2f}个百分点，平均置信度提高了{confidence_gain * 100:.2f}个百分点。加入PlantDoc自然场景样本后，相较两数据集混合模型又提高{plantdoc_gain * 100:.2f}个百分点；PlantVillage准确率仍保持在{percent(refined['plantvillage_test']['accuracy'])}。

## 2. PlantSeg各类别结果

| 类别 | Precision | Recall | F1 | 样本数 |
|---|---:|---:|---:|---:|
{class_table}

## 3. 分割模型对比

| 模型 | Dice | IoU | 像素准确率 |
|---|---:|---:|---:|
| U-Net | {unet['test']['dice']:.4f} | {unet['test']['iou']:.4f} | {unet['test']['pixel_accuracy']:.4f} |
| DeepLabV3+ + MobileNetV3 | {deeplab['test']['dice']:.4f} | {deeplab['test']['iou']:.4f} | {deeplab['test']['pixel_accuracy']:.4f} |

DeepLabV3+相较U-Net，Dice提高{dice_gain * 100:.2f}个百分点，IoU提高{iou_gain * 100:.2f}个百分点。

## 4. 论文第五章建议引用的图片

- `classification_model_comparison.png`：分类模型总体对比；
- `plantseg_confusion_matrix.png`：野外测试集归一化混淆矩阵；
- `plantseg_class_f1.png`：各病害类别F1；
- `segmentation_model_comparison.png`：分割模型指标对比；
- 分类与分割训练目录中的训练曲线及预测效果图。

## 5. 系统验收要求

使用 `system_acceptance_template.csv` 记录30～60张独立图片，覆盖三种作物、健康/病害、简单/复杂背景。不得使用训练集图片。人工记录真实类别、预测类别、置信度、病斑比例、分割主观评分和失败原因。
"""
    (OUTPUT / "experiment_report.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    old = load_json(PROJECT / "runs" / "mobilenet_v3_small" / "mixed_external_evaluation.json")
    mixed = load_json(PROJECT / "runs" / "mobilenet_v3_large_mixed" / "metrics.json")
    refined = load_json(PROJECT / "runs" / "mobilenet_v3_large_plantdoc" / "metrics.json")
    unet = load_json(PROJECT / "runs" / "unet_plantseg" / "metrics.json")
    deeplab = load_json(PROJECT / "runs" / "deeplabv3plus_plantseg" / "metrics.json")
    save_classification_comparison(old, mixed, refined)
    save_plantseg_f1(refined)
    save_confusion_matrix(refined)
    save_segmentation_comparison(unet, deeplab)
    write_summary_csv(old, mixed, refined, unet, deeplab)
    write_acceptance_template()
    write_markdown(old, mixed, refined, unet, deeplab)
    summary = {
        "output": str(OUTPUT),
        "figures": 4,
        "classification_plantseg_accuracy_gain": refined["plantseg_test"]["accuracy"] - old["plantseg"]["accuracy"],
        "classification_plantdoc_increment": refined["plantseg_test"]["accuracy"] - mixed["plantseg_test"]["accuracy"],
        "segmentation_iou_gain": deeplab["test"]["iou"] - unet["test"]["iou"],
    }
    (OUTPUT / "report_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
