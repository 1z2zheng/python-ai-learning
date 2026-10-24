from __future__ import annotations

import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().cpu().reshape(-1)
        target = target.detach().cpu().reshape(-1)
        valid = (target >= 0) & (target < self.num_classes)
        encoded = self.num_classes * target[valid] + prediction[valid]
        self.confusion += torch.bincount(
            encoded, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, object]:
        matrix = self.confusion.float()
        true_positive = matrix.diag()
        predicted = matrix.sum(dim=0)
        actual = matrix.sum(dim=1)
        union = predicted + actual - true_positive
        iou = torch.where(union > 0, true_positive / union, torch.nan)
        precision = torch.where(predicted > 0, true_positive / predicted, torch.nan)
        recall = torch.where(actual > 0, true_positive / actual, torch.nan)
        f1 = torch.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            torch.nan,
        )
        pixel_accuracy = true_positive.sum() / matrix.sum().clamp_min(1)
        return {
            "pixel_accuracy": float(pixel_accuracy),
            "miou": float(torch.nanmean(iou)),
            "macro_f1": float(torch.nanmean(f1)),
            "per_class_iou": [None if torch.isnan(v) else float(v) for v in iou],
            "per_class_precision": [None if torch.isnan(v) else float(v) for v in precision],
            "per_class_recall": [None if torch.isnan(v) else float(v) for v in recall],
            "per_class_f1": [None if torch.isnan(v) else float(v) for v in f1],
            "confusion_matrix": self.confusion.tolist(),
        }

