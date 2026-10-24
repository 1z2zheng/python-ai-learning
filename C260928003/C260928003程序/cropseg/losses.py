from __future__ import annotations

import torch
import torch.nn.functional as F


def multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_background: bool = False,
    smooth: float = 1.0,
) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probabilities * one_hot, dims)
    denominator = torch.sum(probabilities + one_hot, dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    if ignore_background and num_classes > 1:
        dice = dice[1:]
    return 1.0 - dice.mean()


def foreground_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Binary foreground Tversky loss; beta > alpha penalizes missed pixels more."""
    foreground_probability = torch.softmax(logits, dim=1)[:, 1]
    foreground_target = (target == 1).float()
    dims = (1, 2)
    true_positive = torch.sum(foreground_probability * foreground_target, dims)
    false_positive = torch.sum(foreground_probability * (1.0 - foreground_target), dims)
    false_negative = torch.sum((1.0 - foreground_probability) * foreground_target, dims)
    score = (true_positive + smooth) / (
        true_positive + alpha * false_positive + beta * false_negative + smooth
    )
    return 1.0 - score.mean()


def combined_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ce_weight: float,
    dice_weight: float,
    ignore_background_in_dice: bool,
    foreground_ce_weight: float = 1.0,
    tversky_weight: float = 0.0,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
) -> torch.Tensor:
    class_weights = None
    if foreground_ce_weight != 1.0 and num_classes > 1:
        class_weights = torch.ones(num_classes, device=logits.device, dtype=logits.dtype)
        class_weights[1] = foreground_ce_weight
    ce = F.cross_entropy(logits, target, weight=class_weights)
    dice = multiclass_dice_loss(logits, target, num_classes, ignore_background_in_dice)
    loss = ce_weight * ce + dice_weight * dice
    if tversky_weight:
        if num_classes != 2:
            raise ValueError("foreground_tversky_loss目前仅支持二分类语义分割")
        loss = loss + tversky_weight * foreground_tversky_loss(
            logits, target, tversky_alpha, tversky_beta
        )
    return loss

