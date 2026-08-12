"""
Combined Dice + Cross-Entropy loss for multi-class 3D segmentation, implemented from
scratch. Dice alone struggles with class-imbalance edge cases (background is >95% of
voxels in BraTS); cross-entropy alone tends to underweight small structures like the
enhancing tumor core. The weighted combination is the standard fix.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """labels: (B, D, H, W) integer class map -> (B, C, D, H, W) one-hot."""
    return F.one_hot(labels.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


class SoftDiceLoss(nn.Module):
    """Multi-class soft Dice loss, averaged over classes (optionally excluding background)."""

    def __init__(self, num_classes: int, smooth: float = 1e-5, include_background: bool = True,
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background
        self.register_buffer("class_weights", class_weights, persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_oh = one_hot(targets, self.num_classes)

        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets_oh, dim=dims)
        cardinality = torch.sum(probs + targets_oh, dim=dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss_per_class = 1.0 - dice_per_class

        start = 0 if self.include_background else 1
        loss_per_class = loss_per_class[start:]

        if self.class_weights is not None:
            weights = self.class_weights[start:]
            return (loss_per_class * weights).sum() / weights.sum()
        return loss_per_class.mean()


class DiceCELoss(nn.Module):
    """
    weight_dice * DiceLoss + weight_ce * CrossEntropyLoss(with per-class weights).

    class_weights: tensor of shape (num_classes,), typically inverse voxel-frequency,
    used both to reweight cross-entropy and (optionally) the per-class Dice terms.
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
        weight_dice: float = 1.0,
        weight_ce: float = 1.0,
        include_background_in_dice: bool = True,
    ):
        super().__init__()
        self.dice = SoftDiceLoss(
            num_classes=num_classes,
            include_background=include_background_in_dice,
            class_weights=class_weights,
        )
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, targets)
        ce_loss = self.ce(logits, targets.long())
        return self.weight_dice * dice_loss + self.weight_ce * ce_loss


def compute_class_weights(label_volumes: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Inverse-frequency class weights from a batch/dataset of integer label volumes.
    Pass a stacked tensor of label maps (e.g. sampled from the training set) — used to
    counter the extreme background-vs-tumor voxel imbalance in BraTS.
    """
    counts = torch.zeros(num_classes)
    flat = label_volumes.flatten().long()
    for c in range(num_classes):
        counts[c] = (flat == c).sum()
    freq = counts / counts.sum().clamp(min=eps)
    weights = 1.0 / (freq + eps)
    weights = weights / weights.sum() * num_classes
    return weights
