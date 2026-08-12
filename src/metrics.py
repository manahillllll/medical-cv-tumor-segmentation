"""
Evaluation metrics: per-subregion Dice score, and calibration (reliability diagram +
Expected Calibration Error) for the uncertainty-aware classification head.

Calibration matters because there's no ground-truth "uncertainty" to compare against
directly — the standard proxy is checking whether stated confidence matches empirical
accuracy (does 80% confidence correspond to being right ~80% of the time?).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch


@torch.no_grad()
def dice_score_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
                          smooth: float = 1e-5) -> torch.Tensor:
    """
    pred, target: (D, H, W) integer class maps (pred = argmax of softmax output).
    Returns: (num_classes,) Dice score per class.
    """
    scores = torch.zeros(num_classes)
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        scores[c] = (2 * intersection + smooth) / (union + smooth)
    return scores


@torch.no_grad()
def brats_region_dice(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """
    Standard BraTS composite regions (labels here are the remapped {0,1,2,3} scheme:
    1=necrotic core, 2=edema, 3=enhancing tumor):
        Whole Tumor (WT)  = 1 + 2 + 3
        Tumor Core (TC)   = 1 + 3
        Enhancing (ET)    = 3
    Reported this way because it's what published BraTS baselines report — needed to
    know whether your numbers are in a reasonable range.
    """
    def dice(pred_mask, target_mask, smooth=1e-5):
        intersection = (pred_mask & target_mask).sum().float()
        union = pred_mask.sum().float() + target_mask.sum().float()
        return ((2 * intersection + smooth) / (union + smooth)).item()

    wt_pred, wt_t = pred > 0, target > 0
    tc_pred, tc_t = (pred == 1) | (pred == 3), (target == 1) | (target == 3)
    et_pred, et_t = pred == 3, target == 3

    return {
        "whole_tumor": dice(wt_pred, wt_t),
        "tumor_core": dice(tc_pred, tc_t),
        "enhancing_tumor": dice(et_pred, et_t),
    }


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """
    confidences: (N,) predicted probability of the predicted class, in [0, 1].
    correct: (N,) boolean array, whether the prediction was actually correct.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_diagram(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10,
                         save_path: str | None = None, title: str = "Reliability Diagram"):
    """
    Plots empirical accuracy vs. stated confidence per bin, against the perfect-calibration
    diagonal. A model that's well-calibrated hugs the diagonal; bars below the diagonal
    mean the model is overconfident in that confidence range.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    accs, counts = [], []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        accs.append(correct[mask].mean() if mask.sum() > 0 else 0.0)
        counts.append(mask.sum())

    ece = expected_calibration_error(confidences, correct, n_bins)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.bar(bin_centers, accs, width=1.0 / n_bins, edgecolor="black", alpha=0.7, label="model")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, ece
