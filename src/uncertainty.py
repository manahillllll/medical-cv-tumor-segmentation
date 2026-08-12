"""
Uncertainty quantification: Monte Carlo Dropout and Deep Ensembles.

Neither approach gives a ground-truth uncertainty to check against, so the way to
sanity-check this module is (a) confirm uncertainty is visibly higher near tumor
boundaries / ambiguous regions than in confidently-background voxels, and (b) check
calibration on held-out data via src/metrics.py's reliability diagram.
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def enable_mc_dropout(model: nn.Module) -> None:
    """Set only Dropout* layers to train mode, keep BatchNorm/InstanceNorm etc. in eval mode."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 20,
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Run N stochastic forward passes with dropout active and aggregate.

    forward_fn defaults to `model(x)`; pass a custom one for models that return a
    tuple (e.g. SegClassifier -> (seg_logits, cls_logits)) and want the seg logits.

    Returns dict with:
        mean_probs: (C, ...) mean softmax probability
        epistemic_uncertainty: (...) voxel/scan-level variance of the predicted class prob
        predictive_entropy: (...) entropy of the mean prediction (total uncertainty)
    """
    enable_mc_dropout(model)
    forward_fn = forward_fn or (lambda m, inp: m(inp))

    samples = []
    for _ in range(n_samples):
        logits = forward_fn(model, x)
        probs = F.softmax(logits, dim=1)
        samples.append(probs)
    stacked = torch.stack(samples, dim=0)  # (N, B, C, ...)

    mean_probs = stacked.mean(dim=0)
    variance = stacked.var(dim=0)  # (B, C, ...)
    epistemic_uncertainty = variance.mean(dim=1)  # average across classes -> (B, ...)

    entropy = -(mean_probs.clamp(min=1e-8) * mean_probs.clamp(min=1e-8).log()).sum(dim=1)

    return {
        "mean_probs": mean_probs,
        "epistemic_uncertainty": epistemic_uncertainty,
        "predictive_entropy": entropy,
        "all_samples": stacked,
    }


@torch.no_grad()
def ensemble_predict(
    models: Sequence[nn.Module],
    x: torch.Tensor,
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Deep ensemble uncertainty: disagreement across independently-trained models.
    More robust than MC Dropout, more compute (train 3-5 models with different seeds).
    Same return signature as mc_dropout_predict for interchangeability.
    """
    forward_fn = forward_fn or (lambda m, inp: m(inp))

    samples = []
    for model in models:
        model.eval()
        logits = forward_fn(model, x)
        probs = F.softmax(logits, dim=1)
        samples.append(probs)
    stacked = torch.stack(samples, dim=0)

    mean_probs = stacked.mean(dim=0)
    variance = stacked.var(dim=0)
    epistemic_uncertainty = variance.mean(dim=1)
    entropy = -(mean_probs.clamp(min=1e-8) * mean_probs.clamp(min=1e-8).log()).sum(dim=1)

    return {
        "mean_probs": mean_probs,
        "epistemic_uncertainty": epistemic_uncertainty,
        "predictive_entropy": entropy,
        "all_samples": stacked,
    }


def classification_confidence_interval(cls_probs_samples: torch.Tensor, class_idx: int, z: float = 1.96):
    """
    cls_probs_samples: (N, num_classes) softmax probs across MC/ensemble samples for one scan.
    Returns (mean_confidence_pct, half_width_pct) e.g. (87.0, 6.0) -> "87% +/- 6%".
    """
    class_probs = cls_probs_samples[:, class_idx]
    mean = class_probs.mean().item() * 100
    std = class_probs.std().item() * 100
    half_width = z * std / (len(class_probs) ** 0.5)
    return mean, half_width
