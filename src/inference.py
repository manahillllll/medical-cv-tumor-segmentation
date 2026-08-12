"""
Sliding-window inference over full 3D volumes with Gaussian-weighted overlap-averaging.

Full BraTS volumes (~240x240x155) rarely fit in GPU memory at training resolution, so
training happens on patches (see src/data/dataset.py). At inference time we tile the
full volume into overlapping patches, run the model on each, and blend overlapping
predictions with a Gaussian weighting centered on each patch — naive averaging or
non-overlapping tiling both produce visible seam artifacts at patch boundaries.
"""
from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


def _gaussian_kernel(patch_size: tuple[int, int, int], sigma_scale: float = 0.125) -> torch.Tensor:
    """Separable 3D Gaussian weighting window, peak at the patch center, min clamped away from 0."""
    coords = []
    for dim in patch_size:
        sigma = dim * sigma_scale
        x = torch.arange(dim, dtype=torch.float32) - (dim - 1) / 2.0
        coords.append(torch.exp(-(x ** 2) / (2 * sigma ** 2)))
    kernel = coords[0].view(-1, 1, 1) * coords[1].view(1, -1, 1) * coords[2].view(1, 1, -1)
    kernel = kernel / kernel.max()
    kernel = kernel.clamp(min=1e-3)
    return kernel


def _get_patch_starts(dim_size: int, patch_size: int, overlap: float) -> list[int]:
    stride = max(1, int(patch_size * (1 - overlap)))
    starts = list(range(0, max(dim_size - patch_size, 0) + 1, stride))
    last = dim_size - patch_size
    if last < 0:
        return [0]
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def sliding_window_inference(
    volume: torch.Tensor,
    model: Callable[[torch.Tensor], torch.Tensor],
    patch_size: tuple[int, int, int] = (96, 96, 96),
    overlap: float = 0.5,
    num_classes: int = 4,
    device: str | torch.device = "cuda",
    batch_size: int = 1,
) -> torch.Tensor:
    """
    volume: (C, D, H, W) tensor (single scan, all modalities stacked).
    model: callable mapping (B, C, pd, ph, pw) -> (B, num_classes, pd, ph, pw) logits.
    Returns: (num_classes, D, H, W) averaged softmax probabilities.
    """
    c, d, h, w = volume.shape
    pd, ph, pw = patch_size

    pad_d, pad_h, pad_w = max(0, pd - d), max(0, ph - h), max(0, pw - w)
    if pad_d or pad_h or pad_w:
        volume = F.pad(volume, [0, pad_w, 0, pad_h, 0, pad_d])
    _, d_p, h_p, w_p = volume.shape

    starts_d = _get_patch_starts(d_p, pd, overlap)
    starts_h = _get_patch_starts(h_p, ph, overlap)
    starts_w = _get_patch_starts(w_p, pw, overlap)

    gaussian = _gaussian_kernel(patch_size).to(device)

    prob_sum = torch.zeros((num_classes, d_p, h_p, w_p), device=device)
    weight_sum = torch.zeros((1, d_p, h_p, w_p), device=device)

    volume = volume.to(device)
    coords = list(itertools.product(starts_d, starts_h, starts_w))

    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i + batch_size]
            patches = torch.stack([
                volume[:, sd:sd + pd, sh:sh + ph, sw:sw + pw] for sd, sh, sw in batch_coords
            ], dim=0)
            logits = model(patches)
            probs = F.softmax(logits, dim=1)

            for j, (sd, sh, sw) in enumerate(batch_coords):
                weighted = probs[j] * gaussian.unsqueeze(0)
                prob_sum[:, sd:sd + pd, sh:sh + ph, sw:sw + pw] += weighted
                weight_sum[:, sd:sd + pd, sh:sh + ph, sw:sw + pw] += gaussian.unsqueeze(0)

    averaged = prob_sum / weight_sum.clamp(min=1e-8)
    return averaged[:, :d, :h, :w]


if __name__ == "__main__":
    from models.unet3d import UNet3D

    model = UNet3D(in_channels=4, num_classes=4, base_filters=8, depth=3).eval()
    volume = torch.randn(4, 155, 180, 180)
    probs = sliding_window_inference(volume, model, patch_size=(64, 64, 64), device="cpu")
    print(probs.shape)  # expect (4, 155, 180, 180)
