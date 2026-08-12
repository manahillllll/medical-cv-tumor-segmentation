"""Small shared helpers: seeding, checkpointing, and volume overlay visualization."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def to_plain_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Strip MONAI's MetaTensor wrapper (and its per-sample metadata dict) down to a
    plain torch.Tensor. Necessary right after pulling a batch off the DataLoader:
    MetaTensor's batched metadata doesn't collate reliably across randomly-cropped
    patches with heterogeneous per-sample metadata (e.g. from RandCropByPosNegLabeld),
    and propagating it through loss computations can raise on later ops like slicing.
    """
    return x.as_tensor() if hasattr(x, "as_tensor") else x


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer=None, epoch: int = 0,
                     extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), "epoch": epoch}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer=None, map_location="cpu") -> dict:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state


def voxel_count_to_cm3(voxel_count: int, spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """Convert a voxel count to volume in cm^3, given voxel spacing in mm."""
    voxel_volume_mm3 = spacing_mm[0] * spacing_mm[1] * spacing_mm[2]
    return voxel_count * voxel_volume_mm3 / 1000.0


def overlay_mask_on_slice(image_slice: np.ndarray, mask_slice: np.ndarray, alpha: float = 0.4,
                           cmap_colors: dict[int, tuple[float, float, float]] | None = None) -> np.ndarray:
    """
    image_slice: (H, W) grayscale, already normalized to [0, 1].
    mask_slice: (H, W) integer class map.
    Returns an (H, W, 3) RGB image with the mask alpha-blended over the grayscale scan.
    """
    cmap_colors = cmap_colors or {
        1: (0.85, 0.1, 0.1),   # necrotic core - red
        2: (0.1, 0.6, 0.85),   # edema - blue
        3: (0.95, 0.85, 0.1),  # enhancing tumor - yellow
    }
    rgb = np.stack([image_slice] * 3, axis=-1)
    out = rgb.copy()
    for cls, color in cmap_colors.items():
        m = mask_slice == cls
        for c in range(3):
            out[..., c][m] = (1 - alpha) * rgb[..., c][m] + alpha * color[c]
    return np.clip(out, 0, 1)
