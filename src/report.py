"""
Combined clinical-style report: one function that takes a raw scan plus a trained
segmentation model and produces a figure with the segmentation overlay and a
voxel-level uncertainty map. If a trained SegClassifier + Grad-CAM target layer are
also supplied, the report additionally includes the classification decision (with a
calibrated confidence interval) and its Grad-CAM explanation -- but this project ships
without a trained classifier (no real tumor-grade labels for the data source used, see
README), so `seg_classifier=None` is the normal/expected case here.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from .gradcam3d import GradCAM3D
from .inference import sliding_window_inference
from .uncertainty import classification_confidence_interval, mc_dropout_predict
from .utils import overlay_mask_on_slice, voxel_count_to_cm3


@dataclass
class ReportResult:
    segmentation: torch.Tensor            # (D, H, W) predicted class map
    tumor_volumes_cm3: dict[str, float]    # per BraTS region
    uncertainty_map: torch.Tensor         # (D, H, W)
    figure: plt.Figure
    predicted_class: int | None = None
    confidence_pct: float | None = None
    confidence_half_width_pct: float | None = None
    gradcam_map: torch.Tensor | None = None


def _tumor_centered_patch_origin(seg_pred: torch.Tensor, patch_size: tuple[int, int, int],
                                  volume_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Patch origin centered on the tumor's centroid (falls back to volume center if empty)."""
    d, h, w = volume_shape
    pd, ph, pw = patch_size
    tumor_voxels = (seg_pred > 0).nonzero()
    if len(tumor_voxels) == 0:
        center = torch.tensor([d, h, w], dtype=torch.float32) / 2
    else:
        center = tumor_voxels.float().mean(dim=0)
    sd = int(min(max(0, center[0].item() - pd / 2), max(0, d - pd)))
    sh = int(min(max(0, center[1].item() - ph / 2), max(0, h - ph)))
    sw = int(min(max(0, center[2].item() - pw / 2), max(0, w - pw)))
    return sd, sh, sw


def generate_report(
    volume: torch.Tensor,
    seg_model: torch.nn.Module,
    seg_classifier: torch.nn.Module | None = None,
    gradcam_target_layer: torch.nn.Module | None = None,
    class_names: list[str] | None = None,
    patch_size: tuple[int, int, int] = (128, 128, 128),
    overlap: float = 0.5,
    num_seg_classes: int = 4,
    mc_samples: int = 20,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: str | torch.device = "cuda",
    slice_axis: int = 0,
    slice_idx: int | None = None,
) -> ReportResult:
    """
    volume: (C, D, H, W) all modalities for one scan, already preprocessed/normalized.
    seg_model: trained UNet3D (for full-volume sliding-window segmentation, and for
        voxel-level MC-Dropout uncertainty over a tumor-centered patch).
    seg_classifier, gradcam_target_layer, class_names: optional. When provided, adds a
        classification decision (calibrated confidence interval) and its Grad-CAM
        explanation to the report; when seg_classifier is None those panels are simply
        omitted (segmentation + uncertainty only).
    """
    seg_model.to(device).eval()
    volume = volume.to(device)
    has_classifier = seg_classifier is not None

    # 1. Segmentation via sliding-window inference
    probs = sliding_window_inference(
        volume, seg_model, patch_size=patch_size, overlap=overlap,
        num_classes=num_seg_classes, device=device,
    )
    seg_pred = probs.argmax(dim=0)

    voxel_counts = {
        "whole_tumor": int((seg_pred > 0).sum().item()),
        "tumor_core": int(((seg_pred == 1) | (seg_pred == 3)).sum().item()),
        "enhancing_tumor": int((seg_pred == 3).sum().item()),
    }
    tumor_volumes_cm3 = {k: voxel_count_to_cm3(v, spacing_mm) for k, v in voxel_counts.items()}

    # 2. MC-Dropout over a tumor-centered patch (MC sampling the full volume N times is
    #    too expensive; the segmentation itself already tells us where to focus).
    d, h, w = volume.shape[1:]
    pd, ph, pw = patch_size
    sd, sh, sw = _tumor_centered_patch_origin(seg_pred, patch_size, (d, h, w))
    center_patch = volume[:, sd:sd + pd, sh:sh + ph, sw:sw + pw].unsqueeze(0)

    def forward_seg(model, x):
        return model(x)

    seg_mc_out = mc_dropout_predict(seg_model, center_patch, n_samples=mc_samples, forward_fn=forward_seg)
    uncertainty_patch = seg_mc_out["epistemic_uncertainty"].squeeze(0)  # (pd, ph, pw)
    uncertainty_map = torch.zeros((d, h, w), device=device)
    uncertainty_map[sd:sd + pd, sh:sh + ph, sw:sw + pw] = uncertainty_patch

    predicted_class = confidence_pct = half_width_pct = None
    gradcam_map = None

    if has_classifier:
        seg_classifier.to(device).eval()

        def forward_cls(model, x):
            _, cls_logits = model(x)
            return cls_logits

        mc_out = mc_dropout_predict(seg_classifier, center_patch, n_samples=mc_samples, forward_fn=forward_cls)
        predicted_class = int(mc_out["mean_probs"].argmax(dim=1).item())
        cls_samples = mc_out["all_samples"].squeeze(1)  # (N, num_classes)
        confidence_pct, half_width_pct = classification_confidence_interval(cls_samples, predicted_class)

        cam = GradCAM3D(seg_classifier, gradcam_target_layer)
        gradcam_patch = cam(center_patch, class_idx=predicted_class)
        gradcam_map = torch.zeros((d, h, w), device=device)
        gradcam_map[sd:sd + pd, sh:sh + ph, sw:sw + pw] = gradcam_patch

    # 3. Assemble figure
    if slice_idx is None:
        tumor_voxels = (seg_pred > 0).nonzero()
        slice_idx = int(tumor_voxels[:, slice_axis].float().mean().item()) if len(tumor_voxels) else d // 2

    def take_slice(t: torch.Tensor) -> np.ndarray:
        return t.select(slice_axis, slice_idx).cpu().numpy()

    base_img = volume[0]  # T1 modality (channel 0) as grayscale background
    base_slice = take_slice(base_img)
    base_slice = (base_slice - base_slice.min()) / (base_slice.max() - base_slice.min() + 1e-8)

    seg_overlay = overlay_mask_on_slice(base_slice, take_slice(seg_pred))
    uncertainty_slice = take_slice(uncertainty_map)

    n_panels = 4 if has_classifier else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    axes[0].imshow(base_slice, cmap="gray")
    axes[0].set_title("Input (T1)")
    axes[1].imshow(seg_overlay)
    axes[1].set_title(
        f"Segmentation\nWT {tumor_volumes_cm3['whole_tumor']:.1f} cm3 | "
        f"TC {tumor_volumes_cm3['tumor_core']:.1f} cm3 | ET {tumor_volumes_cm3['enhancing_tumor']:.1f} cm3"
    )
    axes[2].imshow(base_slice, cmap="gray")
    im2 = axes[2].imshow(uncertainty_slice, cmap="inferno", alpha=0.6)
    axes[2].set_title("Epistemic uncertainty\n(brighter = model less sure)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    if has_classifier:
        gradcam_slice = take_slice(gradcam_map)
        axes[3].imshow(base_slice, cmap="gray")
        im3 = axes[3].imshow(gradcam_slice, cmap="jet", alpha=0.5)
        axes[3].set_title(
            f"Grad-CAM: {class_names[predicted_class]}\n"
            f"{confidence_pct:.1f}% +/- {half_width_pct:.1f}%"
        )
        fig.colorbar(im3, ax=axes[3], fraction=0.046)

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()

    return ReportResult(
        segmentation=seg_pred.cpu(),
        tumor_volumes_cm3=tumor_volumes_cm3,
        uncertainty_map=uncertainty_map.cpu(),
        figure=fig,
        predicted_class=predicted_class,
        confidence_pct=confidence_pct,
        confidence_half_width_pct=half_width_pct,
        gradcam_map=gradcam_map.cpu() if gradcam_map is not None else None,
    )
