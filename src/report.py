"""
Combined clinical-style report: one function that takes a raw scan plus trained
models and produces a single figure with segmentation overlay, classification +
confidence interval, uncertainty heatmap, and Grad-CAM explanation (milestone 8).
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
    segmentation: torch.Tensor          # (D, H, W) predicted class map
    tumor_volumes_cm3: dict[str, float]  # per BraTS region
    predicted_class: int
    confidence_pct: float
    confidence_half_width_pct: float
    uncertainty_map: torch.Tensor       # (D, H, W)
    gradcam_map: torch.Tensor           # (D, H, W)
    figure: plt.Figure


def generate_report(
    volume: torch.Tensor,
    seg_model: torch.nn.Module,
    seg_classifier: torch.nn.Module,
    gradcam_target_layer: torch.nn.Module,
    class_names: list[str],
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
    seg_model: trained UNet3D (for full-volume sliding-window segmentation).
    seg_classifier: trained SegClassifier (for classification + uncertainty + Grad-CAM);
        should share weights with seg_model's encoder for the report to be self-consistent.
    """
    seg_model.to(device).eval()
    seg_classifier.to(device).eval()
    volume = volume.to(device)

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

    # 2. Classification + MC-Dropout uncertainty (a center patch keeps this tractable;
    #    swap in a tumor-centered crop if the scan's tumor location is known up front)
    d, h, w = volume.shape[1:]
    pd, ph, pw = patch_size
    sd, sh, sw = max(0, (d - pd) // 2), max(0, (h - ph) // 2), max(0, (w - pw) // 2)
    center_patch = volume[:, sd:sd + pd, sh:sh + ph, sw:sw + pw].unsqueeze(0)

    def forward_cls(model, x):
        _, cls_logits = model(x)
        return cls_logits

    mc_out = mc_dropout_predict(seg_classifier, center_patch, n_samples=mc_samples, forward_fn=forward_cls)
    predicted_class = int(mc_out["mean_probs"].argmax(dim=1).item())
    cls_samples = mc_out["all_samples"].squeeze(1)  # (N, num_classes)
    conf_pct, half_width_pct = classification_confidence_interval(cls_samples, predicted_class)

    # voxel-level epistemic uncertainty over the segmentation (separate MC pass on seg head)
    def forward_seg(model, x):
        return model(x)

    seg_mc_out = mc_dropout_predict(seg_model, center_patch, n_samples=mc_samples, forward_fn=forward_seg)
    uncertainty_patch = seg_mc_out["epistemic_uncertainty"].squeeze(0)  # (pd, ph, pw)
    uncertainty_map = torch.zeros((d, h, w), device=device)
    uncertainty_map[sd:sd + pd, sh:sh + ph, sw:sw + pw] = uncertainty_patch

    # 3. Grad-CAM explanation for the predicted class
    cam = GradCAM3D(seg_classifier, gradcam_target_layer)
    gradcam_patch = cam(center_patch, class_idx=predicted_class)
    gradcam_map = torch.zeros((d, h, w), device=device)
    gradcam_map[sd:sd + pd, sh:sh + ph, sw:sw + pw] = gradcam_patch

    # 4. Assemble figure
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
    gradcam_slice = take_slice(gradcam_map)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
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
    axes[3].imshow(base_slice, cmap="gray")
    im3 = axes[3].imshow(gradcam_slice, cmap="jet", alpha=0.5)
    axes[3].set_title(
        f"Grad-CAM: {class_names[predicted_class]}\n"
        f"{conf_pct:.1f}% +/- {half_width_pct:.1f}%"
    )
    fig.colorbar(im3, ax=axes[3], fraction=0.046)

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()

    return ReportResult(
        segmentation=seg_pred.cpu(),
        tumor_volumes_cm3=tumor_volumes_cm3,
        predicted_class=predicted_class,
        confidence_pct=conf_pct,
        confidence_half_width_pct=half_width_pct,
        uncertainty_map=uncertainty_map.cpu(),
        gradcam_map=gradcam_map.cpu(),
        figure=fig,
    )
