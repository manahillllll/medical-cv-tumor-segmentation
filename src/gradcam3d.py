"""
3D Grad-CAM, adapted from the standard 2D formulation.

Standard Grad-CAM (Selvaraju et al.) computes channel weights as the global-average-pool
of gradients w.r.t. a target conv layer's activations, then forms a weighted sum of
activation maps followed by ReLU. The 3D adaptation here just extends every pooling and
upsampling operation to volumetric (D, H, W) tensors instead of (H, W) — the math is the
same, but there's no standard drop-in library for it, so this is implemented directly
against the UNet3D bottleneck / SegClassifier classification head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM3D:
    """
    Usage:
        cam = GradCAM3D(seg_classifier, target_layer=seg_classifier.unet.encoders[-1].conv.block[3])
        heatmap = cam(x, class_idx=predicted_class)  # (D, H, W) in [0, 1], same size as x

    target_layer should be a Conv3d (or the InstanceNorm/activation right after it) whose
    output activations you want to explain the classification decision with — typically
    the last conv block before the bottleneck, since that's what the classification head
    consumes.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> torch.Tensor:
        """x: (1, C, D, H, W). Returns a (D, H, W) heatmap resized to match x's spatial size."""
        self.model.zero_grad(set_to_none=True)
        was_training = self.model.training
        self.model.eval()

        seg_logits, cls_logits = self.model(x)

        if class_idx is None:
            class_idx = int(cls_logits.argmax(dim=1).item())

        score = cls_logits[0, class_idx]
        score.backward()

        if was_training:
            self.model.train()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire; check target_layer is on the forward path.")

        # weights: global-average-pool the gradients over spatial dims -> per-channel importance
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)  # (1, C, 1, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, d, h, w)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0)  # (D, H, W)

        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min).clamp(min=1e-8)
        return cam.detach()
