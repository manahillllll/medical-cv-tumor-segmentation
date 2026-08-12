"""
Classification head that branches off the 3D U-Net's bottleneck features.

Rather than training a separate feature extractor, this reuses the spatial
representation the segmentation encoder already learned (transfer within the model).
Global average pooling collapses the bottleneck volume to a single feature vector so
the head works regardless of input patch/volume size.

A Dropout layer sits before the final linear layer so the whole seg+classifier stack
can be run under Monte Carlo Dropout for uncertainty estimation (src/uncertainty.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout_p: float = 0.3, hidden_dim: int = 128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, bottleneck_features: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(bottleneck_features)
        return self.net(pooled)


class SegClassifier(nn.Module):
    """
    Wraps a UNet3D and a ClassificationHead so a single forward pass produces both
    the voxel-level segmentation logits and the scan-level classification logits.

    `freeze_encoder=True` stops gradients from the classification loss flowing back
    into the segmentation encoder/decoder — useful for the first stage of training the
    head (per milestone 5: "possibly with encoder frozen first, then fine-tuned end-to-end").
    """

    def __init__(self, unet: nn.Module, num_classification_classes: int, dropout_p: float = 0.3,
                 freeze_encoder: bool = False):
        super().__init__()
        self.unet = unet
        self.freeze_encoder = freeze_encoder
        self.classifier = ClassificationHead(
            in_channels=unet.bottleneck_channels,
            num_classes=num_classification_classes,
            dropout_p=dropout_p,
        )

    def forward(self, x: torch.Tensor):
        if self.freeze_encoder:
            with torch.no_grad():
                seg_logits, bottleneck = self.unet(x, return_features=True)
            bottleneck = bottleneck.detach()
        else:
            seg_logits, bottleneck = self.unet(x, return_features=True)

        cls_logits = self.classifier(bottleneck)
        return seg_logits, cls_logits
