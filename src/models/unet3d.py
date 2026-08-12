"""
3D U-Net implemented from scratch (no MONAI/segmentation_models_pytorch prebuilt network).

Encoder-decoder with 3D convolutions, instance normalization, LeakyReLU, and skip
connections. Dropout3d is included in the bottleneck and decoder so the same network
can be reused for Monte Carlo Dropout uncertainty estimation at inference time
(see src/uncertainty.py) without any architecture changes.

The encoder's bottleneck feature map is exposed via `forward(..., return_features=True)`
so the classification head (src/models/classifier.py) can branch off it instead of
training a separate feature extractor from scratch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two (Conv3d -> InstanceNorm3d -> LeakyReLU) layers."""

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.dropout = nn.Dropout3d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.block(x))


class Down(nn.Module):
    """Strided-conv downsample followed by a ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        self.downsample = nn.Conv3d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.downsample(x))


class Up(nn.Module):
    """Transposed-conv upsample, concat skip connection, ConvBlock."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Guard against off-by-one shape mismatches from non-divisible patch sizes.
        diff = [skip.shape[i + 2] - x.shape[i + 2] for i in range(3)]
        if any(diff):
            x = F.pad(x, [diff[2] // 2, diff[2] - diff[2] // 2,
                          diff[1] // 2, diff[1] - diff[1] // 2,
                          diff[0] // 2, diff[0] - diff[0] // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    Configurable 3D U-Net.

    Args:
        in_channels: number of input modalities (BraTS: T1, T1ce, T2, FLAIR -> 4)
        num_classes: number of output segmentation classes, including background
            (BraTS regions: background, necrotic core, edema, enhancing tumor -> 4)
        base_filters: number of filters in the first encoder stage; doubles each stage
        depth: number of downsampling stages
        dropout_p: dropout applied in bottleneck + decoder blocks (used for MC Dropout)
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_filters: int = 16,
        depth: int = 4,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.depth = depth
        self.bottleneck_channels = base_filters * (2 ** depth)

        self.stem = ConvBlock(in_channels, base_filters)

        enc_channels = [base_filters * (2 ** i) for i in range(depth + 1)]
        self.encoders = nn.ModuleList([
            Down(enc_channels[i], enc_channels[i + 1], dropout_p=dropout_p if i >= depth - 2 else 0.0)
            for i in range(depth)
        ])

        dec_channels = list(reversed(enc_channels))
        self.decoders = nn.ModuleList([
            Up(dec_channels[i], dec_channels[i + 1], dec_channels[i + 1], dropout_p=dropout_p if i < 2 else 0.0)
            for i in range(depth)
        ])

        self.head = nn.Conv3d(base_filters, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        skips = [self.stem(x)]
        for enc in self.encoders:
            skips.append(enc(skips[-1]))

        bottleneck = skips[-1]
        x = bottleneck
        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 2)]
            x = dec(x, skip)

        logits = self.head(x)

        if return_features:
            return logits, bottleneck
        return logits


if __name__ == "__main__":
    model = UNet3D(in_channels=4, num_classes=4, base_filters=16, depth=4)
    dummy = torch.randn(1, 4, 96, 96, 96)
    out, feats = model(dummy, return_features=True)
    print("output:", out.shape)       # expect (1, 4, 96, 96, 96)
    print("bottleneck:", feats.shape)  # expect (1, 256, 6, 6, 6)
