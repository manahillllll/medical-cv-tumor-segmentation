"""
Generates illustrative README preview images using the real pipeline code, but on
synthetic random data and untrained (randomly-initialized) weights.

This is NOT a results script — it exists purely so the README can show what the
combined report output and calibration diagram *look like* before real BraTS
training has happened. Re-run scripts/generate_report.py and scripts/evaluate.py
on real trained checkpoints to produce actual results.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import reliability_diagram
from src.models.classifier import SegClassifier
from src.models.unet3d import UNet3D
from src.report import generate_report
from src.utils import set_seed

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)


def make_synthetic_volume(shape=(4, 64, 64, 64), seed=0) -> torch.Tensor:
    """A random volume with a blob of 'signal' in the middle so the segmentation
    head has something spatially coherent to (randomly) react to."""
    rng = np.random.RandomState(seed)
    vol = rng.randn(*shape).astype(np.float32) * 0.3
    c, d, h, w = shape
    zz, yy, xx = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    center = np.array([d, h, w]) / 2
    dist = np.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)
    blob = np.exp(-(dist ** 2) / (2 * (d * 0.15) ** 2))
    for ch in range(c):
        vol[ch] += blob * (0.8 + 0.2 * ch)
    return torch.from_numpy(vol)


def main():
    set_seed(0)
    device = "cpu"
    patch_size = (32, 32, 32)

    unet = UNet3D(in_channels=4, num_classes=4, base_filters=8, depth=3, dropout_p=0.2)
    model = SegClassifier(unet, num_classification_classes=2, dropout_p=0.3)
    gradcam_target_layer = model.unet.encoders[-1].conv.block[3]

    volume = make_synthetic_volume(shape=(4, 64, 64, 64))

    result = generate_report(
        volume=volume,
        seg_model=model.unet,
        seg_classifier=model,
        gradcam_target_layer=gradcam_target_layer,
        class_names=["LGG", "HGG"],
        patch_size=patch_size,
        overlap=0.5,
        num_seg_classes=4,
        mc_samples=8,
        device=device,
    )
    out_path = ASSETS_DIR / "sample_report.png"
    result.figure.savefig(out_path, dpi=150, facecolor="white")
    print(f"saved {out_path}")

    # Illustrative reliability diagram (synthetic, deliberately mildly overconfident
    # to show what a *miscalibrated* model looks like vs. the diagonal).
    rng = np.random.RandomState(1)
    n = 2000
    confidences = rng.beta(5, 2, size=n)
    true_prob_correct = confidences * 0.8
    correct = rng.uniform(size=n) < true_prob_correct
    fig, ece = reliability_diagram(
        confidences, correct, n_bins=10,
        save_path=str(ASSETS_DIR / "reliability_diagram_example.png"),
        title="Reliability Diagram (illustrative example)",
    )
    print(f"saved {ASSETS_DIR / 'reliability_diagram_example.png'} (ECE={ece:.3f})")


if __name__ == "__main__":
    main()
