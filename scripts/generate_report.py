"""
CLI wrapper around src/report.py: load a trained segmentation model (and, if available,
a trained classifier), run one case through the pipeline, and save the report figure.

Usage (format="nifti" in config.yaml, --case_id is the case folder name):
    python scripts/generate_report.py --config config.yaml \
        --seg_checkpoint checkpoints/segmentation/best.pt \
        --case_id case_001

Usage (format="h5_slices", --case_id is "volume_<id>"):
    python scripts/generate_report.py --case_id volume_100 \
        --seg_checkpoint checkpoints/segmentation/best.pt

--cls_checkpoint is optional: pass it once a real classifier is trained (see README's
scope note) to add the classification + confidence interval + Grad-CAM panel.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import discover_cases, discover_h5_cases, get_h5_val_transforms, get_val_transforms
from src.models.classifier import SegClassifier
from src.models.unet3d import UNet3D
from src.report import generate_report
from src.utils import load_checkpoint, to_plain_tensor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--seg_checkpoint", type=str, required=True)
    p.add_argument("--cls_checkpoint", type=str, default=None,
                    help="Optional: adds classification + Grad-CAM panel if provided")
    p.add_argument("--case_id", type=str, required=True,
                    help="Case folder name (format=nifti) or 'volume_<id>' (format=h5_slices)")
    p.add_argument("--class_names", nargs="+", default=["LGG", "HGG"])
    p.add_argument("--output", type=str, default="outputs/reports/report.png")
    return p.parse_args()


def load_volume(data_dir: str, case_id: str, format: str) -> torch.Tensor:
    if format == "h5_slices":
        cases = {c["case_id"]: c for c in discover_h5_cases(data_dir)}
        if case_id not in cases:
            raise FileNotFoundError(f"Could not find case '{case_id}' under {data_dir}")
        sample = get_h5_val_transforms()(cases[case_id])
        return to_plain_tensor(sample["image"])

    cases = {c.case_id: c for c in discover_cases(data_dir, require_label=False)}
    if case_id not in cases:
        raise FileNotFoundError(f"Could not find modality files for case '{case_id}' under {data_dir}")
    case = cases[case_id]
    sample = get_val_transforms()({"image": case.images, "label": case.label or case.images[0]})
    return to_plain_tensor(sample["image"])


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dcfg, mcfg, icfg = cfg["data"], cfg["model"], cfg["inference"]

    volume = load_volume(dcfg["data_dir"], args.case_id, dcfg.get("format", "nifti"))

    unet = UNet3D(in_channels=mcfg["in_channels"], num_classes=mcfg["num_seg_classes"],
                   base_filters=mcfg["base_filters"], depth=mcfg["depth"], dropout_p=mcfg["dropout_p"])
    load_checkpoint(args.seg_checkpoint, unet, map_location=device)

    seg_classifier = gradcam_target_layer = None
    if args.cls_checkpoint:
        seg_classifier = SegClassifier(unet, num_classification_classes=mcfg["num_cls_classes"],
                                        dropout_p=mcfg["cls_dropout_p"])
        load_checkpoint(args.cls_checkpoint, seg_classifier, map_location=device)
        gradcam_target_layer = seg_classifier.unet.encoders[-1].conv.block[3]

    result = generate_report(
        volume=volume,
        seg_model=unet,
        seg_classifier=seg_classifier,
        gradcam_target_layer=gradcam_target_layer,
        class_names=args.class_names,
        patch_size=tuple(icfg["patch_size"]),
        overlap=icfg["overlap"],
        num_seg_classes=mcfg["num_seg_classes"],
        device=device,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.figure.savefig(out_path, dpi=150)
    print(f"saved report to {out_path}")
    if result.predicted_class is not None:
        print(f"predicted class: {args.class_names[result.predicted_class]} "
              f"({result.confidence_pct:.1f}% +/- {result.confidence_half_width_pct:.1f}%)")
    print(f"tumor volumes (cm3): {result.tumor_volumes_cm3}")


if __name__ == "__main__":
    main()
