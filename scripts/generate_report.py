"""
CLI wrapper around src/report.py: load a trained segmentation model + classifier,
run one case through the full pipeline, and save the combined report figure.

Usage:
    python scripts/generate_report.py --config config.yaml \
        --seg_checkpoint checkpoints/segmentation/best.pt \
        --cls_checkpoint checkpoints/classifier/best.pt \
        --case_dir data/brats/case_001 \
        --class_names LGG HGG
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import discover_cases, get_val_transforms
from src.models.classifier import SegClassifier
from src.models.unet3d import UNet3D
from src.report import generate_report
from src.utils import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--seg_checkpoint", type=str, required=True)
    p.add_argument("--cls_checkpoint", type=str, required=True)
    p.add_argument("--case_dir", type=str, required=True)
    p.add_argument("--class_names", nargs="+", default=["LGG", "HGG"])
    p.add_argument("--output", type=str, default="outputs/reports/report.png")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcfg, icfg = cfg["model"], cfg["inference"]

    case_parent = str(Path(args.case_dir).parent)
    case_name = Path(args.case_dir).name
    cases = {c.case_id: c for c in discover_cases(case_parent, require_label=False)}
    if case_name not in cases:
        raise FileNotFoundError(f"Could not find modality files for case '{case_name}' under {case_parent}")
    case = cases[case_name]

    sample = get_val_transforms()({"image": case.images, "label": case.label or case.images[0]})
    volume = sample["image"]

    unet = UNet3D(in_channels=mcfg["in_channels"], num_classes=mcfg["num_seg_classes"],
                   base_filters=mcfg["base_filters"], depth=mcfg["depth"], dropout_p=mcfg["dropout_p"])
    load_checkpoint(args.seg_checkpoint, unet, map_location=device)

    seg_classifier = SegClassifier(unet, num_classification_classes=mcfg["num_cls_classes"],
                                    dropout_p=mcfg["cls_dropout_p"])
    load_checkpoint(args.cls_checkpoint, seg_classifier, map_location=device)

    gradcam_target_layer = seg_classifier.unet.encoders[-1].conv.block[3]

    result = generate_report(
        volume=volume,
        seg_model=seg_classifier.unet,
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
    print(f"predicted class: {args.class_names[result.predicted_class]} "
          f"({result.confidence_pct:.1f}% +/- {result.confidence_half_width_pct:.1f}%)")
    print(f"tumor volumes (cm3): {result.tumor_volumes_cm3}")


if __name__ == "__main__":
    main()
