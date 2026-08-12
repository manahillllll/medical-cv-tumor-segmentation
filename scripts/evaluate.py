"""
Full-volume evaluation: sliding-window inference + per-subregion Dice on the validation
split, compared against published BraTS baselines (milestone 3/4).

Usage:
    python scripts/evaluate.py --config config.yaml --checkpoint checkpoints/segmentation/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import build_datasets
from src.inference import sliding_window_inference
from src.metrics import brats_region_dice, dice_score_per_class
from src.models.unet3d import UNet3D
from src.utils import load_checkpoint, to_plain_tensor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="outputs/eval_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dcfg, mcfg, icfg = cfg["data"], cfg["model"], cfg["inference"]

    _, val_ds = build_datasets(dcfg["data_dir"], patch_size=tuple(dcfg["patch_size"]),
                                val_fraction=dcfg["val_fraction"], cache=False,
                                format=dcfg.get("format", "nifti"))

    model = UNet3D(in_channels=mcfg["in_channels"], num_classes=mcfg["num_seg_classes"],
                    base_filters=mcfg["base_filters"], depth=mcfg["depth"], dropout_p=mcfg["dropout_p"])
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.to(device).eval()

    per_class_scores = []
    region_scores = []

    for sample in tqdm(val_ds, desc="evaluating"):
        image = to_plain_tensor(sample["image"])
        label = to_plain_tensor(sample["label"]).squeeze(0)

        probs = sliding_window_inference(
            image, model, patch_size=tuple(icfg["patch_size"]), overlap=icfg["overlap"],
            num_classes=mcfg["num_seg_classes"], device=device,
            batch_size=icfg["sliding_window_batch_size"],
        )
        pred = probs.argmax(dim=0).cpu()

        per_class_scores.append(dice_score_per_class(pred, label, mcfg["num_seg_classes"]))
        region_scores.append(brats_region_dice(pred, label))

    per_class_mean = torch.stack(per_class_scores).mean(dim=0).tolist()
    region_mean = {
        k: sum(s[k] for s in region_scores) / len(region_scores) for k in region_scores[0]
    }

    results = {
        "per_class_dice": per_class_mean,
        "region_dice": region_mean,
        "num_val_cases": len(val_ds),
    }
    print(json.dumps(results, indent=2))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
