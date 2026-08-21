"""
Exports a handful of held-out validation cases as compact .npz files for the
Streamlit demo (app.py) to load instantly, instead of needing the full ~7.6GB
BraTS2020 h5 dataset on hand just to click through the demo.

Not committed to the repo (app_data/ is gitignored) -- BraTS-derived data carries its
own redistribution terms (see README's Data license note), so this stays local, built
from data you already downloaded yourself.

Usage:
    python scripts/export_demo_cases.py --config config.yaml --n_cases 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import _split_train_val, discover_h5_cases, get_h5_val_transforms
from src.utils import to_plain_tensor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--n_cases", type=int, default=5)
    p.add_argument("--output_dir", type=str, default="app_data/example_cases")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg["data"]

    if dcfg.get("format", "nifti") != "h5_slices":
        raise SystemExit("export_demo_cases.py currently only supports format='h5_slices'.")

    cases = discover_h5_cases(dcfg["data_dir"])
    _, val_cases = _split_train_val(cases, dcfg["val_fraction"], cfg["seed"])
    print(f"{len(val_cases)} held-out validation cases available.")

    transform = get_h5_val_transforms()

    # Rank by whole-tumor size so the demo set spans small/medium/large tumors
    # rather than whatever happened to sort first.
    sized = []
    for case in tqdm(val_cases, desc="scanning tumor sizes"):
        sample = transform(case)
        label = to_plain_tensor(sample["label"]).squeeze(0)
        tumor_voxels = int((label > 0).sum().item())
        if tumor_voxels > 0:
            sized.append((tumor_voxels, case, sample))

    sized.sort(key=lambda x: x[0])
    if len(sized) < args.n_cases:
        chosen = sized
    else:
        # evenly spaced across the size-sorted list -> small, medium, large examples
        idxs = np.linspace(0, len(sized) - 1, args.n_cases).round().astype(int)
        chosen = [sized[i] for i in sorted(set(idxs))]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for tumor_voxels, case, sample in chosen:
        image = to_plain_tensor(sample["image"]).numpy().astype(np.float16)  # (4, D, H, W)
        label = to_plain_tensor(sample["label"]).squeeze(0).numpy().astype(np.uint8)  # (D, H, W)
        out_path = out_dir / f"{case['case_id']}.npz"
        np.savez_compressed(out_path, image=image, label=label, case_id=case["case_id"])
        print(f"saved {out_path} (tumor voxels: {tumor_voxels})")

    print(f"\nExported {len(chosen)} example cases to {out_dir}/")


if __name__ == "__main__":
    main()
