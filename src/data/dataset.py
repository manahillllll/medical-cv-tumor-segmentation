"""
BraTS-style dataset loading and patch sampling, built on MONAI's data utilities.

Per the project scope, medical-image-specific I/O and augmentation (NIfTI loading,
resampling, intensity normalization) is fiddly, well-solved plumbing — we use MONAI's
transforms for that rather than reimplementing it. The interesting from-scratch pieces
(model, loss, stitching, uncertainty, Grad-CAM) live in the sibling modules.

Expected directory layout (matches both the official BraTS release and MONAI's
auto-downloadable Task01_BrainTumour / Medical Segmentation Decathlon mirror — see
scripts/download_data.md):

    data_dir/
        case_001/
            case_001_t1.nii.gz
            case_001_t1ce.nii.gz
            case_001_t2.nii.gz
            case_001_flair.nii.gz
            case_001_seg.nii.gz
        case_002/
            ...

BraTS segmentation labels (as shipped): 0=background, 1=necrotic/non-enhancing core,
2=edema, 4=enhancing tumor. We remap {0,1,2,4} -> {0,1,2,3} so the label map is dense
and usable directly as a class index by nn.CrossEntropyLoss / one_hot.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import torch

try:
    from monai.data import CacheDataset, Dataset
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        LoadImaged,
        NormalizeIntensityd,
        Orientationd,
        RandCropByPosNegLabeld,
        RandFlipd,
        RandGaussianNoised,
        RandScaleIntensityd,
        RandShiftIntensityd,
        Spacingd,
        Lambdad,
        EnsureTyped,
        CenterSpatialCropd,
        SpatialPadd,
    )
    _MONAI_AVAILABLE = True
except ImportError:  # allows shape/logic to be inspected without MONAI installed
    _MONAI_AVAILABLE = False

MODALITIES = ("t1", "t1ce", "t2", "flair")
BRATS_LABEL_MAP = {0: 0, 1: 1, 2: 2, 4: 3}
BRATS_CLASS_NAMES = ("background", "necrotic_core", "edema", "enhancing_tumor")


def _require_monai():
    if not _MONAI_AVAILABLE:
        raise ImportError(
            "MONAI is required for data loading. Install with `pip install monai` "
            "(see requirements.txt)."
        )


def remap_brats_labels(label: np.ndarray) -> np.ndarray:
    out = np.zeros_like(label)
    for src, dst in BRATS_LABEL_MAP.items():
        out[label == src] = dst
    return out


@dataclass
class CaseFiles:
    case_id: str
    images: list[str]
    label: str | None


def discover_cases(data_dir: str, require_label: bool = True) -> list[CaseFiles]:
    """Scan data_dir for case subfolders following the *_t1/t1ce/t2/flair/seg.nii.gz convention."""
    cases = []
    for case_dir in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if not os.path.isdir(case_dir):
            continue
        case_id = os.path.basename(case_dir)
        images = []
        missing = False
        for mod in MODALITIES:
            matches = glob.glob(os.path.join(case_dir, f"*{mod}.nii.gz")) + \
                      glob.glob(os.path.join(case_dir, f"*{mod}.nii"))
            if not matches:
                missing = True
                break
            images.append(matches[0])
        if missing:
            continue

        label_matches = glob.glob(os.path.join(case_dir, "*seg.nii.gz")) + \
                         glob.glob(os.path.join(case_dir, "*seg.nii"))
        label = label_matches[0] if label_matches else None
        if require_label and label is None:
            continue

        cases.append(CaseFiles(case_id=case_id, images=images, label=label))
    return cases


def _to_data_dicts(cases: list[CaseFiles]) -> list[dict]:
    return [{"image": c.images, "label": c.label, "case_id": c.case_id} for c in cases]


def get_train_transforms(patch_size: tuple[int, int, int] = (128, 128, 128)):
    _require_monai()
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"]),
        Lambdad(keys=["label"], func=lambda x: remap_brats_labels(np.asarray(x))[None, ...]
                 if np.asarray(x).ndim == 3 else remap_brats_labels(np.asarray(x))),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=2,  # oversample tumor-containing patches ~2:1 over background-only patches
            neg=1,
            num_samples=2,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.15, std=0.01),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms():
    """No cropping/augmentation — full volumes for sliding-window inference."""
    _require_monai()
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"]),
        Lambdad(keys=["label"], func=lambda x: remap_brats_labels(np.asarray(x))[None, ...]
                 if np.asarray(x).ndim == 3 else remap_brats_labels(np.asarray(x))),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_classification_val_transforms(patch_size: tuple[int, int, int] = (128, 128, 128)):
    """
    Deterministic, fixed-size volume for classifier validation: a center crop (padded
    if the scan is smaller than patch_size) rather than a random patch or a full
    variable-size volume — keeps eval batches a consistent shape without needing
    sliding-window inference just to check classification accuracy during training.
    """
    _require_monai()
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"]),
        Lambdad(keys=["label"], func=lambda x: remap_brats_labels(np.asarray(x))[None, ...]
                 if np.asarray(x).ndim == 3 else remap_brats_labels(np.asarray(x))),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        CenterSpatialCropd(keys=["image", "label"], roi_size=patch_size),
        EnsureTyped(keys=["image", "label"]),
    ])


def build_datasets(data_dir: str, patch_size: tuple[int, int, int] = (128, 128, 128),
                    val_fraction: float = 0.15, seed: int = 42, cache: bool = False):
    """Split discovered cases into train/val and wrap in MONAI Datasets."""
    _require_monai()
    cases = discover_cases(data_dir)
    if not cases:
        raise FileNotFoundError(
            f"No BraTS-style cases found under {data_dir}. See scripts/download_data.md."
        )

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(cases))
    n_val = max(1, int(len(cases) * val_fraction))
    val_idx, train_idx = set(indices[:n_val]), set(indices[n_val:])

    train_cases = [cases[i] for i in sorted(train_idx)]
    val_cases = [cases[i] for i in sorted(val_idx)]

    train_dicts = _to_data_dicts(train_cases)
    val_dicts = _to_data_dicts(val_cases)

    ds_cls = CacheDataset if cache else Dataset
    train_ds = ds_cls(data=train_dicts, transform=get_train_transforms(patch_size))
    val_ds = ds_cls(data=val_dicts, transform=get_val_transforms())
    return train_ds, val_ds
