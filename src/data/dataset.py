"""
BraTS-style dataset loading and patch sampling, built on MONAI's data utilities.

Per the project scope, medical-image-specific I/O and augmentation (NIfTI loading,
resampling, intensity normalization) is fiddly, well-solved plumbing — we use MONAI's
transforms for that rather than reimplementing it. The interesting from-scratch pieces
(model, loss, stitching, uncertainty, Grad-CAM) live in the sibling modules.

Two source layouts are supported:

1. ``format="nifti"`` — one folder per case, four modality NIfTI files + a
   segmentation NIfTI file (matches the official BraTS release and MONAI's
   auto-downloadable Task01_BrainTumour mirror — see scripts/download_data.md):

    data_dir/
        case_001/
            case_001_t1.nii.gz
            case_001_t1ce.nii.gz
            case_001_t2.nii.gz
            case_001_flair.nii.gz
            case_001_seg.nii.gz
        case_002/
            ...

2. ``format="h5_slices"`` — the Kaggle "BraTS2020 Training Data" repackaging, where
   each case is split into 155 per-slice ``.h5`` files (keys "image" (H,W,4),
   "mask" (H,W,3)) instead of one 3D NIfTI volume per modality:

    data_dir/
        volume_1_slice_0.h5
        volume_1_slice_1.h5
        ...
        volume_369_slice_154.h5

   Image channel order in this format is fixed as [FLAIR, T1, T1ce, T2] (verified
   empirically — see LoadBraTSH5Volumed docstring). Mask channel order is
   [necrotic/non-enhancing core, edema, enhancing tumor], i.e. channel i maps to our
   dense class i+1.

BraTS segmentation labels as shipped in the NIfTI format: 0=background,
1=necrotic/non-enhancing core, 2=edema, 4=enhancing tumor. We remap {0,1,2,4} ->
{0,1,2,3} so the label map is dense and usable directly as a class index by
nn.CrossEntropyLoss / one_hot. The h5_slices format's one-hot masks are converted
straight to this same {0,1,2,3} dense scheme.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import torch

try:
    from monai.config import KeysCollection
    from monai.data import CacheDataset, Dataset
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        LoadImaged,
        MapTransform,
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

try:
    import h5py
    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False

MODALITIES = ("t1", "t1ce", "t2", "flair")
BRATS_LABEL_MAP = {0: 0, 1: 1, 2: 2, 4: 3}
BRATS_CLASS_NAMES = ("background", "necrotic_core", "edema", "enhancing_tumor")

H5_SLICE_PATTERN = re.compile(r"volume_(\d+)_slice_(\d+)\.h5$")
H5_MODALITY_ORDER = ("flair", "t1", "t1ce", "t2")  # channel order of the "image" dataset


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


def discover_h5_cases(data_dir: str) -> list[dict]:
    """
    Scan data_dir for the Kaggle-style ``volume_<id>_slice_<n>.h5`` layout and group
    files by volume id, sorted by slice index. Every case in this dataset has a
    segmentation mask (it's baked into each slice file), so there's no require_label
    flag here the way there is for discover_cases.
    """
    per_volume: dict[int, list[tuple[int, str]]] = {}
    for fp in glob.glob(os.path.join(data_dir, "*.h5")):
        m = H5_SLICE_PATTERN.search(os.path.basename(fp))
        if not m:
            continue
        vol_id, slice_id = int(m.group(1)), int(m.group(2))
        per_volume.setdefault(vol_id, []).append((slice_id, fp))

    cases = []
    for vol_id in sorted(per_volume):
        slices = sorted(per_volume[vol_id], key=lambda x: x[0])
        cases.append({
            "case_id": f"volume_{vol_id}",
            "slice_files": [fp for _, fp in slices],
        })
    return cases


class LoadBraTSH5Volumed(MapTransform if _MONAI_AVAILABLE else object):
    """
    Reconstructs a full 3D case from the Kaggle BraTS2020 per-slice .h5 format.

    Each slice file holds "image" (H, W, 4) and "mask" (H, W, 3) one-hot arrays.
    Channel order was determined empirically (not documented upstream) by comparing
    intensities inside the enhancing-tumor vs. necrotic-core masks (T1ce shows by far
    the strongest enhancing/necrotic contrast) and inside a CSF proxy region vs. the
    edema mask (T2's CSF is bright, FLAIR's is suppressed/dark):
        image channel 0 = FLAIR, 1 = T1, 2 = T1ce, 3 = T2
        mask channel   0 = necrotic/non-enhancing core, 1 = edema, 2 = enhancing tumor

    Reads the dict's "slice_files" key (list of paths, already sorted by slice index
    by discover_h5_cases) and writes "image" -> (4, D, H, W) float32 and
    "label" -> (1, D, H, W) int64, using the same dense {0,1,2,3} class scheme as the
    NIfTI-format loader (see remap_brats_labels).
    """

    def __init__(self, keys: "KeysCollection" = ("slice_files",), image_key: str = "image",
                 label_key: str = "label", allow_missing_keys: bool = False):
        if _MONAI_AVAILABLE:
            super().__init__(keys, allow_missing_keys)
        self.slice_files_key = list(keys)[0] if not isinstance(keys, str) else keys
        self.image_key = image_key
        self.label_key = label_key

    def __call__(self, data: dict) -> dict:
        if not _H5PY_AVAILABLE:
            raise ImportError("h5py is required to load the h5_slices BraTS format (pip install h5py).")

        d = dict(data)
        slice_files = d[self.slice_files_key]

        images, masks = [], []
        for fp in slice_files:
            with h5py.File(fp, "r") as f:
                images.append(np.asarray(f["image"], dtype=np.float32))  # (H, W, 4)
                masks.append(np.asarray(f["mask"], dtype=np.uint8))       # (H, W, 3)

        image_vol = np.stack(images, axis=0)               # (D, H, W, 4)
        image_vol = np.transpose(image_vol, (3, 0, 1, 2))  # (4, D, H, W)

        mask_vol = np.stack(masks, axis=0)                 # (D, H, W, 3)
        label_vol = np.zeros(mask_vol.shape[:3], dtype=np.int64)
        label_vol[mask_vol[..., 0] > 0] = 1  # necrotic / non-enhancing core
        label_vol[mask_vol[..., 1] > 0] = 2  # edema
        label_vol[mask_vol[..., 2] > 0] = 3  # enhancing tumor (last -> wins any overlap)

        d[self.image_key] = torch.from_numpy(image_vol)
        d[self.label_key] = torch.from_numpy(label_vol)[None, ...]  # (1, D, H, W)
        return d


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


def get_h5_train_transforms(patch_size: tuple[int, int, int] = (128, 128, 128)):
    """
    Same augmentation recipe as get_train_transforms, minus Orientationd/Spacingd:
    this data source is already uniform (240x240x155, 1mm isotropic, co-registered)
    straight out of the official BraTS preprocessing, so there's no heterogeneous
    orientation/spacing left to normalize away.
    """
    _require_monai()
    return Compose([
        LoadBraTSH5Volumed(keys=["slice_files"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=2,
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


def get_h5_val_transforms():
    _require_monai()
    return Compose([
        LoadBraTSH5Volumed(keys=["slice_files"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_h5_classification_val_transforms(patch_size: tuple[int, int, int] = (128, 128, 128)):
    _require_monai()
    return Compose([
        LoadBraTSH5Volumed(keys=["slice_files"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        CenterSpatialCropd(keys=["image", "label"], roi_size=patch_size),
        EnsureTyped(keys=["image", "label"]),
    ])


def _split_train_val(items: list, val_fraction: float, seed: int) -> tuple[list, list]:
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(items))
    n_val = max(1, int(len(items) * val_fraction))
    val_idx, train_idx = set(indices[:n_val].tolist()), set(indices[n_val:].tolist())
    train_items = [items[i] for i in sorted(train_idx)]
    val_items = [items[i] for i in sorted(val_idx)]
    return train_items, val_items


def build_datasets(data_dir: str, patch_size: tuple[int, int, int] = (128, 128, 128),
                    val_fraction: float = 0.15, seed: int = 42, cache: bool = False,
                    format: str = "nifti"):
    """
    Split discovered cases into train/val and wrap in MONAI Datasets.

    format="nifti": per-case folders of NIfTI files (official BraTS / Decathlon layout).
    format="h5_slices": Kaggle's per-slice .h5 repackaging (see module docstring).
    """
    _require_monai()
    ds_cls = CacheDataset if cache else Dataset

    if format == "h5_slices":
        cases = discover_h5_cases(data_dir)
        if not cases:
            raise FileNotFoundError(
                f"No volume_<id>_slice_<n>.h5 files found under {data_dir}. "
                "See scripts/download_data.md."
            )
        train_cases, val_cases = _split_train_val(cases, val_fraction, seed)
        train_ds = ds_cls(data=train_cases, transform=get_h5_train_transforms(patch_size))
        val_ds = ds_cls(data=val_cases, transform=get_h5_val_transforms())
        return train_ds, val_ds

    if format != "nifti":
        raise ValueError(f"Unknown data format '{format}', expected 'nifti' or 'h5_slices'.")

    cases = discover_cases(data_dir)
    if not cases:
        raise FileNotFoundError(
            f"No BraTS-style cases found under {data_dir}. See scripts/download_data.md."
        )
    train_cases, val_cases = _split_train_val(cases, val_fraction, seed)
    train_dicts = _to_data_dicts(train_cases)
    val_dicts = _to_data_dicts(val_cases)

    train_ds = ds_cls(data=train_dicts, transform=get_train_transforms(patch_size))
    val_ds = ds_cls(data=val_dicts, transform=get_val_transforms())
    return train_ds, val_ds
