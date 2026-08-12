"""
Regression test for the h5_slices data pipeline, using small synthetic .h5 files
(no real BraTS download needed) so it still runs everywhere pytest does.

This specifically reproduces a real bug found while running the pipeline against
real Kaggle BraTS2020 data: MONAI's MetaTensor attaches per-sample metadata during
LoadBraTSH5Volumed/EnsureTyped, and that metadata doesn't collate cleanly across a
batch of randomly-cropped patches (RandCropByPosNegLabeld) — computing the loss on
the raw batched MetaTensor raised a collate error inside DiceCELoss. The fix
(src/utils.py::to_plain_tensor) strips MetaTensor down to a plain tensor right after
pulling a batch off the DataLoader; this test fails again if that regresses.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from monai.data import Dataset, list_data_collate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import discover_h5_cases, get_h5_train_transforms, get_h5_val_transforms
from src.losses import DiceCELoss
from src.models.unet3d import UNet3D
from src.utils import to_plain_tensor


def _write_synthetic_h5_case(root: Path, volume_id: int, num_slices: int = 20, size: int = 32):
    rng = np.random.RandomState(volume_id)
    for s in range(num_slices):
        image = rng.randn(size, size, 4).astype(np.float64)
        mask = np.zeros((size, size, 3), dtype=np.uint8)
        if num_slices // 4 < s < 3 * num_slices // 4:
            mask[size // 4: size // 2, size // 4: size // 2, 0] = 1  # necrotic
            mask[size // 2: 3 * size // 4, size // 4: 3 * size // 4, 1] = 1  # edema
            mask[size // 4: size // 2, size // 2: 3 * size // 4, 2] = 1  # enhancing
        with h5py.File(root / f"volume_{volume_id}_slice_{s}.h5", "w") as f:
            f.create_dataset("image", data=image)
            f.create_dataset("mask", data=mask)


@pytest.fixture
def synthetic_h5_dir(tmp_path):
    _write_synthetic_h5_case(tmp_path, volume_id=1, num_slices=20, size=32)
    _write_synthetic_h5_case(tmp_path, volume_id=2, num_slices=20, size=32)
    return tmp_path


def test_discover_h5_cases(synthetic_h5_dir):
    cases = discover_h5_cases(str(synthetic_h5_dir))
    assert len(cases) == 2
    assert cases[0]["case_id"] == "volume_1"
    assert len(cases[0]["slice_files"]) == 20


def test_h5_val_transform_reconstructs_volume(synthetic_h5_dir):
    cases = discover_h5_cases(str(synthetic_h5_dir))
    sample = get_h5_val_transforms()(cases[0])
    assert sample["image"].shape == (4, 20, 32, 32)
    assert sample["label"].shape == (1, 20, 32, 32)
    assert set(torch.unique(to_plain_tensor(sample["label"])).tolist()) <= {0, 1, 2, 3}


def test_h5_train_pipeline_batched_loss_does_not_raise(synthetic_h5_dir):
    """The exact regression: load -> crop -> collate -> loss on a real batch."""
    cases = discover_h5_cases(str(synthetic_h5_dir))
    transform = get_h5_train_transforms(patch_size=(16, 16, 16))
    ds = Dataset(data=cases, transform=transform)

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=2, collate_fn=list_data_collate)
    batch = next(iter(loader))

    images = to_plain_tensor(batch["image"])
    labels = to_plain_tensor(batch["label"]).squeeze(1)
    assert isinstance(images, torch.Tensor) and not hasattr(images, "meta")

    model = UNet3D(in_channels=4, num_classes=4, base_filters=4, depth=2)
    criterion = DiceCELoss(num_classes=4)

    logits = model(images)
    loss = criterion(logits, labels)  # this line raised before the to_plain_tensor fix
    assert torch.isfinite(loss)
