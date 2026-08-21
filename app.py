"""
Streamlit demo for the 3D tumor segmentation + uncertainty pipeline.

Run locally:
    streamlit run app.py

Two ways to get a scan in:
  - Example case: a handful of held-out BraTS2020 validation cases, pre-exported by
    scripts/export_demo_cases.py into app_data/example_cases/ (not committed to the
    repo -- run that script yourself first; see README).
  - Upload your own scan: four NIfTI files (FLAIR, T1, T1ce, T2). Must be in that exact
    channel order internally to match how the model was trained -- see the note in
    _load_uploaded_case below before touching the channel ordering.

No trained classifier ships with this project (see README's scope note: the BraTS2020
download used has no real tumor grade label, and this project doesn't train on
invented ones), so this demo shows segmentation + voxel-level uncertainty only.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.dataset import get_val_transforms
from src.metrics import brats_region_dice
from src.models.unet3d import UNet3D
from src.report import ReportResult, generate_report
from src.utils import load_checkpoint, overlay_mask_on_slice, to_plain_tensor

EXAMPLE_DIR = Path("app_data/example_cases")
CONFIG_PATH = "config.yaml"
CLASS_NAMES = ("background", "necrotic core", "edema", "enhancing tumor")


@st.cache_resource
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_model(checkpoint_path: str) -> tuple[torch.nn.Module, str]:
    cfg = load_config()
    mcfg = cfg["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet3D(
        in_channels=mcfg["in_channels"], num_classes=mcfg["num_seg_classes"],
        base_filters=mcfg["base_filters"], depth=mcfg["depth"], dropout_p=mcfg["dropout_p"],
    )
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.to(device).eval()
    return model, device


def list_example_cases() -> list[Path]:
    if not EXAMPLE_DIR.exists():
        return []
    return sorted(EXAMPLE_DIR.glob("*.npz"))


def load_example_case(path: Path) -> tuple[torch.Tensor, torch.Tensor | None, str]:
    data = np.load(path)
    image = torch.from_numpy(data["image"].astype(np.float32))  # (4, D, H, W)
    label = torch.from_numpy(data["label"].astype(np.int64))    # (D, H, W)
    case_id = str(data["case_id"])
    return image, label, case_id


def _load_uploaded_case(flair_file, t1_file, t1ce_file, t2_file) -> torch.Tensor:
    """
    IMPORTANT: the trained checkpoint was trained exclusively on the Kaggle h5_slices
    data source, whose channel order is fixed as (FLAIR, T1, T1ce, T2) -- see
    src/data/dataset.py's H5_MODALITY_ORDER / LoadBraTSH5Volumed docstring. Uploaded
    files MUST be assembled in that same order, or the model sees scrambled channels
    relative to what it learned and produces meaningless output. Do not reorder this
    to match MODALITIES ("t1","t1ce","t2","flair") -- that ordering is for the
    from-scratch NIfTI training path, which this checkpoint was never trained on.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        paths = []
        for name, uploaded in zip(("flair", "t1", "t1ce", "t2"),
                                   (flair_file, t1_file, t1ce_file, t2_file)):
            p = tmpdir / f"{name}.nii.gz"
            p.write_bytes(uploaded.getvalue())
            paths.append(str(p))

        sample = get_val_transforms()({"image": paths, "label": paths[0]})
        return to_plain_tensor(sample["image"]).float()


def run_inference(volume: torch.Tensor, mc_samples: int) -> ReportResult:
    cfg = load_config()
    mcfg, icfg = cfg["model"], cfg["inference"]
    model, device = load_model("checkpoints/segmentation/best.pt")

    return generate_report(
        volume=volume,
        seg_model=model,
        seg_classifier=None,
        patch_size=tuple(icfg["patch_size"]),
        overlap=icfg["overlap"],
        num_seg_classes=mcfg["num_seg_classes"],
        mc_samples=mc_samples,
        device=device,
    )


def render_slice(volume: np.ndarray, seg_pred: np.ndarray, uncertainty: np.ndarray,
                  slice_idx: int, modality_idx: int, ground_truth: np.ndarray | None) -> None:
    base = volume[modality_idx, slice_idx]
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)

    cols = st.columns(4 if ground_truth is not None else 3)

    with cols[0]:
        st.markdown("**Input scan**")
        st.image(base, clamp=True, use_container_width=True)

    with cols[1]:
        st.markdown("**Model prediction**")
        overlay = overlay_mask_on_slice(base, seg_pred[slice_idx])
        st.image(overlay, clamp=True, use_container_width=True)

    col_offset = 2
    if ground_truth is not None:
        with cols[2]:
            st.markdown("**Ground truth**")
            gt_overlay = overlay_mask_on_slice(base, ground_truth[slice_idx])
            st.image(gt_overlay, clamp=True, use_container_width=True)
        col_offset = 3

    with cols[col_offset]:
        st.markdown("**Uncertainty**")
        u_slice = uncertainty[slice_idx]
        u_max = uncertainty.max()
        u_norm = np.clip(u_slice / u_max, 0, 1) if u_max > 0 else np.zeros_like(u_slice)
        alpha = 0.8 * u_norm  # transparent where uncertainty is ~0, so only genuinely
        magenta = np.array([1.0, 0.15, 0.85])  # uncertain regions get tinted
        gray3 = np.stack([base] * 3, axis=-1)
        blended = gray3 * (1 - alpha[..., None]) + magenta * alpha[..., None]
        st.image(np.clip(blended, 0, 1), clamp=True, use_container_width=True)


def main():
    st.set_page_config(page_title="3D Tumor Segmentation Demo", layout="wide")
    st.title("3D Brain Tumor Segmentation + Uncertainty")
    st.caption(
        "3D U-Net (built from scratch) trained on real BraTS2020 MRI data, with "
        "Monte Carlo Dropout uncertainty quantification. See the "
        "[GitHub repo](https://github.com/manahillllll/medical-cv-tumor-segmentation) "
        "for full details, training results, and code."
    )

    with st.expander("About this demo / project scope", expanded=False):
        st.markdown(
            "- **Segmentation model**: trained to convergence (200 epochs) on the real "
            "369-case BraTS2020 dataset. Held-out Dice: Whole Tumor 0.901, Tumor Core "
            "0.876, Enhancing Tumor 0.761.\n"
            "- **Uncertainty map**: Monte Carlo Dropout (20 stochastic forward passes) "
            "over a tumor-centered patch — brighter = the model is less confident, "
            "which should concentrate near tumor boundaries.\n"
            "- **No classification/Grad-CAM here**: the original design also includes "
            "a malignancy/subtype classifier with a confidence interval and a Grad-CAM "
            "explanation. That code is implemented and unit-tested, but **not trained** "
            "— the BraTS2020 source used here has no real tumor grade label, and this "
            "project doesn't train on invented ones. Segmentation + uncertainty is the "
            "real, trained result shown below."
        )

    checkpoint_path = Path("checkpoints/segmentation/best.pt")
    if not checkpoint_path.exists():
        st.error(
            f"No checkpoint found at `{checkpoint_path}`. Train the model first: "
            "`python scripts/train_segmentation.py --config config.yaml`."
        )
        return

    st.sidebar.header("Input")
    mode = st.sidebar.radio("Scan source", ["Example case", "Upload your own scan"])

    volume = label = case_id = None

    if mode == "Example case":
        examples = list_example_cases()
        if not examples:
            st.sidebar.warning(
                "No example cases found. Run "
                "`python scripts/export_demo_cases.py --config config.yaml` first "
                "(requires the BraTS2020 dataset locally — see scripts/download_data.md)."
            )
        else:
            choice = st.sidebar.selectbox(
                "Pick a case", examples, format_func=lambda p: p.stem
            )
            volume, label, case_id = load_example_case(choice)
    else:
        st.sidebar.caption("Upload all 4 modalities as NIfTI files (.nii or .nii.gz).")
        flair_file = st.sidebar.file_uploader("FLAIR", type=["nii", "nii.gz"], key="flair")
        t1_file = st.sidebar.file_uploader("T1", type=["nii", "nii.gz"], key="t1")
        t1ce_file = st.sidebar.file_uploader("T1ce", type=["nii", "nii.gz"], key="t1ce")
        t2_file = st.sidebar.file_uploader("T2", type=["nii", "nii.gz"], key="t2")
        if flair_file and t1_file and t1ce_file and t2_file:
            with st.spinner("Loading and preprocessing scan..."):
                volume = _load_uploaded_case(flair_file, t1_file, t1ce_file, t2_file)
            case_id = "uploaded scan"

    mc_samples = st.sidebar.slider("MC-Dropout samples", min_value=5, max_value=30, value=20,
                                    help="More samples = smoother uncertainty estimate, slower to compute.")

    if volume is None:
        st.info("Pick an example case or upload a scan from the sidebar to begin.")
        return

    run_key = f"result::{case_id}::{mc_samples}"
    if st.sidebar.button("Run inference", type="primary", key="run_btn"):
        with st.spinner(f"Running sliding-window inference + {mc_samples}-sample MC-Dropout..."):
            result = run_inference(volume, mc_samples)
        st.session_state[run_key] = (result, volume, label, case_id)

    if run_key not in st.session_state:
        st.info("Click **Run inference** in the sidebar to segment this scan.")
        return

    result, volume, label, case_id = st.session_state[run_key]
    volume_np = volume.numpy()
    seg_np = result.segmentation.numpy()
    uncertainty_np = result.uncertainty_map.numpy()
    label_np = label.numpy() if label is not None else None

    st.subheader(f"Case: {case_id}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Whole Tumor volume", f"{result.tumor_volumes_cm3['whole_tumor']:.1f} cm³")
    m2.metric("Tumor Core volume", f"{result.tumor_volumes_cm3['tumor_core']:.1f} cm³")
    m3.metric("Enhancing Tumor volume", f"{result.tumor_volumes_cm3['enhancing_tumor']:.1f} cm³")

    if label_np is not None:
        regions = brats_region_dice(result.segmentation, label)
        d1, d2, d3 = st.columns(3)
        d1.metric("Dice (WT)", f"{regions['whole_tumor']:.3f}")
        d2.metric("Dice (TC)", f"{regions['tumor_core']:.3f}")
        d3.metric("Dice (ET)", f"{regions['enhancing_tumor']:.3f}")

    modality_names = ["FLAIR", "T1", "T1ce", "T2"]
    modality_idx = st.selectbox("Background modality", range(4), format_func=lambda i: modality_names[i])

    tumor_slices = np.where((seg_np > 0).any(axis=(1, 2)))[0]
    default_slice = int(tumor_slices.mean()) if len(tumor_slices) else volume_np.shape[1] // 2
    slice_idx = st.slider("Slice", 0, volume_np.shape[1] - 1, default_slice)

    render_slice(volume_np, seg_np, uncertainty_np, slice_idx, modality_idx, label_np)

    st.caption(
        "Segmentation colors: red = necrotic core, blue = edema, yellow = enhancing "
        "tumor. Uncertainty: brighter magenta = higher epistemic uncertainty."
    )


if __name__ == "__main__":
    main()
