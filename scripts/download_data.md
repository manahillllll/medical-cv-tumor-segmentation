# Getting BraTS data

Three practical routes, in order of how much setup they need. `config.yaml`'s
`data.format` field tells the loader which layout to expect (`"h5_slices"` or
`"nifti"`).

## Is the Kaggle h5-slices download (Option A) actually complete?

Short answer: yes, for segmentation. The official BraTS2020 training release has
**369 patient cases** ([CBICA data page](https://www.med.upenn.edu/cbica/brats2020/data.html)),
and the Kaggle repackaging has exactly 369 `volume_*` cases too — same case count,
same four modalities + segmentation per case. It's not missing patients.

What it genuinely doesn't have:
- **Tumor grade (HGG/LGG) labels.** Official BraTS2020 has 293 HGG / 76 LGG cases,
  but that label lives in `name_mapping.csv`, which ships with the official download,
  not with this Kaggle repackaging — see Option C below if you need it for the
  classifier.
- **Raw NIfTI volumes.** It's repackaged as 155 per-slice `.h5` files per case instead
  of 4 NIfTI files + 1 segmentation NIfTI. `LoadBraTSH5Volumed` reconstructs the 3D
  volume from those slices at load time, so this doesn't cost you anything
  functionally, but it does mean the exact preprocessing the Kaggle uploader applied
  (values are float, already somewhat intensity-adjusted) isn't independently
  verified against the official release.
- **The 125-case validation set** (unlabeled, used for the actual challenge
  leaderboard) — not needed for your own train/val split, which this repo does out of
  the 369 training cases regardless.

If segmentation is your main goal, Option A (what you already have) is fine as-is.
If you want the classifier trained on real grade labels, get the official download
(Option C) — matching Kaggle's arbitrary `volume_N` numbering back to official
`BraTS20_Training_XXX` patient IDs isn't independently confirmed anywhere, so the
reliable path for labels is switching to the official raw NIfTI data directly
(`format: "nifti"`) rather than trying to graft official labels onto the Kaggle case IDs.

## Option A — Kaggle "BraTS2020 Training Data" (h5 slices) — what this repo is configured for by default

[kaggle.com/datasets/awsaf49/brats20-dataset-training-validation](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)
repackages the official BraTS2020 training set (369 cases) as one `.h5` file per
2D slice (155 slices/case) instead of one NIfTI volume per modality:

```
data/brats2020_h5/BraTS2020_training_data/content/data/
    volume_1_slice_0.h5
    volume_1_slice_1.h5
    ...
    volume_369_slice_154.h5
```

Each `.h5` file has two datasets: `"image"` (240, 240, 4) and `"mask"` (240, 240, 3).
`src/data/dataset.py::LoadBraTSH5Volumed` reconstructs the full 3D volume from the
155 per-case slices at load time (no separate conversion step needed) and figures out
channel semantics as follows — **note these orders aren't documented upstream**, they
were determined empirically (see the docstring on `LoadBraTSH5Volumed`) by checking
which image channel shows the strongest T1ce-style enhancing/necrotic contrast and
which shows T2-style bright CSF vs. FLAIR-style suppressed CSF:

- `image` channels: `[FLAIR, T1, T1ce, T2]`
- `mask` channels: `[necrotic/non-enhancing core, edema, enhancing tumor]` (one-hot;
  remapped internally to the same dense `{0,1,2,3}` class scheme as the NIfTI path)

Download via the Kaggle CLI or web UI, unzip, and point `config.yaml`'s
`data.data_dir` at the `.../content/data` folder shown above (already the default).

**This format does not include tumor grade/subtype (HGG/LGG) labels** — its metadata
CSV only has per-slice pixel counts and a "has tumor" flag, so `scripts/train_classifier.py`
needs a labels CSV from elsewhere (see Option C) if you want to train the
classification head against this download.

## Option B — MONAI auto-download (no registration, gives `format: "nifti"`)

The Medical Segmentation Decathlon's **Task01_BrainTumour** is BraTS-derived data
(same 4 MRI modalities, same tumor-subregion labeling scheme) and MONAI can fetch it
directly:

```python
from monai.apps import DecathlonDataset

DecathlonDataset(root_dir="data/", task="Task01_BrainTumour", section="training", download=True)
```

Task01_BrainTumour ships each case as a single 4-channel NIfTI
(`imagesTr/BRATS_XXX.nii.gz`, channel order FLAIR/T1w/T1gd/T2w) rather than four
separate files — split each 4D volume into four per-modality files to match the
`format: "nifti"` layout documented in `src/data/dataset.py`:

```
data/brats/<case_id>/<case_id>_t1.nii.gz
data/brats/<case_id>/<case_id>_t1ce.nii.gz
data/brats/<case_id>/<case_id>_t2.nii.gz
data/brats/<case_id>/<case_id>_flair.nii.gz
data/brats/<case_id>/<case_id>_seg.nii.gz
```

Then set `data.format: "nifti"` and `data.data_dir: "data/brats"` in `config.yaml`.

## Option C — Official BraTS release (needed for grade/subtype classification labels)

Neither of the above ships tumor grade labels. Get those from the official challenge
via **[synapse.org/brats](https://synapse.org/brats)** — this is the current, single
access point for all BraTS challenge years. CBICA's old Image Processing Portal
(`ipp.cbica.upenn.edu`), which used to host the BraTS2020-specific download and is
what earlier revisions of this doc pointed to, is now **deprecated** and redirects
to Synapse instead.

1. Create a free [Synapse](https://www.synapse.org) account, then go to
   [synapse.org/brats](https://synapse.org/brats) and follow its instructions to
   request access to the BraTS project/team for the year you want (BraTS2020 for the
   HGG/LGG grade split used elsewhere in this doc, or a later year — Synapse access is
   generally gated behind accepting a data use agreement, and may involve an approval
   step; exact clicks change over time, so follow what the page shows you).
2. Once access is granted, download the training set through Synapse's web UI or the
   `synapseclient` Python package. You should get the same per-case structure as
   before: a folder with T1/T1ce/T2/FLAIR + segmentation NIfTI files
   (`format: "nifti"` in this repo's terms), plus a `name_mapping.csv` (grade:
   BraTS2020 has 293 HGG / 76 LGG cases) and `survival_info.csv` (survival-time
   labels — a second possible classification task).
3. Point `config.yaml` at this download with `data.format: "nifti"` and
   `data.data_dir` set to wherever you extracted it.
4. Build `data/labels.csv` with columns `case_id,label` using the folder names from
   this download (e.g. `BraTS20_Training_001,1` for HGG) — read the grade straight out
   of `name_mapping.csv`.
5. Pass it to `scripts/train_classifier.py --labels_csv data/labels.csv`.

### If you're using the BraTS 2023 Challenge data specifically (Synapse syn51156910)

BraTS 2023 bundles 9 sub-challenges into one release; the task equivalent to BraTS2020
(adult glioma segmentation, same 4 modalities) is **Task 1: Segmentation – Adult
Glioma**. Two things to verify once you have access:

- **Filenames may differ from BraTS2020's `_t1`/`_t1ce`/`_t2`/`_flair`/`_seg`
  convention** — newer BraTS releases have used different modality suffixes. Check one
  downloaded case's filenames against `MODALITIES` in `src/data/dataset.py` and adjust
  if they don't match; `discover_cases()`'s glob patterns are the only thing that'd
  need to change.
- Whether this release still ships an HGG/LGG-style grade label the way BraTS2020's
  `name_mapping.csv` did is not confirmed — check the Task 1 data description on
  Synapse before assuming `data/labels.csv` can be built the same way.

**License: this data is CC-BY-NC 4.0 (non-commercial use only).** If you train on it,
any resulting writeup/publication must include the attribution statement Synapse
requires:

> Data used in this publication were obtained as part of the Brain Tumor Segmentation
> (BraTS) Challenge project through Synapse ID: syn51156910.

...plus citations to the BraTS flagship and challenge-specific manuscripts listed on
Synapse's Data Access/Downloads page. See the full terms at
[creativecommons.org/licenses/by-nc/4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Sanity-checking the download

```bash
# format: "h5_slices"
python -c "from src.data.dataset import discover_h5_cases; print(len(discover_h5_cases('data/brats2020_h5/BraTS2020_training_data/content/data')))"

# format: "nifti"
python -c "from src.data.dataset import discover_cases; print(len(discover_cases('data/brats')))"
```

Either should report the number of complete cases found.

## No GPU available yet?

`config_cpu_smoketest.yaml` runs the exact same pipeline at a much smaller scale
(64³ patches, a narrower U-Net, `overfit_single_batch: true`) so you can confirm
everything — real data loading included — works correctly on a CPU-only machine in
a couple of minutes, before renting/using GPU compute for a real training run:

```bash
python scripts/train_segmentation.py --config config_cpu_smoketest.yaml
```
