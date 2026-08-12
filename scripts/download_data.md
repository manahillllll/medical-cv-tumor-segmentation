# Getting BraTS data

Three practical routes, in order of how much setup they need. `config.yaml`'s
`data.format` field tells the loader which layout to expect (`"h5_slices"` or
`"nifti"`).

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

Neither of the above ships tumor grade labels. Get those from the official challenge:

1. Register at the [Synapse BraTS page](https://www.synapse.org/brats) (free, requires
   an account and accepting the data use agreement).
2. Download the training set (BraTS 2020 ships an explicit HGG/LGG folder split,
   which is the simplest classification label to start with).
3. Build `data/labels.csv` with columns `case_id,label` (e.g. `volume_1,1` for HGG,
   `volume_5,0` for LGG if using `format: "h5_slices"` case ids, or the NIfTI folder
   name if using `format: "nifti"`) — matching case IDs between this label source and
   whichever data source you used above is on you, since the two aren't
   pre-mapped to each other.
4. Pass it to `scripts/train_classifier.py --labels_csv data/labels.csv`.

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
