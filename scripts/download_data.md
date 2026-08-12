# Getting BraTS data

There are two practical routes. Start with the auto-downloadable one; move to the
official release once the pipeline works end-to-end.

## Option A — MONAI auto-download (fastest, no registration)

The Medical Segmentation Decathlon's **Task01_BrainTumour** is BraTS-derived data
(same 4 MRI modalities, same tumor-subregion labeling scheme) and MONAI can fetch it
directly:

```python
from monai.apps import DecathlonDataset

DecathlonDataset(
    root_dir="data/",
    task="Task01_BrainTumour",
    section="training",
    download=True,
)
```

This downloads and extracts into `data/Task01_BrainTumour/`. Reorganize (or symlink)
into this project's expected per-case layout:

```
data/brats/<case_id>/<case_id>_t1.nii.gz
data/brats/<case_id>/<case_id>_t1ce.nii.gz
data/brats/<case_id>/<case_id>_t2.nii.gz
data/brats/<case_id>/<case_id>_flair.nii.gz
data/brats/<case_id>/<case_id>_seg.nii.gz
```

Note: Task01_BrainTumour ships each case as a single 4-channel NIfTI
(`imagesTr/BRATS_XXX.nii.gz`, channel order FLAIR/T1w/T1gd/T2w) rather than four
separate files. Either write a one-off script to split each 4D volume into the four
per-modality files above, or adapt `src/data/dataset.py::discover_cases` to load the
combined 4D file directly — whichever is less work for the split you're using.

## Option B — Official BraTS release (needed for grade/subtype classification labels)

The classification head (tumor grade / subtype) needs labels the auto-downloadable
Decathlon mirror doesn't include. Get those from the official challenge data:

1. Register at the [Synapse BraTS page](https://www.synapse.org/brats) (free, requires
   an account and accepting the data use agreement).
2. Download the training set for the year you want (BraTS 2020/2021 both work fine;
   2020 ships an explicit HGG/LGG folder split, which is the simplest classification
   label to start with).
3. Place cases under `data/brats/` following the layout above.
4. Build `data/labels.csv` with columns `case_id,label` (e.g. `BraTS20_Training_001,1`
   for HGG, `...,0` for LGG) — this is what `scripts/train_classifier.py --labels_csv`
   expects.

## Sanity-checking the download

Before training anything:

```bash
python -c "from src.data.dataset import discover_cases; print(len(discover_cases('data/brats')))"
```

This should report the number of complete cases (all 4 modalities + segmentation
label found) discovered under `data/brats/`.
