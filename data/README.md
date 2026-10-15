# Data

This repository contains **no dataset images except ten IDRiD demo images** (CC BY 4.0). It holds the evaluation
lists, the labels, the split files and per-image model predictions that make the reported numbers reproducible. The
datasets themselves are downloaded from their owners, subject to their terms.

## What is here

| Path | Content |
|---|---|
| `splits/aptos_train_val.csv` | APTOS 2019 `train_images`: id, grade, split (2,929 train / 733 val); `usable = 0` for the 11 images whose field of view is empty (2,921 / 730 are used) |
| `splits/eyepacs_train_val.csv` | EyePACS images used for training (31,610) and validation (3,512): image, patient, eye, grade, patient_max_grade, split; no patient is in more than one of train, validation and held-out test |
| `splits/eyepacs_heldout_test.csv` | EyePACS held-out test list (16,249 images). Not read by any script in this repository |
| `splits/idrid_stage4_train_val.csv` | the 54 IDRiD lesion-segmentation training images, 44 train / 10 validation (stratified by the presence of soft exudates) |
| `splits/tjdr_stage4.csv`, `splits/tjdr_excluded.csv` | TJDR images used for Stage 4: 443 train, and the 110 images of the official test split used as validation; 8 duplicates removed |
| `splits/ddr_test.csv` | the DDR grading test images used for evaluation, in prediction order: 3,725 images, `primary = 1` for the 3,540 of the primary population; `family` is the file-name family (`007` or `timestamp`) |
| `splits/idrid_test.csv` | the 103 IDRiD test images, `primary = 1` for 100 (three are copies of Stage-4 training images) |
| `splits/ddr_lesion_localisation_sample.csv` | the 60 DDR lesion-segmentation training images of the localisation check |
| `baselines/<ddr,idrid>/` | per-image predictions of the two baseline graders, P and P-EP, three seeds each, on the same images in the same order |
| `training_runs.csv` | per grader run: best epoch (0-based), epochs run, validation QWKs |
| `reliability_calibration.json` | validation rates behind the demo's reliability wording (rebuilt by `scripts/make_reliability_calibration.py`) |
| `demo_images/` | ten IDRiD training-set images with reference grades (`labels.csv`) |

The per-image predictions of the released graders are in `results/predictions/` (DDR, IDRiD) and
`results/validation/` (APTOS and EyePACS validation sets).

### Populations

**DDR primary (3,540 images):** the 4,105 images of the DDR grading test list, minus the 346 labelled ungradable,
minus the 185 that also appear in DDR's lesion-segmentation subset, minus 34 byte-identical duplicates (the first
of each group, by file name, is kept). Grade counts 0-4: 1,846 / 159 / 1,224 / 57 / 254. **Broader (3,725 images):**
the primary population plus the 185 lesion-subset images (a sensitivity set, not untouched). The DDR images carry no
patient identifiers, and the file-name family is tied to the grade (grades 0 and 4 occur in only one family).

**IDRiD primary (100 images):** the 103 test images minus `IDRiD_088`, `IDRiD_089` and `IDRiD_091`, which are copies
of Stage-4 training images. Grade counts of all 103, 0-4: 34 / 5 / 32 / 19 / 13.

## Datasets to download

Download each dataset from its owner and accept its terms. The folder each script expects is set in `config.yaml`
(`datasets:`); only the datasets a command needs have to exist.

| Dataset | Get it from | Needed for | Expected structure under the configured root |
|---|---|---|---|
| APTOS 2019 | https://www.kaggle.com/c/aptos2019-blindness-detection (competition rules apply) | grader training | `train_images/<id_code>.png` |
| EyePACS | https://www.kaggle.com/c/diabetic-retinopathy-detection (competition rules apply) | grader training | `train/<image>.jpeg` |
| EyeQ | https://github.com/HzFu/EyeQ (derived from EyePACS) | Stage 1 | `train/` and `test/`, each with `labels.csv` (image, quality, DR_grade) and `images/` |
| IDRiD | https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid (CC BY 4.0) | Stage 4 (A. Segmentation), external evaluation (B. Disease Grading) | as distributed: `1. Original Images/`, `2. All Segmentation Groundtruths/` (segmentation root); `1. Original Images/b. Testing Set/`, `2. Groundtruths/` (grading root) |
| TJDR | https://github.com/NekoPii/TJDR | Stage 4 | `train/` and `test/`, each with `image/<id>.png` and `annotation/<id>.png` (palette-index masks: EX 1, HE 2, MA 3, SE 4) |
| DDR | https://github.com/nkicsl/DDR-dataset | external evaluation, localisation check | `DR_grading/test/<image>.jpg`; `lesion_segmentation/train/image/` and `lesion_segmentation/train/label/<MA,HE,EX,SE>/*.tif` |

The expected structure is as the dataset owners distribute it, except where a dataset has several layouts in
circulation; the code names the file it could not find. Check the licence of each dataset before using or
redistributing it: TJDR in particular carries no licence on the data files themselves, so its images should not be
redistributed.

## Not included, on purpose

Images of APTOS, EyePACS, EyeQ, DDR and TJDR; model weights (see `../WEIGHTS_TERMS.md` and
`../weights_manifest.json`); any file that identifies a patient.
