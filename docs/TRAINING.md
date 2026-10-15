# Retraining the models

Every model the project trained can be retrained from the public datasets. Stage 3 (LWNet vessel segmentation) is a
published model that is used as it is.

Retraining is **not bit-exact**: GPU kernels are nondeterministic, so a retrained model differs from the released one
in its last digits and its validation scores differ by a few thousandths. What is fixed is the data, its order,
the augmentation draws, the loss and the schedule.

## Order

```bash
python scripts/download_weights.py --only convnext_tiny_imagenet_notop.h5   # ImageNet initialisation of the grader
python scripts/train_quality.py                                             # Stage 1, EyeQ
python scripts/train_lesions.py                                             # Stage 4, IDRiD + TJDR
python scripts/prepare_frames.py --dataset aptos   --arm crop               # grader frames
python scripts/prepare_frames.py --dataset eyepacs --arm crop
python scripts/train_grader.py --frame crop --seed 42                       # one run per seed: 42, 123, 2026
```

All four trainers resume: rerun the same command after an interrupted session. Output goes to `output_dir` from
`config.yaml`. The released file names are used when a run finishes and the file is not already in `weights_dir`.

## Stage 1: image quality (`train_quality.py`)

EfficientNet-B0 (ImageNet weights, first 100 layers frozen) on the EyeQ training images, 85 / 15 train / validation
split stratified by quality (seed 42), batch 32, Adam at 1e-4, class-weighted cross-entropy, halve the learning rate
after 4 epochs without a lower validation loss, stop after 8, keep the lowest validation loss. The EyeQ test split
is read only by the evaluation at the end (16,249 images).

## Stage 4: lesion segmentation (`train_lesions.py`)

U-Net with an ImageNet SE-ResNet-101 encoder. IDRiD: the 54 lesion-segmentation training images, 44 for training and
10 for validation (stratified by the presence of soft exudates). TJDR: 443 images for training and the 110 images of
its official test split as the validation set, after removing 8 duplicates (`data/splits/tjdr_excluded.csv`).
Batches of 8, half from each dataset, 512 x 512 patches centred on a lesion half of the time, 12,000 iterations,
weights averaged (EMA 0.999), validation every 1,000 steps.

**The best checkpoint is selected on TJDR's official test split, used as the validation set (with the 10 IDRiD
validation images). The Stage-4 numbers it reports are selection-set numbers, not a held-out evaluation.**

The IDRiD Stage-2 images pass through a JPEG round trip (quality 95) before they are resized, because the released
model was trained on Stage-2 files that had been written as `.jpg`; this reproduces its training images exactly
(`--no-idrid-jpeg` keeps them lossless, as at inference).

The ImageNet encoder is downloaded from Hugging Face (`smp-hub/se_resnet101.imagenet`, 198 MB).

## Grader (`prepare_frames.py`, `train_grader.py`)

Joint APTOS 2019 + EyePACS training. One epoch is all 2,921 APTOS training images plus 5,842 EyePACS images (a
rotating slice of one seeded shuffle of the 31,610 EyePACS training images); batch 16; AdamW 1e-4, weight decay
0.05 (none on biases and normalisation parameters); weighted CORN loss with the grade weights
0.6456 / 1.8421 / 1.1954 / 2.8293 / 2.5958; mixed precision on GPU. After every epoch both full validation sets (730
APTOS, 3,512 EyePACS images) are scored and the mean of the two QWKs is monitored: the learning rate is halved after
4 epochs without improvement (down to 1e-6), training stops after 12, at most 50 epochs; the best epoch is kept.
`--frame crop` cuts every image to its fundus box before the resize, `--frame direct` resizes the whole image.

Frames built from raw images were verified against an independently built cache only on a few images, locally: 12
APTOS and 28 EyePACS crop frames came out bit-identical (float16). A full build of the 35,122 EyePACS frames was not run
during development of this repository.

Optional: a frame cache of the same shard layout made by other means (`frames_<split>_<nnn>.npy` shards of float16
(n, 512, 512, 3) frames and an `index.json` listing, per shard, `split`, `images`, `grades` and `bytes`) can be used
instead of building frames: pass its folder as `--aptos-frames` / `--eyepacs-frames` (the index need not name its arm;
`--frame` states it). Copy it to local disk first, since reading 50 GB through a mounted drive is slow.
`prepare_frames.py --from-cache DIR --arm crop --limit N --limit-val M --out OUT` copies the first N training and M
validation frames of such a cache into a small one (this is how the GPU smoke test can run without raw EyePACS images).

Disk for the frames of ONE arm (float16, 1.5 MB per frame): APTOS 3,651 frames 5.5 GB, EyePACS 35,122 frames 53 GB. A
Colab T4 runtime has about 71 GB free, so build and train one arm at a time and delete the other arm's frames.

## How long it takes

Measured on a Colab T4 during the original runs (batch 16, mixed precision; end-to-end per epoch, including both
validation passes, checkpointing and start-up of each session):

| | epochs before early stopping | per epoch | per seed |
|---|---|---|---|
| whole-image frames (`direct`), seed 123 | 31 | about 11.5 min | about 5.9 h |
| fundus-cropped frames (`crop`), seed 123 | 36 | about 10.3 min | about 6.2 h |

That is about 6 hours per seed when uninterrupted, so about 18 hours for the three seeds of one arm and about 36 hours
for both arms, plus interruptions (the original runs were spread over one to two days per arm). The data loading of one batch (0.23 s) is far below the optimiser step (0.60 s per step of 16
images, 26.6 images per second), so a T4 is the limit. Stage 4 was sized for about 3 T4-hours (12,000 iterations); no
measured duration of that run is available. Stage 1 took about 25 minutes for five small epochs in the GPU smoke test,
dominated by reading images through a mounted drive; copy the data to local disk first.

## Smoke test on a GPU

`python scripts/smoke_gpu.py` trains every trainer for a few hundred steps on small subsets of the real datasets
(two invocations each, to exercise resume) and checks that losses are finite and fall and that checkpoints exist.
On a Colab T4 all checks passed: Stage 1, Stage 4 and the grader each ran, resumed from their checkpoint after the
first invocation and had finite, falling training losses (Stage 4's three validation scores sat at chance level,
as expected after 300 steps of an averaged-weights model). The epoch logs of the grader and Stage-1 trainers print the
host RAM of the process, which should stay flat from epoch to epoch; **whether it does was not verified** (the epoch
lines of that run were not kept).
