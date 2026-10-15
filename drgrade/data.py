"""Split files and dataset paths.

Split files are in data/splits/ (ids and labels only, no images). Dataset roots come from config.yaml; the folder
structure expected inside each root is described in data/README.md.
"""
import csv
import os

import pandas as pd

from .config import ROOT

SPLITS_DIR = os.path.join(ROOT, "data", "splits")
LESION_CLASSES = ("MA", "HE", "EX", "SE")
IDRID_MASK_DIRS = {"MA": ("1. Microaneurysms", "MA"), "HE": ("2. Haemorrhages", "HE"),
                   "EX": ("3. Hard Exudates", "EX"), "SE": ("4. Soft Exudates", "SE")}


def _csv(name, splits_dir=None, **kwargs):
    return pd.read_csv(os.path.join(splits_dir or SPLITS_DIR, name), **kwargs)


# --------------------------------------------------------------------------- grader data

def aptos_entries(splits_dir=None):
    """{'train': [(id_code, grade)], 'val': [...]} of the usable APTOS 2019 images (2,921 / 730; the 11 images
    whose field of view is empty are excluded)."""
    t = _csv("aptos_train_val.csv", splits_dir, dtype={"id_code": str})
    t = t[t["usable"] == 1]
    return {s: [(r.id_code, int(r.diagnosis)) for r in t[t["split"] == s].itertuples()] for s in ("train", "val")}


def eyepacs_entries(splits_dir=None):
    """{'train': [(image, grade)], 'val': [...]} of the EyePACS adaptation split (31,610 / 3,512 images)."""
    t = _csv("eyepacs_train_val.csv", splits_dir, dtype={"image": str})
    return {s: [(r.image, int(r.grade)) for r in t[t["split"] == s].itertuples()] for s in ("train", "val")}


def aptos_path(cfg, id_code):
    return os.path.join(cfg["datasets"]["aptos"], "train_images", f"{id_code}.png")


def eyepacs_path(cfg, image):
    return os.path.join(cfg["datasets"]["eyepacs"], "train", image)


# --------------------------------------------------------------------------- Stage-1 data (EyeQ)

QUALITY_CLASSES = ("Good", "Usable", "Reject")


def eyeq_table(cfg, split):
    """EyeQ labels of 'train' or 'test' with the resolved image path (rows whose file is missing are dropped)."""
    root = os.path.join(cfg["datasets"]["eyeq"], split)
    t = pd.read_csv(os.path.join(root, "labels.csv"))
    t["path"] = [os.path.join(root, "images", n) for n in t["image"]]
    return t[[os.path.exists(p) for p in t["path"]]].reset_index(drop=True)


# --------------------------------------------------------------------------- Stage-4 data

def idrid_seg_ids(splits_dir=None):
    """{'train': 44 ids, 'val': 10 ids} of the IDRiD lesion-segmentation training images."""
    t = _csv("idrid_stage4_train_val.csv", splits_dir)
    return {s: t[t["split"] == s]["image_id"].tolist() for s in ("train", "val")}


def idrid_seg_image_path(cfg, image_id):
    return os.path.join(cfg["datasets"]["idrid_segmentation"], "1. Original Images", "a. Training Set", f"{image_id}.jpg")


def idrid_seg_mask_path(cfg, image_id, lesion):
    folder, suffix = IDRID_MASK_DIRS[lesion]
    return os.path.join(cfg["datasets"]["idrid_segmentation"], "2. All Segmentation Groundtruths", "a. Training Set", folder,
                        f"{image_id}_{suffix}.tif")


def tjdr_entries(splits_dir=None):
    """Rows of the usable TJDR images: image_id, official_split (train / test), role (train / val), camera."""
    return _csv("tjdr_stage4.csv", splits_dir)


def tjdr_image_path(cfg, official_split, image_id):
    return os.path.join(cfg["datasets"]["tjdr"], official_split, "image", f"{image_id}.png")


def tjdr_mask_path(cfg, official_split, image_id):
    return os.path.join(cfg["datasets"]["tjdr"], official_split, "annotation", f"{image_id}.png")


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- DDR lesion segmentation (localisation check)

def ddr_lesion_sample(splits_dir=None):
    """The 60 DDR lesion-segmentation TRAINING images of the localisation check (names in the order they are
    processed) and whether the Occlusion method also runs on each (the first 20)."""
    t = _csv("ddr_lesion_localisation_sample.csv", splits_dir, dtype={"image": str})
    return [(r.image, bool(r.occlusion)) for r in t.itertuples()]


def ddr_lesion_paths(cfg, name):
    """(image path, {class: mask path}) of one DDR lesion-segmentation training image. Two folder layouts are
    understood, both under <ddr>/lesion_segmentation: train/image + train/label/<class> (as distributed) and
    images/train + annotations/train/<class>."""
    root = os.path.join(cfg["datasets"]["ddr"], "lesion_segmentation")
    stem = os.path.splitext(name)[0]
    layouts = [(os.path.join(root, "train", "image", name), lambda c: os.path.join(root, "train", "label", c, stem + ".tif")),
               (os.path.join(root, "images", "train", name), lambda c: os.path.join(root, "annotations", "train", c, stem + ".tif"))]
    for image, mask in layouts:
        if os.path.exists(image):
            return image, {c: mask(c) for c in LESION_CLASSES}
    raise FileNotFoundError(f"{name} not found under {root} (looked in train/image and images/train)")
