"""The grader's 512 x 512 frames of the training images, built once from the raw files and read back by the trainer.

A cache folder holds shards of at most 1,000 frames, `frames_<split>_<nnn>.npy` (float16, (n, 512, 512, 3), values
in [0, 1]; float16 is the precision the released graders were trained with), a small `.json` record next to each shard
(images, grades, fundus boxes, native sizes) and `index.json` once every requested shard exists. A shard is written
whole or not at all, so an interrupted build resumes at the first missing shard.

Frames are Stage 2 (gamma, CLAHE) of the raw image, then (crop arm) cut to the fundus box, then resized to 512 x 512.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import preprocess
from .training import atomic_json, read_json

SHARD_SIZE = 1000
FRAME = preprocess.FRAME_SIZE


class FrameCacheError(RuntimeError):
    pass


def plan(items, shard_size=SHARD_SIZE):
    """{split: [(image id, grade, raw path)]} -> [(shard name, split, items of that shard)] in build order."""
    out = []
    for split, rows in items.items():
        for k in range(0, len(rows), shard_size):
            out.append((f"frames_{split}_{k // shard_size:03d}.npy", split, rows[k:k + shard_size]))
    return out


def build(items, out_dir, arm, workers=4, shard_size=SHARD_SIZE, log=print):
    """Builds every missing shard. `items`: {split: [(image id, grade, raw path), ...]}. Returns the index dict."""
    if arm not in ("crop", "direct"):
        raise ValueError(f"arm must be 'crop' or 'direct', got {arm!r}")
    os.makedirs(out_dir, exist_ok=True)
    shards, built = {}, 0

    def one(item):
        image, _, path = item
        native = preprocess.stage2(path)
        bgr = preprocess.read_bgr(path)
        if bgr.shape[:2] != native.shape[:2]:
            raise FrameCacheError(f"{path}: the raw image and its Stage-2 image differ in size")
        box = preprocess.fundus_box(bgr) if arm == "crop" else (0, native.shape[0], 0, native.shape[1])
        top, bottom, left, right = box
        rgb = native if arm == "direct" else np.ascontiguousarray(native[top:bottom, left:right])
        return preprocess.resize_frame(rgb).astype(np.float16), box, native.shape[:2]

    planned = plan(items, shard_size)
    for number, (name, split, rows) in enumerate(planned, 1):
        path, record_path = os.path.join(out_dir, name), os.path.join(out_dir, name + ".json")
        images = [r[0] for r in rows]
        if os.path.exists(path) and os.path.exists(record_path):
            record = read_json(record_path)
            if record["images"] == images and record["arm"] == arm and record["bytes"] == os.path.getsize(path):
                shards[name] = record
                continue
        block = np.lib.format.open_memmap(path + ".part", mode="w+", dtype=np.float16, shape=(len(rows), FRAME, FRAME, 3))
        boxes, natives = [], []
        with ThreadPoolExecutor(max(1, workers)) as pool:
            for row, (frame, box, native) in enumerate(pool.map(one, rows)):
                block[row] = frame
                boxes.append(list(box))
                natives.append([int(native[0]), int(native[1])])
        block.flush()
        del block
        os.replace(path + ".part", path)
        record = {"split": split, "arm": arm, "images": images, "grades": [int(r[1]) for r in rows], "boxes": boxes,
                  "native_hw": natives, "bytes": os.path.getsize(path)}
        atomic_json(record_path, record)
        shards[name] = record
        built += 1
        log(f"  {name} ({len(rows)} frames) | shard {number} of {len(planned)}")
    index = {"arm": arm, "frame_size": FRAME, "dtype": "float16", "rule": {
        "gamma": preprocess.GAMMA, "clahe_clip_limit": preprocess.CLAHE_CLIP_LIMIT, "clahe_tile_grid": list(preprocess.CLAHE_TILE_GRID),
        "gray_threshold": preprocess.GRAY_THRESHOLD, "line_fraction": preprocess.LINE_FRACTION,
        "min_side_fraction": preprocess.MIN_SIDE_FRACTION},
        "frames": {split: len(rows) for split, rows in items.items()}, "shards": shards}
    atomic_json(os.path.join(out_dir, "index.json"), index)
    log(f"  built {built} shard(s), {len(planned) - built} already present")
    return index


class FrameCache:
    """Read access: `entries(split)` in build order, `frame(image)` as float32 (512, 512, 3) in [0, 1], and the
    arm the cache was built for. Refuses a folder without an index (an unfinished build).

    The index may come from `build` (it names its arm) or from an earlier build with the same shard layout, which
    states the crop rule in its `version` text instead; pass `arm` for any other index."""

    def __init__(self, directory, arm=None):
        path = os.path.join(directory, "index.json")
        if not os.path.exists(path):
            raise FrameCacheError(f"{directory} has no index.json: run scripts/prepare_frames.py first")
        index = read_json(path)
        found = index.get("arm") or ("crop" if str(index.get("version", "")).startswith("crop-frames") else None)
        if arm and found and arm != found:
            raise FrameCacheError(f"{directory} holds {found} frames, not {arm}")
        if not (arm or found):
            raise FrameCacheError(f"{directory}: the index does not name the arm; pass arm='crop' or 'direct'")
        if index.get("dtype", "float16") != "float16" or index.get("frame_size", FRAME) != FRAME:
            raise FrameCacheError(f"{directory} does not hold float16 {FRAME} x {FRAME} frames")
        self.directory, self.arm, self.index = directory, arm or found, index
        self._where, self._entries, self._open = {}, {}, {}
        for name in sorted(index["shards"]):
            record = index["shards"][name]
            shard = os.path.join(directory, name)
            if not os.path.exists(shard) or os.path.getsize(shard) != record["bytes"]:
                raise FrameCacheError(f"{shard} is missing or has another size")
            for row, (image, grade) in enumerate(zip(record["images"], record["grades"])):
                if image in self._where:
                    raise FrameCacheError(f"{image} is in the cache twice")
                self._where[image] = (name, row)
                self._entries.setdefault(record["split"], []).append((image, int(grade)))

    def entries(self, split):
        return list(self._entries.get(split, []))

    def __contains__(self, image):
        return str(image) in self._where

    def frame(self, image):
        name, row = self._where[str(image)]
        block = self._open.get(name)
        if block is None:
            block = self._open[name] = np.load(os.path.join(self.directory, name), mmap_mode="r")
        return np.asarray(block[row], dtype=np.float32)

    def geometry(self, image):
        """(fundus box, native (h, w)) recorded for an image, or (None, None)."""
        name, row = self._where[str(image)]
        record = self.index["shards"][name]
        box, native = record.get("boxes"), record.get("native_hw")
        return (box[row] if box else None), (native[row] if native else None)


def take(source_dir, out_dir, counts, arm=None, shard_size=SHARD_SIZE, log=print):
    """A small local cache from the first `counts[split]` frames of each split of an existing cache with the same
    layout, e.g. a few hundred frames of a large cache on a mounted drive, for smoke tests. Returns the new index."""
    source = FrameCache(source_dir, arm)
    os.makedirs(out_dir, exist_ok=True)
    shards = {}
    for split, n in counts.items():
        rows = source.entries(split)[:n]
        for k in range(0, len(rows), shard_size):
            part = rows[k:k + shard_size]
            name = f"frames_{split}_{k // shard_size:03d}.npy"
            path = os.path.join(out_dir, name)
            block = np.lib.format.open_memmap(path + ".part", mode="w+", dtype=np.float16, shape=(len(part), FRAME, FRAME, 3))
            geometry = [source.geometry(image) for image, _ in part]
            for row, (image, _) in enumerate(part):
                block[row] = source.frame(image).astype(np.float16)
            block.flush()
            del block
            os.replace(path + ".part", path)
            record = {"split": split, "arm": source.arm, "images": [i for i, _ in part], "grades": [g for _, g in part],
                      "boxes": [g[0] for g in geometry] if all(g[0] for g in geometry) else None,
                      "native_hw": [g[1] for g in geometry] if all(g[1] for g in geometry) else None,
                      "bytes": os.path.getsize(path)}
            atomic_json(path + ".json", record)
            shards[name] = record
            log(f"  {name} ({len(part)} frames) from {source_dir}")
    index = {"arm": source.arm, "frame_size": FRAME, "dtype": "float16",
             "frames": {s: min(n, len(source.entries(s))) for s, n in counts.items()}, "shards": shards}
    atomic_json(os.path.join(out_dir, "index.json"), index)
    return index


def cache_dir(cfg, dataset, arm):
    return os.path.join(cfg["output_dir"], "frames", f"{dataset}_{arm}")


def describe(directory):
    index = read_json(os.path.join(directory, "index.json"))
    boxes = np.asarray([b for s in index["shards"].values() for b in (s.get("boxes") or [])], np.float64)
    native = np.asarray([n for s in index["shards"].values() for n in (s.get("native_hw") or [])], np.float64)
    if not len(boxes):
        return {"frames": index["frames"]}
    height, width = (boxes[:, 1] - boxes[:, 0]) / native[:, 0], (boxes[:, 3] - boxes[:, 2]) / native[:, 1]
    q = lambda v: [round(float(x), 3) for x in np.percentile(v, [10, 50, 90])]      # noqa: E731
    return {"arm": index["arm"], "frames": index["frames"], "left_uncut": int(np.sum((height == 1.0) & (width == 1.0))),
            "box_width_fraction_p10_50_90": q(width), "box_height_fraction_p10_50_90": q(height)}


def json_summary(directory):
    return json.dumps(describe(directory))
