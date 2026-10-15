import json
import os

import numpy as np
import pytest

from drgrade import frames


def _write_cache(directory, index_extra, n=3):
    """A cache of n float16 frames in the layout `frames.build` writes, with the index fields given."""
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(0)
    block = (rng.random((n, 512, 512, 3)) * 0.9).astype(np.float16)
    np.save(os.path.join(directory, "frames_train_000.npy"), block)
    record = {"split": "train", "images": [f"img{i}.jpeg" for i in range(n)], "grades": [0, 2, 4][:n],
              "boxes": [[0, 100, 0, 100]] * n, "native_hw": [[100, 100]] * n,
              "bytes": os.path.getsize(os.path.join(directory, "frames_train_000.npy"))}
    with open(os.path.join(directory, "index.json"), "w") as fh:
        json.dump({"dtype": "float16", "frame_size": 512, "frames": {"train": n}, "shards": {"frames_train_000.npy": record}, **index_extra}, fh)
    return block


def test_cache_names_its_arm_or_is_refused(tmp_path):
    _write_cache(tmp_path / "named", {"arm": "direct"})
    assert frames.FrameCache(str(tmp_path / "named")).arm == "direct"
    _write_cache(tmp_path / "older", {"version": "crop-frames-v1: Stage 2 -> fundus box cut -> resize -> float16"})
    assert frames.FrameCache(str(tmp_path / "older")).arm == "crop"                 # the layout of an earlier build
    _write_cache(tmp_path / "unnamed", {})
    with pytest.raises(frames.FrameCacheError):
        frames.FrameCache(str(tmp_path / "unnamed"))
    assert frames.FrameCache(str(tmp_path / "unnamed"), arm="direct").arm == "direct"
    with pytest.raises(frames.FrameCacheError):
        frames.FrameCache(str(tmp_path / "named"), arm="crop")                      # contradicts the index


def test_take_copies_the_first_frames_exactly(tmp_path):
    block = _write_cache(tmp_path / "source", {"arm": "crop"})
    index = frames.take(str(tmp_path / "source"), str(tmp_path / "small"), {"train": 2}, log=lambda *_: None)
    small = frames.FrameCache(str(tmp_path / "small"))
    assert small.arm == "crop" and index["frames"] == {"train": 2}
    assert small.entries("train") == [("img0.jpeg", 0), ("img1.jpeg", 2)]
    assert np.array_equal(small.frame("img1.jpeg"), block[1].astype(np.float32)) and "img2.jpeg" not in small
    assert small.geometry("img0.jpeg") == ([0, 100, 0, 100], [100, 100])
