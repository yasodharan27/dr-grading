import dataclasses
import json
import os

import numpy as np
import pytest

from drgrade import train_lesions as tl

torch = pytest.importorskip("torch")


def test_balanced_batches_are_half_and_half_and_reproducible():
    names = {"IDRiD": [f"i{k}" for k in range(5)], "TJDR": [f"t{k}" for k in range(9)]}
    a, b = tl.balanced_batch_plan(names, 8, 17, 42), tl.balanced_batch_plan(names, 8, 17, 42)
    assert a == b and a != tl.balanced_batch_plan(names, 8, 18, 42)
    assert sum(n.startswith("i") for n, _ in a) == 4 and sum(n.startswith("t") for n, _ in a) == 4
    with pytest.raises(ValueError):
        tl.balanced_batch_plan(names, 7, 0, 42)


def test_patch_is_centred_on_a_lesion_pixel_when_asked():
    masks = np.zeros((4, 256, 256), np.uint8)
    masks[2, 200, 40] = 1
    rng = np.random.default_rng(0)
    y0, x0 = tl.sample_patch_origin(rng, masks, patch=64, p_lesion=1.0)
    assert y0 <= 200 < y0 + 64 and x0 <= 40 < x0 + 64 and 0 <= y0 <= 192 and 0 <= x0 <= 192
    y0, x0 = tl.sample_patch_origin(np.random.default_rng(1), np.zeros((4, 256, 256), np.uint8), patch=64, p_lesion=1.0)   # no lesion at all
    assert 0 <= y0 <= 192 and 0 <= x0 <= 192


def test_positive_weights_are_clipped():
    w = tl.positive_weights([1, 400, 10000], [10 ** 6, 10 ** 6, 10 ** 6])
    assert w.tolist() == pytest.approx([20.0, 20.0, 10.0])
    assert tl.positive_weights([10 ** 6], [10 ** 6 // 4])[0] == 1.0
    with pytest.raises(ValueError):
        tl.positive_weights([0], [1])


def test_loss_is_small_for_a_perfect_map_and_large_for_the_inverse():
    y = torch.zeros(2, 4, 32, 32)
    y[:, :, 8:16, 8:16] = 1.0
    fov = torch.ones(2, 1, 32, 32)
    good = (y * 2 - 1) * 12.0
    l_good, _ = tl.loss_fn(good, y, fov, [1.0] * 4)
    l_bad, _ = tl.loss_fn(-good, y, fov, [1.0] * 4)
    assert float(l_good) < 0.01 and float(l_bad) > 5.0


def test_tjdr_mask_codes_are_split_by_class_without_conversion(tmp_path):
    from PIL import Image
    index = np.zeros((6, 6), np.uint8)
    index[0, 0], index[1, 1], index[2, 2], index[3, 3] = 3, 2, 1, 4                      # MA, HE, EX, SE
    os.makedirs(tmp_path / "train" / "annotation")
    im = Image.fromarray(index, mode="P")
    im.putpalette([0, 0, 0] + [255, 255, 255] * 4 + [0] * (768 - 15))
    im.save(tmp_path / "train" / "annotation" / "TJDR_train_001.png")
    masks = tl.read_tjdr_masks({"datasets": {"tjdr": str(tmp_path)}}, "train", "TJDR_train_001")
    assert masks.shape == (4, 6, 6)
    for k, (r, c) in enumerate([(0, 0), (1, 1), (2, 2), (3, 3)]):
        assert masks[k].sum() == 1 and masks[k, r, c] == 1


class TinySegmenter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU())
        self.head = torch.nn.Conv2d(8, 4, 1)

    def forward(self, x):
        return self.head(self.encoder(x))


def _make_cache(directory):
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(0)
    files = {}
    for dataset, role, n in (("IDRiD", "train", 3), ("TJDR", "train", 3), ("IDRiD", "val", 1), ("TJDR", "val", 1)):
        for k in range(n):
            image_id = f"{dataset}_{role}_{k}"
            name = tl.sample_name(dataset, role, image_id)
            rgb = (rng.random((96, 96, 3)) * 200 + 30).astype(np.uint8)
            masks = np.zeros((4, 96, 96), np.uint8)
            masks[k % 4, 20:40, 20:40] = 1
            masks[(k + 1) % 4, 60:70, 10:30] = 1
            meta = {"dataset": dataset, "role": role, "split": "x", "image_id": image_id, "camera": "TJDR-TRC50DX" if dataset == "TJDR" else "IDRiD-Kowa"}
            np.savez(os.path.join(directory, name), rgb=rgb, masks=masks, meta=np.asarray(json.dumps(meta)))
            files[name] = meta
    counts = {}
    for m in files.values():
        counts[f"{m['dataset']}/{m['role']}"] = counts.get(f"{m['dataset']}/{m['role']}", 0) + 1
    with open(os.path.join(directory, "manifest.json"), "w") as fh:
        json.dump({"files": files, "counts": counts, "complete": False}, fh)
    return tl.TrainingCache(directory, require_complete=False)


def _train(run_dir, cache, steps):
    cfg = dataclasses.replace(tl.TrainConfig(), iterations=6, batch_size=2, patch=64, val_every=6, checkpoint_every=3, warmup_steps=2,
                              num_workers=0, amp=False)
    torch.manual_seed(0)
    return tl.train(str(run_dir), cache, cfg, "cpu", steps, model=TinySegmenter(), log=lambda *_: None)


def test_training_is_resumable_and_equals_an_uninterrupted_run(tmp_path):
    cache = _make_cache(tmp_path / "cache")
    whole = _train(tmp_path / "whole", cache, 6)
    _train(tmp_path / "split", cache, 3)
    resumed = _train(tmp_path / "split", cache, 3)
    assert whole["step"] == resumed["step"] == 6
    a = torch.load(tmp_path / "whole" / "latest.pt", weights_only=False)
    b = torch.load(tmp_path / "split" / "latest.pt", weights_only=False)
    for key in a["ema"]:
        assert torch.allclose(a["ema"][key].float(), b["ema"][key].float(), atol=1e-6), key
    for key in a["model"]:
        assert torch.allclose(a["model"][key].float(), b["model"][key].float(), atol=1e-6), key
    assert np.isfinite(whole["best"]["score"]) and os.path.exists(tmp_path / "whole" / "best_ema.pt")
    out = tl.export_best(str(tmp_path / "whole"), str(tmp_path / "export" / "lesions.pt"))
    assert os.path.exists(out)
