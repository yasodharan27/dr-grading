"""Pieces shared by the trainers: seeding, atomic file writes, and the learning-rate plateau / early-stopping rule.

`PlateauController` reproduces Keras' `ReduceLROnPlateau` (min_delta 1e-4, cooldown 0) and `EarlyStopping`
(min_delta 0, restore best weights) on one monitored value per epoch, as a small object whose state can be saved
and restored, so a crashed run resumes with the same counters.
"""
import json
import os

import numpy as np


def atomic_json(path, obj):
    """Writes JSON through a temporary file so a crash never leaves a half-written file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(tmp, path)


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class PlateauController:
    """Per-epoch bookkeeping of one monitored value.

    `update(value, lr)` returns {"improved": best so far (EarlyStopping rule, first epoch with the highest value),
    "lr": the learning rate to use next, "stop": patience exceeded}. Rules, as in Keras:
      * EarlyStopping: wait += 1; if value - stop_min_delta beats the best, record it and wait = 0; stop when
        wait >= stop_patience and epoch > 0.
      * ReduceLROnPlateau: if value - lr_min_delta beats ITS best, record it and wait = 0, otherwise wait += 1 and,
        at wait >= lr_patience, lr = max(lr * lr_factor, min_lr) and wait = 0 (only while lr > min_lr).
    `mode` is "max" or "min".
    """

    def __init__(self, mode="max", stop_patience=12, lr_patience=4, lr_factor=0.5, min_lr=1e-6, lr_min_delta=1e-4, stop_min_delta=0.0):
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        self.mode, self.stop_patience, self.lr_patience = mode, int(stop_patience), int(lr_patience)
        self.lr_factor, self.min_lr = float(lr_factor), float(min_lr)
        self.lr_min_delta, self.stop_min_delta = float(lr_min_delta), float(stop_min_delta)
        self.stop_best = self.lr_best = None
        self.stop_wait = self.lr_wait = 0
        self.best_epoch = None

    def _beats(self, value, reference, delta):
        if reference is None:
            return True
        return (value - delta) > reference if self.mode == "max" else (value + delta) < reference

    def update(self, epoch, value, lr):
        value = float(value)
        self.stop_wait += 1
        improved = self._beats(value, self.stop_best, self.stop_min_delta)
        stop = False
        if improved:
            self.stop_best, self.best_epoch, self.stop_wait = value, int(epoch), 0
        elif self.stop_wait >= self.stop_patience and epoch > 0:
            stop = True
        if self._beats(value, self.lr_best, self.lr_min_delta):
            self.lr_best, self.lr_wait = value, 0
        else:
            self.lr_wait += 1
            if self.lr_wait >= self.lr_patience:
                if lr > np.float32(self.min_lr):
                    lr = max(lr * self.lr_factor, self.min_lr)
                    self.lr_wait = 0
        return {"improved": improved, "lr": float(lr), "stop": stop}

    def state_dict(self):
        return {"stop_best": self.stop_best, "lr_best": self.lr_best, "stop_wait": self.stop_wait, "lr_wait": self.lr_wait,
                "best_epoch": self.best_epoch}

    def load_state_dict(self, state):
        self.stop_best, self.lr_best = state["stop_best"], state["lr_best"]
        self.stop_wait, self.lr_wait, self.best_epoch = int(state["stop_wait"]), int(state["lr_wait"]), state["best_epoch"]


def seed_everything(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def optimizer_state(optimizer):
    """All optimizer variables as a list of numpy arrays (for resuming)."""
    return [np.asarray(v.numpy()) for v in optimizer.variables]


def restore_optimizer_state(optimizer, arrays):
    variables = optimizer.variables
    if len(variables) != len(arrays):
        raise RuntimeError(f"the saved optimizer has {len(arrays)} variables, this one {len(variables)}")
    for v, a in zip(variables, arrays):
        v.assign(a)


# --------------------------------------------------------------------------- checkpoint files of a run

def save_weights_atomic(model, path):
    """Writes through a temporary file (Keras insists on the `.weights.h5` ending) so a crash never leaves a
    partial file."""
    tmp = path[:-len(".weights.h5")] + ".part.weights.h5"
    model.save_weights(tmp)
    os.replace(tmp, path)


def save_generation(run_dir, model, epoch, save_weights=None):
    """Weights and optimiser state after `epoch` into <run_dir>/checkpoints; returns the two file names.
    `save_weights(model, path)` replaces the default `save_weights_atomic` (e.g. to leave the optimiser out)."""
    folder = os.path.join(run_dir, "checkpoints")
    os.makedirs(folder, exist_ok=True)
    weights = os.path.join(folder, f"last_{epoch:04d}.weights.h5")
    optimizer = os.path.join(folder, f"last_{epoch:04d}_optimizer.npz")
    (save_weights or save_weights_atomic)(model, weights)
    np.savez(optimizer + ".part.npz", *optimizer_state(model.optimizer))
    os.replace(optimizer + ".part.npz", optimizer)
    return os.path.basename(weights), os.path.basename(optimizer)


def load_generation(run_dir, model, weights_name, optimizer_name):
    """Restores weights and optimiser state written by `save_generation` into a compiled model."""
    folder = os.path.join(run_dir, "checkpoints")
    model.load_weights(os.path.join(folder, weights_name))
    model.optimizer.build(model.trainable_variables)
    with np.load(os.path.join(folder, optimizer_name)) as saved:
        restore_optimizer_state(model.optimizer, [saved[f"arr_{i}"] for i in range(len(saved.files))])


def prune_generations(run_dir, keep):
    folder = os.path.join(run_dir, "checkpoints")
    for name in os.listdir(folder):
        if name.startswith("last_") and name not in keep:
            os.remove(os.path.join(folder, name))


def set_learning_rate(optimizer, lr):
    """Sets the learning rate of an optimiser, or of the one a loss-scale wrapper holds."""
    inner = getattr(optimizer, "inner_optimizer", None) or optimizer
    inner.learning_rate = lr


def rss_gb():
    """Resident memory of this process in GB (None when psutil is not installed): logged once per epoch so a leak
    shows as a rising number."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 30
    except ImportError:
        return None


def rss_text():
    value = rss_gb()
    return "" if value is None else f" | host RAM {value:.1f} GB"
