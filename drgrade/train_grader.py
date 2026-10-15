"""Training of the grader: ConvNeXt-Tiny (ImageNet) + CORN head on joint APTOS 2019 + EyePACS frames.

Protocol (one run = one seed, one frame arm):
  * epoch: every APTOS training image plus twice as many EyePACS training images, the EyePACS part a rotating
    slice of ONE seeded shuffle (so the epochs of a resumed run are those of an uninterrupted run);
  * batch 16, AdamW (learning rate 1e-4, weight decay 0.05, none on 1-D parameters), mixed precision on GPU;
  * loss: CORN with per-grade weights w_c ~ p_c^-0.5 normalised so that sum p_c w_c = 1, where p is the grade
    distribution of an epoch; the weighted sum is divided by the UNWEIGHTED number of (image, threshold) pairs;
  * augmentation (training only): random flips and quarter turns, brightness and contrast jitter of +-10 %, each
    image with its own generator seeded from (run seed, epoch, image id);
  * after every epoch BOTH full validation sets are scored; the monitored value is the mean of the APTOS and the
    EyePACS validation QWK; the checkpoint with the highest value is kept (first epoch on ties);
  * learning rate x0.5 after 4 epochs without improvement (down to 1e-6); stop after 12 epochs without improvement;
    at most 50 epochs;
  * no EMA, no dropout.

The loop is resumable: after every epoch the weights, the optimiser state and the bookkeeping are written, and a
restarted run continues from there. Training and prediction run as traced functions (see `drgrade.explain` for why).
GPU nondeterminism means a retrained model is not bit-identical to the released one.
"""
import hashlib
import json
import os
import shutil
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import metrics as mt
from . import grader as gr
from .training import (PlateauController, atomic_json, load_generation, prune_generations, read_json, rss_text, save_generation,
                       save_weights_atomic, set_learning_rate)

PROTOCOL = {"batch_size": 16, "max_epochs": 50, "learning_rate": 1e-4, "weight_decay": 0.05, "stop_patience": 12,
            "lr_patience": 4, "lr_factor": 0.5, "min_lr": 1e-6}
EYEPACS_PER_APTOS = 2
ORDER_TAG = "joint-eyepacs-rotation-v1"
AUGMENTATION_TAG = "improved_training_data.augmentation.v1"
EPOCH_ORDER_TAG = "improved_training_data.order.v1"
WEIGHT_POWER = 0.5
#: Grade weights of the full joint data (2,921 APTOS + 5,842 EyePACS images per epoch), to four decimals.
APPROVED_CLASS_WEIGHTS = (0.6456, 1.8421, 1.1954, 2.8293, 2.5958)
FULL_COUNTS = {"aptos_train": 2921, "eyepacs_train": 31610, "aptos_val": 730, "eyepacs_val": 3512}
WEIGHT_TOLERANCE = 0.02
EVAL_BATCH = 16


class TrainingError(RuntimeError):
    pass


# --------------------------------------------------------------------------- epochs, weights, augmentation

def seed_from_key(tag, *parts):
    """A deterministic, position-independent 63-bit seed derived from `tag` and `parts`."""
    key = "|".join(str(p) for p in (tag,) + parts)
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def eyepacs_rotation(eyepacs_train, seed):
    """The EyePACS training images in ONE fixed shuffled order for this seed (sorted by name, then permuted)."""
    entries = sorted((str(i), int(g)) for i, g in eyepacs_train)
    digest = hashlib.sha256(f"{ORDER_TAG}|{int(seed)}".encode()).digest()
    order = np.random.default_rng(int.from_bytes(digest[:8], "little")).permutation(len(entries))
    return [entries[k] for k in order]


def epoch_entries(aptos_train, rotation, epoch):
    """Every APTOS training image and the next 2 x len(APTOS) images of the rotation, wrapping round. A pure
    function of (rotation, epoch)."""
    n = EYEPACS_PER_APTOS * len(aptos_train)
    if n > len(rotation):
        raise TrainingError("the EyePACS training part is smaller than one epoch's share")
    start = (int(epoch) * n) % len(rotation)
    return [(str(i), int(g)) for i, g in aptos_train] + [rotation[(start + k) % len(rotation)] for k in range(n)]


def epoch_order(entries, seed, epoch):
    """The epoch's images in a permutation that depends only on (run seed, epoch) and the SET of images."""
    canonical = sorted(((str(i), int(g)) for i, g in entries), key=lambda e: e[0])
    rng = np.random.default_rng(seed_from_key(EPOCH_ORDER_TAG, seed, epoch))
    return [canonical[i] for i in rng.permutation(len(canonical))]


def grade_counts(entries):
    return np.bincount([int(g) for _, g in entries], minlength=5)


def class_weights_from_counts(counts, power=WEIGHT_POWER):
    """w_c ~ p_c^-power, normalised so that sum_c p_c w_c = 1."""
    counts = np.asarray(counts, np.float64)
    if counts.ndim != 1 or counts.size < 2 or np.any(counts <= 0):
        raise ValueError(f"need one positive count per class, got {counts.tolist()}")
    p = counts / counts.sum()
    w = p ** (-power)
    return w / np.sum(p * w)


def expected_counts(aptos_train, eyepacs_train):
    """Expected grade counts of one epoch: every APTOS image plus EyePACS at its training prevalence, scaled to
    twice the APTOS image count."""
    share = EYEPACS_PER_APTOS * len(aptos_train) / len(eyepacs_train)
    return grade_counts(aptos_train) + grade_counts(eyepacs_train) * share


def class_weights(aptos_train, eyepacs_train, seed=None, epochs=None):
    """(weights as a list of 5 floats, expected counts). For the full joint data the weights must be the approved
    ones, and -- when a seed is given -- within WEIGHT_TOLERANCE of the weights of the mixture actually sampled over
    `epochs`. Subsets (smoke tests) use the approved weights as they are."""
    full = (len(aptos_train), len(eyepacs_train)) == (FULL_COUNTS["aptos_train"], FULL_COUNTS["eyepacs_train"])
    if not full:
        return list(APPROVED_CLASS_WEIGHTS), None
    expected = expected_counts(aptos_train, eyepacs_train)
    w = class_weights_from_counts(expected)
    if not np.allclose(w, APPROVED_CLASS_WEIGHTS, atol=5e-5):
        raise TrainingError(f"class weights {np.round(w, 4).tolist()} are not the approved {APPROVED_CLASS_WEIGHTS}")
    if seed is not None:
        rotation = eyepacs_rotation(eyepacs_train, seed)
        mean = np.mean([grade_counts(epoch_entries(aptos_train, rotation, e)) for e in range(int(epochs or PROTOCOL["max_epochs"]))], axis=0)
        gap = float(np.abs(class_weights_from_counts(mean) - w).max())
        if gap > WEIGHT_TOLERANCE:
            raise TrainingError(f"the sampled mixture gives weights {gap:.3f} away from the approved ones")
    return [float(x) for x in w], expected.tolist()


def augment(rgb, rng):
    """Random horizontal and vertical flip and quarter turn, then brightness and contrast jitter of +-10 %
    (the random draws are taken in this order)."""
    if rng.random() < 0.5:
        rgb = rgb[:, ::-1, :]
    if rng.random() < 0.5:
        rgb = rgb[::-1, :, :]
    k = int(rng.integers(0, 4))
    if k:
        rgb = np.rot90(rgb, k=k, axes=(0, 1))
    brightness = 1.0 + rng.uniform(-0.1, 0.1)
    contrast = 1.0 + rng.uniform(-0.1, 0.1)
    rgb = np.array(np.ascontiguousarray(rgb), dtype=np.float32, copy=True)
    mean = rgb.mean()
    rgb = (rgb - mean) * contrast + mean
    return np.clip(rgb * brightness, 0.0, 1.0)


def batches(frames, entries, seed, epoch, batch_size, train=True, workers=2, lookahead=4):
    """Yields (rgb (B, S, S, 3) float32, grades (B,) int32) for one pass over `entries`. Training: the epoch's
    permutation and per-image augmentation; otherwise the given order, unaugmented. Images are read by `workers`
    threads, `lookahead` batches ahead."""
    ordered = epoch_order(entries, seed, epoch) if train else [(str(i), int(g)) for i, g in entries]
    chunks = [ordered[s:s + batch_size] for s in range(0, len(ordered), batch_size)]

    def load(row):
        image, _ = row
        rgb = frames.frame(image)
        return augment(rgb, np.random.default_rng(seed_from_key(AUGMENTATION_TAG, seed, epoch, image))) if train else rgb

    out = queue.Queue(maxsize=max(1, lookahead))
    stop = threading.Event()

    def produce():
        try:
            with ThreadPoolExecutor(max(1, workers)) as pool:
                for chunk in chunks:
                    if stop.is_set():
                        return
                    out.put((np.stack(list(pool.map(load, chunk))), np.asarray([g for _, g in chunk], np.int32)))
        except BaseException as error:  # noqa: BLE001 -- handed to the consumer
            out.put(error)
            return
        out.put(None)

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    try:
        while True:
            item = out.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        while not out.empty():
            out.get_nowait()


class JointFrames:
    """`frame(image)` over the APTOS and EyePACS frame caches; image names never coincide (checked)."""

    def __init__(self, aptos, eyepacs):
        self.aptos, self.eyepacs = aptos, eyepacs

    def frame(self, image):
        image = str(image)
        if image in self.aptos:
            return self.aptos.frame(image)
        if image in self.eyepacs:
            return self.eyepacs.frame(image)
        raise TrainingError(f"{image} is in neither frame cache")


# --------------------------------------------------------------------------- model, loss, prediction

def weighted_corn_loss(class_weights_):
    """Keras loss(y_true grades, y_pred CORN logits): sum over images and thresholds k <= grade of
    w_grade * BCE(logit_k, [grade > k]), divided by the number of included (image, threshold) pairs."""
    import tensorflow as tf
    weights = tf.constant(class_weights_, tf.float32)

    def loss(y_true, y_pred):
        logits = tf.cast(y_pred, tf.float32)
        grades = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        k = tf.range(logits.shape[-1], dtype=tf.int32)[None, :]
        mask = tf.cast(grades[:, None] >= k, tf.float32)
        target = tf.cast(grades[:, None] > k, tf.float32)
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits) * mask
        w = tf.gather(weights, tf.clip_by_value(grades, 0, weights.shape[0] - 1))
        return tf.reduce_sum(bce * w[:, None]) / tf.reduce_sum(mask)

    return loss


def build_compiled(seed, imagenet_weights, protocol, class_weights_, image_size=512, mixed_precision=True, model_factory=None):
    """The grader compiled for training. `mixed_precision` is used only when a GPU is present."""
    import keras
    import tensorflow as tf

    from . import grader as gr
    use_mp = bool(mixed_precision and tf.config.list_physical_devices("GPU"))
    keras.mixed_precision.set_global_policy("mixed_float16" if use_mp else "float32")
    model = model_factory() if model_factory else gr.build_model(seed, imagenet_weights, image_size)
    optimizer = keras.optimizers.AdamW(learning_rate=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
    one_d = [v for v in model.trainable_variables if len(v.shape) <= 1]
    optimizer.exclude_from_weight_decay(var_list=one_d, var_names=["bias", "gamma", "beta"])
    on_gpu = bool(tf.config.list_physical_devices("GPU"))
    model.compile(optimizer=optimizer, loss=weighted_corn_loss(class_weights_), jit_compile="auto" if on_gpu else False)
    return model, use_mp


def make_predict(model, image_size=512):
    """A traced rgb -> float32 logits function with a fixed input signature (no retracing, flat memory)."""
    import tensorflow as tf

    @tf.function(input_signature=[tf.TensorSpec([None, image_size, image_size, 3], tf.float32)])
    def predict(x):
        return tf.cast(model(x, training=False), tf.float32)

    return predict


def predict_logits(predict, frames, entries, batch_size=EVAL_BATCH):
    """CORN logits (N, 4) float64 of `entries` in order, unaugmented."""
    out = [np.asarray(predict(rgb), np.float64) for rgb, _ in batches(frames, entries, 0, 0, batch_size, train=False)]
    return np.concatenate(out, 0) if out else np.zeros((0, 4))


def validation_qwk(predict, frames, entries):
    from .grader import decode
    grades = np.asarray([int(g) for _, g in entries])
    pred = decode(predict_logits(predict, frames, entries))["grade"].astype(int)
    value = float(mt.qwk(grades, pred))
    return value if np.isfinite(value) else 0.0          # undefined when references and predictions are one grade: chance level


# --------------------------------------------------------------------------- the run

def inference_weights_saver(factory):
    """A `save_weights(model, path)` that writes only the model's weights (the file `drgrade.grader.Grader` loads,
    about 112 MB): a compiled Keras model's own `save_weights` also stores the two Adam moments (three times the size).
    The weights are copied into one fresh, uncompiled model built by `factory`."""
    clean = factory()

    def save(model, path):
        clean.set_weights(model.get_weights())
        save_weights_atomic(clean, path)

    return save


def train(run_dir, frames, aptos, eyepacs, seed, arm, *, imagenet_weights=None, protocol=None, image_size=512,
          mixed_precision=True, max_epochs=None, steps_per_epoch=None, workers=2, model_factory=None, log=print):
    """Trains (or resumes) one run under `run_dir`; returns the final state dict. `aptos`, `eyepacs`: {'train':
    [(id, grade)], 'val': [...]}. `frames`: object with `frame(image)`. `steps_per_epoch` truncates epochs (smoke
    tests only). The best checkpoint is `run_dir/checkpoints/best.weights.h5`."""
    import keras

    protocol = dict(PROTOCOL if protocol is None else protocol)
    epochs = int(max_epochs or protocol["max_epochs"])
    names = [set(str(i) for i, _ in part) for part in (aptos["train"], aptos["val"], eyepacs["train"], eyepacs["val"])]
    if (names[0] | names[2]) & (names[1] | names[3]):
        raise TrainingError("a validation image is in a training part")
    weights, expected = class_weights(aptos["train"], eyepacs["train"], seed, epochs)
    identity = {"seed": int(seed), "arm": arm, "protocol": protocol, "class_weights": weights, "image_size": int(image_size),
                "counts": {"aptos_train": len(aptos["train"]), "eyepacs_train": len(eyepacs["train"]),
                           "aptos_val": len(aptos["val"]), "eyepacs_val": len(eyepacs["val"])}}
    os.makedirs(run_dir, exist_ok=True)
    config_path, state_path = os.path.join(run_dir, "run_config.json"), os.path.join(run_dir, "state.json")
    if os.path.exists(config_path):
        if read_json(config_path) != json.loads(json.dumps(identity)):
            raise TrainingError(f"{run_dir} holds a run with another configuration; use a new folder")
    else:
        atomic_json(config_path, identity)

    keras.backend.clear_session()
    model, used_mp = build_compiled(seed, imagenet_weights, protocol, weights, image_size, mixed_precision, model_factory)
    predict = make_predict(model, image_size)
    saver = inference_weights_saver(model_factory or (lambda: gr.build_model(seed, None, image_size)))
    rotation = eyepacs_rotation(eyepacs["train"], seed)
    controller = PlateauController("max", protocol["stop_patience"], protocol["lr_patience"], protocol["lr_factor"], protocol["min_lr"])
    state = {"completed_epochs": 0, "lr": protocol["learning_rate"], "best": None, "history": [], "stopped": None}
    if os.path.exists(state_path):
        state = read_json(state_path)
        controller.load_state_dict(state["controller"])
        load_generation(run_dir, model, state["last_weights"], state["last_optimizer"])
        log(f"resumed after epoch index {state['completed_epochs'] - 1} (0-based); learning rate {state['lr']:.2e}")
    if state["stopped"]:
        log(f"run already finished: {state['stopped']}")
        return state
    log(f"grader {arm} seed {seed}: mixed precision {used_mp}, class weights {np.round(weights, 4).tolist()}")

    for epoch in range(state["completed_epochs"], epochs):
        set_learning_rate(model.optimizer, state["lr"])
        started, losses = time.time(), []
        entries = epoch_entries(aptos["train"], rotation, epoch)
        for step, (rgb, grades) in enumerate(batches(frames, entries, seed, epoch, protocol["batch_size"], True, workers)):
            if steps_per_epoch and step >= steps_per_epoch:
                break
            losses.append(float(model.train_on_batch(rgb, grades)))
            if (step + 1) % 100 == 0:
                log(f"  epoch index {epoch} step {step + 1}: loss {np.mean(losses[-100:]):.4f}")
        if not losses or not np.all(np.isfinite(losses)):
            raise TrainingError(f"epoch index {epoch}: the training loss is not finite ({losses[:3]}...)")
        a, e = validation_qwk(predict, frames, aptos["val"]), validation_qwk(predict, frames, eyepacs["val"])
        selection = 0.5 * (a + e)
        decision = controller.update(epoch, selection, state["lr"])
        if decision["improved"]:
            best = os.path.join(run_dir, "checkpoints", "best.weights.h5")
            os.makedirs(os.path.dirname(best), exist_ok=True)
            saver(model, best)
            state["best"] = {"epoch_index": epoch, "selection": selection, "aptos_val_qwk": a, "eyepacs_val_qwk": e}
        state["history"].append({"epoch_index": epoch, "train_loss": float(np.mean(losses)), "steps": len(losses), "aptos_val_qwk": a,
                                 "eyepacs_val_qwk": e, "selection": selection, "lr": state["lr"], "seconds": round(time.time() - started, 1)})
        state["lr"], state["completed_epochs"] = decision["lr"], epoch + 1
        state["controller"] = controller.state_dict()
        if decision["stop"]:
            state["stopped"] = f"early stopping after epoch index {epoch} (best epoch index {controller.best_epoch})"
        elif epoch + 1 >= protocol["max_epochs"]:
            state["stopped"] = f"epoch cap after epoch index {epoch}"
        state["last_weights"], state["last_optimizer"] = save_generation(run_dir, model, epoch, saver)
        atomic_json(state_path, state)                       # written last: it commits the generation
        prune_generations(run_dir, {state["last_weights"], state["last_optimizer"]})
        log(f"epoch index {epoch}: loss {np.mean(losses):.4f} | APTOS val QWK {a:.4f} | EyePACS val QWK {e:.4f} | selection {selection:.4f}"
            f"{' (best)' if decision['improved'] else ''} | lr {state['lr']:.2e} | {time.time() - started:.0f} s{rss_text()}")
        if state["stopped"]:
            log(state["stopped"])
            break
    return state


def export_best(run_dir, out_path):
    """Copies the best checkpoint to `out_path` (the file `grader_<arm>_seed<seed>.weights.h5` of the release)."""
    src = os.path.join(run_dir, "checkpoints", "best.weights.h5")
    if not os.path.exists(src):
        raise TrainingError(f"{src} does not exist: no epoch has been scored yet")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    shutil.copyfile(src, out_path)
    return out_path
