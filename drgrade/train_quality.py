"""Training and evaluation of Stage 1, image quality: EfficientNet-B0 on EyeQ (Good / Usable / Reject).

Protocol: the EyeQ training images are split 85 / 15 into training and validation (stratified by quality, seed 42);
images are decoded and resized to 224 x 224 (pixel values 0-255, the network normalises them); EfficientNet-B0
(ImageNet weights, its first 100 layers frozen) -> global average pooling -> Dense(256, relu) -> Dropout(0.3) ->
Dense(3, softmax); Adam (learning rate 1e-4), categorical cross-entropy with per-class weights N / (3 n_c) from the
training part, batch 32, at most 50 epochs; the monitored value is the validation loss: the learning rate is halved
after 4 epochs without improvement (down to 1e-6), training stops after 8, and the checkpoint with the lowest
validation loss is kept. The EyeQ test split is read only by `evaluate_test`.

Resumable after every epoch, like the grader's trainer. Mixed precision is used on GPU.
"""
import os
import time

import numpy as np

from . import data, quality
from .training import (PlateauController, atomic_json, load_generation, prune_generations, read_json, rss_text, save_generation, set_learning_rate)

PROTOCOL = {"batch_size": 32, "max_epochs": 50, "learning_rate": 1e-4, "freeze_layers": 100, "val_split": 0.15, "seed": 42,
            "stop_patience": 8, "lr_patience": 4, "lr_factor": 0.5, "min_lr": 1e-6}
NUM_CLASSES = len(quality.CLASSES)


class QualityTrainingError(RuntimeError):
    pass


def split_training_images(table, val_split=0.15, seed=42):
    """Stratified (train part, validation part) of the EyeQ training table."""
    from sklearn.model_selection import train_test_split
    train, val = train_test_split(table, test_size=val_split, stratify=table["quality"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True)


def class_weights(train_part):
    counts = train_part["quality"].value_counts().sort_index()
    total = float(counts.sum())
    return {int(c): total / (NUM_CLASSES * n) for c, n in counts.items()}


def make_dataset(paths, labels, batch_size, image_size=quality.IMAGE_SIZE):
    """tf.data pipeline of (RGB float32 0-255, one-hot label) in the given order."""
    import tensorflow as tf
    onehot = tf.keras.utils.to_categorical(np.asarray(labels, int), NUM_CLASSES)
    ds = tf.data.Dataset.from_tensor_slices((np.asarray(paths), onehot))
    ds = ds.map(lambda p, y: (quality.decode_image(p, image_size), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(input_shape=(224, 224, 3), learning_rate=1e-4, freeze_layers=100, weights="imagenet", mixed_precision=True):
    """The compiled Stage-1 model. `weights=None` builds it untrained (tests)."""
    import keras
    import tensorflow as tf
    from keras import Input, Model, layers
    from keras.applications import EfficientNetB0
    on_gpu = bool(tf.config.list_physical_devices("GPU"))
    keras.mixed_precision.set_global_policy("mixed_float16" if (mixed_precision and on_gpu) else "float32")
    inputs = Input(shape=input_shape)
    base = EfficientNetB0(include_top=False, weights=weights, input_shape=input_shape)
    for layer in base.layers[:freeze_layers]:
        layer.trainable = False
    x = layers.GlobalAveragePooling2D()(base(inputs))
    x = layers.Dropout(0.3)(layers.Dense(256, activation="relu")(x))
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
    model = Model(inputs, outputs, name="quality_efficientnet_b0")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="categorical_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")], jit_compile="auto" if on_gpu else False)
    return model


def train(run_dir, cfg, *, protocol=None, limit=None, max_epochs=None, weights="imagenet", log=print):
    """Trains (or resumes) Stage 1. `limit`: use only the first N training images (smoke tests). The best model
    (lowest validation loss) is saved as `<run_dir>/best_model.keras`; returns the state dict."""
    import keras

    protocol = dict(PROTOCOL if protocol is None else protocol)
    epochs = int(max_epochs or protocol["max_epochs"])
    table = data.eyeq_table(cfg, "train")
    if limit:
        table = table.sample(n=min(limit, len(table)), random_state=protocol["seed"]).reset_index(drop=True)
    train_part, val_part = split_training_images(table, protocol["val_split"], protocol["seed"])
    weights_by_class = class_weights(train_part)
    identity = {"protocol": protocol, "train_images": len(train_part), "val_images": len(val_part), "class_weights": weights_by_class}
    os.makedirs(run_dir, exist_ok=True)
    config_path, state_path = os.path.join(run_dir, "run_config.json"), os.path.join(run_dir, "state.json")
    import json
    if os.path.exists(config_path):
        if read_json(config_path) != json.loads(json.dumps(identity)):
            raise QualityTrainingError(f"{run_dir} holds a run with another configuration; use a new folder")
    else:
        atomic_json(config_path, identity)
    log(f"EyeQ: {len(train_part)} training / {len(val_part)} validation images; class weights "
        f"{ {k: round(v, 3) for k, v in weights_by_class.items()} }")

    keras.backend.clear_session()
    keras.utils.set_random_seed(protocol["seed"])
    model = build_model(learning_rate=protocol["learning_rate"], freeze_layers=protocol["freeze_layers"], weights=weights)
    val_ds = make_dataset(val_part["path"], val_part["quality"], protocol["batch_size"])
    controller = PlateauController("min", protocol["stop_patience"], protocol["lr_patience"], protocol["lr_factor"], protocol["min_lr"])
    state = {"completed_epochs": 0, "lr": protocol["learning_rate"], "best": None, "history": [], "stopped": None}
    if os.path.exists(state_path):
        state = read_json(state_path)
        controller.load_state_dict(state["controller"])
        load_generation(run_dir, model, state["last_weights"], state["last_optimizer"])
        log(f"resumed after epoch index {state['completed_epochs'] - 1} (0-based)")
    if state["stopped"]:
        log(f"run already finished: {state['stopped']}")
        return state

    for epoch in range(state["completed_epochs"], epochs):
        set_learning_rate(model.optimizer, state["lr"])
        started = time.time()
        order = np.random.default_rng([protocol["seed"], epoch]).permutation(len(train_part))     # a pure function of (seed, epoch)
        shuffled = train_part.iloc[order]
        losses = []
        for x, y in make_dataset(shuffled["path"], shuffled["quality"], protocol["batch_size"]):
            sample_weight = np.asarray([weights_by_class[int(k)] for k in np.argmax(y.numpy(), axis=1)], np.float32)
            losses.append(float(model.train_on_batch(x, y, sample_weight=sample_weight)[0]))
        if not losses or not np.all(np.isfinite(losses)):
            raise QualityTrainingError(f"epoch index {epoch}: the training loss is not finite")
        val_loss, val_accuracy, val_auc = (float(v) for v in model.evaluate(val_ds, verbose=0))
        decision = controller.update(epoch, val_loss, state["lr"])
        if decision["improved"]:
            model.save(os.path.join(run_dir, "best_model.keras"))
            state["best"] = {"epoch_index": epoch, "val_loss": val_loss, "val_accuracy": val_accuracy, "val_auc": val_auc}
        state["history"].append({"epoch_index": epoch, "train_loss": float(np.mean(losses)), "steps": len(losses), "val_loss": val_loss,
                                 "val_accuracy": val_accuracy, "val_auc": val_auc, "lr": state["lr"], "seconds": round(time.time() - started, 1)})
        state["lr"], state["completed_epochs"] = decision["lr"], epoch + 1
        state["controller"] = controller.state_dict()
        if decision["stop"]:
            state["stopped"] = f"early stopping after epoch index {epoch} (best epoch index {controller.best_epoch})"
        elif epoch + 1 >= protocol["max_epochs"]:
            state["stopped"] = f"epoch cap after epoch index {epoch}"
        state["last_weights"], state["last_optimizer"] = save_generation(run_dir, model, epoch)
        atomic_json(state_path, state)
        prune_generations(run_dir, {state["last_weights"], state["last_optimizer"]})
        log(f"epoch index {epoch}: loss {np.mean(losses):.4f} | val loss {val_loss:.4f} | val accuracy {val_accuracy:.4f} | "
            f"val AUC {val_auc:.4f}{' (best)' if decision['improved'] else ''} | lr {state['lr']:.2e} | {time.time() - started:.0f} s{rss_text()}")
        if state["stopped"]:
            log(state["stopped"])
            break
    return state


def evaluate_test(model_path, cfg, limit=None, batch_size=32, out_path=None, log=print):
    """Accuracy, per-class precision / recall / F1 and the confusion matrix of a saved model on the EyeQ TEST split
    (16,249 images; `limit`: a random subset for smoke tests)."""
    from sklearn.metrics import classification_report, confusion_matrix
    model = quality.load_model(model_path)
    table = data.eyeq_table(cfg, "test")
    if limit:
        table = table.sample(n=min(limit, len(table)), random_state=0).reset_index(drop=True)
    probabilities = []
    for x, _ in make_dataset(table["path"], table["quality"], batch_size):
        probabilities.append(np.asarray(model.predict_on_batch(x)))
    pred = np.argmax(np.concatenate(probabilities), axis=1)
    labels = table["quality"].to_numpy(int)
    report = classification_report(labels, pred, labels=list(range(NUM_CLASSES)), target_names=list(quality.CLASSES), output_dict=True, zero_division=0)
    result = {"images": int(len(table)), "accuracy": float((pred == labels).mean()), "report": report,
              "confusion_matrix": confusion_matrix(labels, pred, labels=list(range(NUM_CLASSES))).tolist()}
    log(f"EyeQ test: {result['images']} images, accuracy {result['accuracy']:.4f}")
    if out_path:
        atomic_json(out_path, result)
    return result
