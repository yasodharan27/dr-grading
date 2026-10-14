"""The grader: ConvNeXt-Tiny (ImageNet initialised) + CORN ordinal head, on the 512 x 512 RGB frame.

    frame (512, 512, 3) in [0, 1] -> (x - ImageNet mean) / ImageNet std -> ConvNeXt-Tiny -> feature map
    (16 x 16 x 768) -> global average pooling -> LayerNorm -> Dense(768 -> 4) = CORN logits z_0..z_3

CORN (Shi et al., 2023): sigmoid(z_k) = P(grade > k | grade >= k); the cumulative probabilities are their running
products, P(grade > k) = prod_{i <= k} sigmoid(z_i), and the grade is the number of cumulative probabilities above
0.5. Class probabilities are the differences of consecutive cumulative probabilities.
"""
import numpy as np

IMAGE_SIZE = 512
FEATURE_DIM = 768
NUM_THRESHOLDS = 4
NUM_GRADES = 5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BACKBONE_NAME = "convnext_tiny"


# --------------------------------------------------------------------------- CORN (numpy)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


def cumulative(logits):
    """P(grade > k), k = 0..3, from CORN logits (N, 4)."""
    return np.cumprod(sigmoid(logits), axis=-1)


def class_probabilities(p_gt):
    """(N, 5) from cumulative probabilities (N, 4): 1 - p_0, p_0 - p_1, ..., p_3."""
    p_gt = np.asarray(p_gt, np.float64)
    return np.concatenate([1.0 - p_gt[..., :1], p_gt[..., :-1] - p_gt[..., 1:], p_gt[..., -1:]], axis=-1)


def decode(logits):
    """The project's decode of CORN logits (N, 4): {'p_cond', 'p_gt', 'grade', 'class_prob'}. Probabilities are
    returned in float32, the precision the per-image tables store."""
    logits = np.asarray(logits, np.float64)
    p_cond = sigmoid(logits)
    p_gt = np.cumprod(p_cond, axis=-1)
    return {"p_cond": p_cond.astype(np.float32), "p_gt": p_gt.astype(np.float32),
            "grade": np.sum(p_gt > 0.5, axis=-1).astype(np.int64), "class_prob": class_probabilities(p_gt).astype(np.float32)}


def expected_grade(logits):
    """E[grade] = sum_k P(grade > k)."""
    return cumulative(logits).sum(axis=-1)


# --------------------------------------------------------------------------- the Keras graph

def _normalization_layer():
    from keras import layers, ops

    class ImageNetNormalization(layers.Layer):
        """(x - mean) / std for RGB in [0, 1]: the same values Keras' ConvNeXt preprocessing gives for x * 255."""

        def call(self, x):
            mean = ops.convert_to_tensor(IMAGENET_MEAN, dtype=x.dtype)
            std = ops.convert_to_tensor(IMAGENET_STD, dtype=x.dtype)
            return (x - mean) / std

    return ImageNetNormalization


def build_model(seed=42, imagenet_weights=None, image_size=IMAGE_SIZE):
    """rgb (B, S, S, 3) -> CORN logits (B, 4). Build under the dtype policy the run uses (set it first).
    `imagenet_weights`: the Keras ConvNeXt-Tiny `notop` file to initialise the backbone from (training); the CORN
    head starts as GlorotUniform(seed) with zero bias. Without it the backbone is random (load a checkpoint next)."""
    import keras
    from keras import layers
    keras.utils.set_random_seed(int(seed))
    rgb = keras.Input((image_size, image_size, 3), name="rgb")
    backbone = keras.applications.ConvNeXtTiny(include_top=False, weights=None, include_preprocessing=False, pooling="avg",
                                               input_shape=(image_size, image_size, 3), name=BACKBONE_NAME)
    features = keras.Input((FEATURE_DIM,), name="F")
    head = keras.Model(features, layers.Dense(NUM_THRESHOLDS, name="corn_logits")(features), name="corn")
    kernel = keras.initializers.GlorotUniform(seed=int(seed))((FEATURE_DIM, NUM_THRESHOLDS))
    head.get_layer("corn_logits").set_weights([np.asarray(kernel, np.float32), np.zeros(NUM_THRESHOLDS, np.float32)])
    if imagenet_weights:
        copy_imagenet(backbone, imagenet_weights, image_size)
    logits = head(backbone(_normalization_layer()(name="imagenet_normalization")(rgb)))
    return keras.Model(rgb, logits, name="grader")


def copy_imagenet(backbone, path, image_size=IMAGE_SIZE):
    """Copies the ImageNet arrays of Keras' ConvNeXt-Tiny into `backbone`, positionally with shape checks (its
    LayerScale weights are unnamed, so names cannot be matched); the reference's preprocessing variables are
    skipped because `backbone` has no preprocessing layer."""
    import keras
    reference = keras.applications.ConvNeXtTiny(include_top=False, weights=None, include_preprocessing=True, pooling="avg",
                                                input_shape=(image_size, image_size, 3), name=BACKBONE_NAME)
    reference.load_weights(path)
    arrays = [np.asarray(v.numpy()) for v in reference.weights if "prestem" not in v.path]
    if len(arrays) != len(backbone.weights):
        raise RuntimeError(f"{len(backbone.weights)} backbone weights vs {len(arrays)} ImageNet arrays")
    for variable, array in zip(backbone.weights, arrays):
        if tuple(variable.shape) != array.shape:
            raise RuntimeError(f"shape mismatch at {variable.path}: {tuple(variable.shape)} vs {array.shape}")
        variable.assign(array.astype(variable.dtype))
    del reference


class Grader:
    """One trained grader checkpoint, float32. `logits(frames)` for prediction; `grad_model` maps rgb to
    (feature map (B, 16, 16, 768), logits) for Grad-CAM and Integrated Gradients (same layers, same weights)."""

    def __init__(self, weights_path, image_size=IMAGE_SIZE):
        import keras
        keras.mixed_precision.set_global_policy("float32")
        self.model = build_model(image_size=image_size)
        self.model.load_weights(weights_path)
        self.model.trainable = False
        backbone = self.model.get_layer(BACKBONE_NAME)
        pooling = next(l for l in backbone.layers if isinstance(l, keras.layers.GlobalAveragePooling2D))
        inner = keras.Model(backbone.inputs, [pooling.input, backbone.output], name="backbone_with_features")
        rgb = keras.Input((image_size, image_size, 3), name="rgb")
        normalised = self.model.get_layer("imagenet_normalization")(rgb)
        feature_map, pooled = inner(normalised)
        self.grad_model = keras.Model(rgb, [feature_map, self.model.get_layer("corn")(pooled)], name="grader_explainer")
        self.path = weights_path

    def logits(self, frames, batch_size=8):
        """CORN logits (N, 4) float64 of frames (N, 512, 512, 3) float32 in [0, 1]."""
        frames = np.asarray(frames, np.float32)
        return np.concatenate([np.asarray(self.model.predict_on_batch(frames[s:s + batch_size]), np.float64)
                               for s in range(0, len(frames), batch_size)], 0)

    def __call__(self, frames):
        return self.logits(frames)


def load_arm(cfg, arm, seeds=None):
    """{seed: Grader} of one arm's three checkpoints (one arm at a time keeps memory at about 1.5 GB)."""
    from .config import SEEDS, weight_path
    return {seed: Grader(weight_path(cfg, f"grader_{arm}_{seed}")) for seed in (seeds or SEEDS)}
