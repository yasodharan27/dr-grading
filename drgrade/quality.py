"""Stage 1 -- image quality (EfficientNet-B0 trained on EyeQ: Good / Usable / Reject).

The model sees the RAW image (no Stage-2 preprocessing) resized to 224 x 224, pixel values in [0, 255]; the
EfficientNet-B0 graph normalises them itself.
"""
import numpy as np

CLASSES = ("Good", "Usable", "Reject")
IMAGE_SIZE = (224, 224)


def load_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path)


def decode_image(path, image_size=IMAGE_SIZE):
    """RGB float32 (H, W, 3) in [0, 255], resized as in training (tf.image.resize, bilinear)."""
    import tensorflow as tf
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)        # decodes PNG as well
    return tf.cast(tf.image.resize(image, image_size), tf.float32)


def result_from_probabilities(probabilities):
    k = int(np.argmax(probabilities))
    return {"label": CLASSES[k], "class_index": k, "confidence": float(probabilities[k]),
            "probabilities": {name: float(p) for name, p in zip(CLASSES, probabilities)}}


def predict(model, path):
    """{'label', 'class_index', 'confidence', 'probabilities'} of one image file."""
    probabilities = model.predict(decode_image(path)[None], verbose=0)[0]
    return result_from_probabilities(probabilities)
