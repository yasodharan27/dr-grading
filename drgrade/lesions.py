"""Stage 4 -- lesion segmentation: U-Net with an SE-ResNet-101 encoder (segmentation_models_pytorch 0.5.0),
four sigmoid outputs in the order MA (microaneurysms), HE (haemorrhages), EX (hard exudates), SE (soft exudates).

Geometry: the whole native Stage-2 image is resized to 1536 x 1536 (three times the 512 frame), segmented in
one pass, and each class map is pooled over exact 3 x 3 blocks to 512 x 512 -- once by mean, once by max. The
maps are stored as uint8 (probability x 255, rounded) with the channels interleaved per class:
[MA:mean, MA:max, HE:mean, HE:max, EX:mean, EX:max, SE:mean, SE:max].
"""
import numpy as np

CLASSES = ("MA", "HE", "EX", "SE")
NAMES = {"MA": "microaneurysms", "HE": "haemorrhages", "EX": "hard exudates", "SE": "soft exudates"}
POOLINGS = ("mean", "max")
ENCODER = "se_resnet101"
DECODER_CHANNELS = (256, 128, 64, 32, 16)
INFERENCE_SIZE = 1536
FRAME_SIZE = 512
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def channel_names(classes=CLASSES):
    return tuple(f"{c}:{p}" for c in classes for p in POOLINGS)


def build_model(classes=CLASSES):
    """The U-Net, randomly initialised (training loads the ImageNet encoder separately; inference loads a
    trained state dict)."""
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=ENCODER, encoder_weights=None, decoder_channels=DECODER_CHANNELS, in_channels=3,
                    classes=len(classes), activation=None)


def load_model(path, device="cpu", classes=CLASSES):
    import torch
    model = build_model(classes)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def resize_full_frame(rgb, size=INFERENCE_SIZE):
    """Whole image -> (size, size, 3) float32 in [0, 1]; the same resize as the 512 frame, at a larger size."""
    from skimage.transform import resize
    x = np.asarray(rgb, dtype=np.float32)
    if np.asarray(rgb).dtype == np.uint8 or x.max() > 1.0:
        x = x / 255.0
    out = resize(x, (size, size, x.shape[-1]), order=1, mode="reflect", anti_aliasing=True, preserve_range=True)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def to_model_input(rgb01):
    """(H, W, 3) in [0, 1] -> (1, 3, H, W) tensor, ImageNet mean / std normalised."""
    import torch
    x = (np.asarray(rgb01, np.float32) - np.asarray(IMAGENET_MEAN, np.float32)) / np.asarray(IMAGENET_STD, np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]


def predict_probabilities(model, rgb01, device="cpu", amp=False):
    """(H, W, 3) in [0, 1] -> (H, W, K) float32 probabilities, whole image in one pass (no tiling)."""
    import torch
    model.eval()
    x = to_model_input(rgb01).to(device)
    with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=bool(amp)):
        probs = torch.sigmoid(model(x).float())
    return probs[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)


def pool_mean_max(probabilities, factor):
    """Exact non-overlapping factor x factor block mean and max of (H, W, K) maps."""
    p = np.asarray(probabilities, dtype=np.float32)
    h, w, k = p.shape
    if h % factor or w % factor:
        raise ValueError(f"{p.shape} is not divisible by {factor}")
    blocks = p.reshape(h // factor, factor, w // factor, factor, k)
    return blocks.mean(axis=(1, 3), dtype=np.float64).astype(np.float32), blocks.max(axis=(1, 3))


def to_uint8(p):
    p = np.asarray(p, dtype=np.float32)
    if not np.all(np.isfinite(p)):
        raise ValueError("probabilities contain non-finite values")
    return np.rint(np.clip(p, 0.0, 1.0) * 255.0).astype(np.uint8)


def frame_maps(model, stage2_rgb, device="cpu", amp=False, size=INFERENCE_SIZE, frame_size=FRAME_SIZE):
    """Native Stage-2 image -> uint8 (512, 512, 2K) maps, channels as `channel_names()`."""
    if size % frame_size:
        raise ValueError(f"inference size {size} is not a multiple of the frame size {frame_size}")
    probs = predict_probabilities(model, resize_full_frame(stage2_rgb, size), device, amp)
    mean, maximum = pool_mean_max(probs, size // frame_size)
    stacked = np.stack([mean, maximum], axis=-1)                         # (h, w, K, 2) -> interleaved per class
    return to_uint8(stacked.reshape(*stacked.shape[:2], -1))


def max_maps(maps):
    """(512, 512, K) float32 in [0, 1]: the per-class max maps of `frame_maps`."""
    return np.asarray(maps)[..., 1::2].astype(np.float32) / 255.0
