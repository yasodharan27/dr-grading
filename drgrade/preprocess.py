"""Stage 2 preprocessing and the grader's input frames.

    raw image -> gamma correction (1.2) -> CLAHE on the L channel of LAB (clip 2.0, 8 x 8 tiles)   = Stage-2 image
    Stage-2 image -> [crop arm: cut to the fundus bounding box] -> resize to 512 x 512, float32 in [0, 1] = frame

The fundus box is found on the RAW image: a pixel is lit when its grey value exceeds 12; a row (column) belongs
to the fundus when more than 2 % of its pixels are lit; a box narrower than 10 % of the image side counts as a
failure and the whole image is kept.
"""
import numpy as np

GAMMA = 1.2
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
FRAME_SIZE = 512
GRAY_THRESHOLD = 12
LINE_FRACTION = 0.02
MIN_SIDE_FRACTION = 0.10


def read_bgr(path):
    import cv2
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"cannot read the image {path}")
    return bgr


def gamma_correction(image, gamma=GAMMA):
    """Power-law tone curve through a 256-entry lookup table (gamma > 1 brightens mid-tones)."""
    import cv2
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def clahe(bgr, clip_limit=CLAHE_CLIP_LIMIT, tile_grid=CLAHE_TILE_GRID):
    """Contrast-limited adaptive histogram equalisation of the lightness channel only (hue is kept)."""
    import cv2
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(l_channel)
    return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def stage2(path):
    """The Stage-2 image of one raw file: RGB uint8 at the native size."""
    import cv2
    return cv2.cvtColor(clahe(gamma_correction(read_bgr(path))), cv2.COLOR_BGR2RGB)


def fundus_box(bgr):
    """(top, bottom, left, right) of the fundus in a RAW BGR image, bottom / right exclusive, or the whole image
    when no usable box is found."""
    import cv2
    lit = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) > GRAY_THRESHOLD
    h, w = lit.shape
    rows = np.flatnonzero(lit.mean(axis=1) > LINE_FRACTION)
    cols = np.flatnonzero(lit.mean(axis=0) > LINE_FRACTION)
    if rows.size == 0 or cols.size == 0:
        return 0, h, 0, w
    top, bottom, left, right = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    if bottom - top < MIN_SIDE_FRACTION * h or right - left < MIN_SIDE_FRACTION * w:
        return 0, h, 0, w
    return top, bottom, left, right


def resize_frame(rgb, size=FRAME_SIZE):
    """RGB uint8 (any size) -> float32 (size, size, 3) in [0, 1]: bilinear, anti-aliased, the aspect ratio is not
    kept (the whole image fills the frame)."""
    from skimage.transform import resize
    x = np.asarray(rgb, dtype=np.float32) / 255.0
    out = resize(x, (size, size, 3), order=1, mode="reflect", anti_aliasing=True, preserve_range=True)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def frames(path):
    """Everything the later stages need from one raw image, with Stage 2 run once:
    {'stage2': RGB uint8 native, 'box': fundus box, 'direct': frame, 'crop': frame}."""
    native = stage2(path)
    bgr = read_bgr(path)
    if bgr.shape[:2] != native.shape[:2]:
        raise ValueError(f"{path}: the raw image and its Stage-2 image differ in size")
    box = fundus_box(bgr)
    top, bottom, left, right = box
    direct = resize_frame(native)
    whole = box == (0, native.shape[0], 0, native.shape[1])
    crop = direct if whole else resize_frame(np.ascontiguousarray(native[top:bottom, left:right]))
    return {"stage2": native, "box": box, "direct": direct, "crop": crop}


def frame(path, arm):
    """The grader's input frame of one raw image for `arm` ('crop' or 'direct')."""
    if arm not in ("crop", "direct"):
        raise ValueError(f"arm must be 'crop' or 'direct', got {arm!r}")
    return frames(path)[arm]


def resize_with_extra_channel(rgb, extra, size=FRAME_SIZE):
    """RGB (uint8 or [0, 1]) and one extra map of the same size resized TOGETHER to (size, size, 4), the way the
    vessel map is brought to the frame (one joint resize, so both share the same interpolation)."""
    from skimage.transform import resize
    x = np.asarray(rgb, dtype=np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    extra = np.asarray(extra, dtype=np.float32)
    if extra.ndim == 2:
        extra = extra[..., None]
    combined = np.concatenate([x, extra], axis=-1)
    return resize(combined, (size, size, combined.shape[-1]), order=1, mode="reflect", anti_aliasing=True,
                  preserve_range=True).astype(np.float32)
