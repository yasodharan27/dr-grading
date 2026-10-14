"""Stage 3 -- retinal vessel segmentation with the pretrained LWNet (inference only, not trained here).

Model and checkpoint: A. Galdran et al., "The Little W-Net That Could: State-of-the-Art Retinal Vessel
Segmentation with Minimalistic Models" (arXiv:2009.01907), https://github.com/agaldran/lwnet, MIT License
(see LICENSES/LWNet-MIT.txt). The checkpoint is the authors' `experiments/wnet_drive/model_checkpoint.pth`,
trained by them on DRIVE. The architecture below is ported from their `models/get_model.py` (`wnet`) and
`models/res_unet_adrian.py`; the field-of-view detection, crop, test-time augmentation and paste-back are ported
from their `predict_one_image.py` (with `skimage.draw.circle`, removed from scikit-image, replaced by
`skimage.draw.disk`).

Here the model runs on the Stage-2 image, which differs from the DRIVE images it was trained on; the map is a
supporting view and does not feed the grade.
"""
import numpy as np

IN_CHANNELS, NUM_CLASSES, LAYERS = 3, 1, (8, 16, 32)
MODEL_INPUT_SIZE = 512
FOV_WORKING_SIZE = 500


# --------------------------------------------------------------------------- architecture (W-Net = two U-Nets)

def _modules():
    import torch
    import torch.nn as nn

    def conv1x1(in_planes, out_planes):
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)

    class ConvBlock(nn.Module):
        def __init__(self, in_c, out_c, k_sz=3, shortcut=False, pool=True):
            super().__init__()
            self.shortcut = nn.Sequential(conv1x1(in_c, out_c), nn.BatchNorm2d(out_c)) if shortcut else False
            pad = (k_sz - 1) // 2
            self.pool = nn.MaxPool2d(kernel_size=2) if pool else False
            self.block = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=k_sz, padding=pad), nn.ReLU(), nn.BatchNorm2d(out_c),
                                       nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding=pad), nn.ReLU(), nn.BatchNorm2d(out_c))

        def forward(self, x):
            if self.pool:
                x = self.pool(x)
            out = self.block(x)
            return out + self.shortcut(x) if self.shortcut else out

    class UpsampleBlock(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.block = nn.Sequential(nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2))   # key names: block.0.*

        def forward(self, x):
            return self.block(x)

    class ConvBridgeBlock(nn.Module):
        def __init__(self, channels, k_sz=3):
            super().__init__()
            self.block = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=k_sz, padding=(k_sz - 1) // 2), nn.ReLU(),
                                       nn.BatchNorm2d(channels))

        def forward(self, x):
            return self.block(x)

    class UpConvBlock(nn.Module):
        def __init__(self, in_c, out_c, k_sz=3, conv_bridge=False, shortcut=False):
            super().__init__()
            self.conv_bridge = conv_bridge
            self.up_layer = UpsampleBlock(in_c, out_c)
            self.conv_layer = ConvBlock(2 * out_c, out_c, k_sz=k_sz, shortcut=shortcut, pool=False)
            if conv_bridge:
                self.conv_bridge_layer = ConvBridgeBlock(out_c, k_sz=k_sz)

        def forward(self, x, skip):
            up = self.up_layer(x)
            skip = self.conv_bridge_layer(skip) if self.conv_bridge else skip
            return self.conv_layer(torch.cat([up, skip], dim=1))

    class UNet(nn.Module):
        def __init__(self, in_c, n_classes, layers, k_sz=3, conv_bridge=True, shortcut=True):
            super().__init__()
            self.first = ConvBlock(in_c, layers[0], k_sz=k_sz, shortcut=shortcut, pool=False)
            self.down_path = nn.ModuleList([ConvBlock(layers[i], layers[i + 1], k_sz=k_sz, shortcut=shortcut, pool=True)
                                            for i in range(len(layers) - 1)])
            rev = list(reversed(layers))
            self.up_path = nn.ModuleList([UpConvBlock(rev[i], rev[i + 1], k_sz=k_sz, conv_bridge=conv_bridge, shortcut=shortcut)
                                          for i in range(len(layers) - 1)])
            self.final = nn.Conv2d(layers[0], n_classes, kernel_size=1)

        def forward(self, x):
            x = self.first(x)
            skips = []
            for down in self.down_path:
                skips.append(x)
                x = down(x)
            for up, skip in zip(self.up_path, reversed(skips)):
                x = up(x, skip)
            return self.final(x)

    class WNet(nn.Module):
        """unet2 refines unet1: its input is the image concatenated with unet1's raw logits."""

        def __init__(self):
            super().__init__()
            self.unet1 = UNet(IN_CHANNELS, NUM_CLASSES, LAYERS)
            self.unet2 = UNet(IN_CHANNELS + NUM_CLASSES, NUM_CLASSES, LAYERS)

        def forward(self, x):
            return self.unet2(torch.cat([x, self.unet1(x)], dim=1))

    return WNet


def load_model(path, device="cpu"):
    """The W-Net with the authors' DRIVE checkpoint (its `model_state_dict`), in eval mode."""
    import torch
    model = _modules()()
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


# --------------------------------------------------------------------------- field of view (circle fit)

def _fit_circle(binary_mask):
    from scipy.optimize import fmin
    from skimage.draw import disk
    from skimage.measure import regionprops
    labeled = binary_mask.astype(int)
    regions = regionprops(labeled)
    if not regions:
        raise ValueError("no field of view found: the thresholded image has no foreground")
    row0, col0 = regions[0].centroid
    radius0 = regions[0].axis_major_length / 2.0

    def cost(params):
        col, row, r = params
        rr, cc = disk((row, col), r, shape=labeled.shape)
        template = np.zeros_like(labeled)
        template[rr, cc] = 1
        return -np.sum(template == labeled)

    return fmin(cost, (col0, row0, radius0), disp=False)


def fov_mask(pil_image):
    """Boolean field-of-view mask at the image's own size: contrast equalisation, minimum threshold, hole filling
    and a circle fit (on a copy whose shorter side is 500 px when the image is larger)."""
    from scipy.ndimage import binary_fill_holes
    from skimage.exposure import equalize_adapthist
    from skimage.filters import threshold_minimum
    from skimage.transform import resize
    from torchvision import transforms
    width, height = pil_image.size
    working = transforms.Resize(FOV_WORKING_SIZE)(pil_image) if max(width, height) > FOV_WORKING_SIZE else pil_image
    with np.errstate(divide="ignore"):
        value = equalize_adapthist(np.array(working))[:, :, 1]
    binary = binary_fill_holes(value > threshold_minimum(value))
    col0, row0, radius = _fit_circle(binary)
    y, x = np.ogrid[:binary.shape[0], :binary.shape[1]]
    small = np.sqrt((x - col0) ** 2 + (y - row0) ** 2) <= radius
    return resize(small.astype(float), (height, width), order=1) > 0.5


# --------------------------------------------------------------------------- prediction

def predict(model, rgb, tta=True):
    """Vessel probability map (H, W) float32 in [0, 1] of an RGB uint8 image at its own size. `tta`: mean of the
    sigmoid outputs over the identity and the horizontal, vertical and double flips."""
    import torch
    from PIL import Image
    from skimage.measure import regionprops
    from skimage.transform import resize
    from torchvision import transforms
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got {rgb.shape}")
    rgb = rgb.astype(np.uint8) if rgb.dtype != np.uint8 else rgb
    mask = fov_mask(Image.fromarray(rgb))
    if not mask.any():
        raise ValueError("no field of view found: the fitted circle is empty")
    top, left, bottom, right = regionprops(mask.astype(int))[0].bbox
    crop = rgb[top:bottom, left:right]
    to_tensor = transforms.Compose([transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)), transforms.ToTensor()])
    x = to_tensor(Image.fromarray(crop)).to(next(model.parameters()).device)
    with torch.no_grad():
        pred = torch.sigmoid(model(x[None])[0])
        if tta:
            views = [torch.sigmoid(model(x.flip(-1)[None])[0].flip(-1)),
                     torch.sigmoid(model(x.flip(-2)[None])[0].flip(-2)),
                     torch.sigmoid(model(x.flip(-1).flip(-2)[None])[0].flip(-1).flip(-2))]
            pred = torch.mean(torch.stack([pred] + views), dim=0)
    pred = resize(pred.cpu().numpy()[-1], output_shape=crop.shape[:2], order=3)
    full = np.zeros(mask.shape, dtype=np.float64)
    full[top:bottom, left:right] = pred
    full[~mask] = 0
    return full.astype(np.float32)


def vessel_frame(model, stage2_rgb, size=512):
    """The vessel map on the direct 512 frame: predicted on the native Stage-2 image, then resized together with
    the image (one joint resize)."""
    from . import preprocess
    return preprocess.resize_with_extra_channel(stage2_rgb, predict(model, stage2_rgb), size)[..., 3]
