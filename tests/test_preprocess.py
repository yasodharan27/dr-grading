import cv2
import numpy as np

from drgrade import preprocess as pp


def _fundus(h=300, w=400, radius=110, centre=(150, 200), value=120):
    img = np.zeros((h, w, 3), np.uint8)
    cv2.circle(img, (centre[1], centre[0]), radius, (value, value, value), -1)
    return img


def test_fundus_box_finds_the_disk():
    top, bottom, left, right = pp.fundus_box(_fundus())
    assert abs(top - 40) <= 2 and abs(bottom - 261) <= 2 and abs(left - 90) <= 2 and abs(right - 311) <= 2


def test_dark_or_tiny_fundus_keeps_the_whole_image():
    assert pp.fundus_box(np.zeros((100, 120, 3), np.uint8)) == (0, 100, 0, 120)             # nothing lit
    assert pp.fundus_box(_fundus(radius=4, value=200)) == (0, 300, 0, 400)                 # box below 10 % of a side


def test_pixels_at_the_threshold_are_not_lit():
    img = np.full((50, 50, 3), pp.GRAY_THRESHOLD, np.uint8)
    assert pp.fundus_box(img) == (0, 50, 0, 50)
    img[10:40, 5:45] = pp.GRAY_THRESHOLD + 1
    assert pp.fundus_box(img) == (10, 40, 5, 45)


def test_gamma_brightens_midtones_and_keeps_ends():
    x = np.array([[0, 128, 255]], np.uint8)
    y = pp.gamma_correction(x)
    assert y[0, 0] == 0 and y[0, 2] == 255 and y[0, 1] > 128


def test_frames_from_a_file(tmp_path):
    path = str(tmp_path / "eye.png")
    img = _fundus()
    img[100:200, 150:250] = (60, 90, 140)
    cv2.imwrite(path, img)
    f = pp.frames(path)
    assert f["stage2"].shape == (300, 400, 3) and f["stage2"].dtype == np.uint8
    for arm in ("direct", "crop"):
        assert f[arm].shape == (512, 512, 3) and f[arm].dtype == np.float32
        assert 0.0 <= f[arm].min() and f[arm].max() <= 1.0
    assert not np.array_equal(f["direct"], f["crop"])
    assert np.array_equal(pp.frame(path, "crop"), f["crop"])
