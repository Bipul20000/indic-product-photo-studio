import cv2
import numpy as np
import pytest

from studio.coach import CoachConfig, CoachReport, assess_photo
from studio.coach.checks import (
    check_blur,
    check_brightness,
    check_clutter,
    check_framing,
    check_resolution,
)


@pytest.fixture
def baseline_image():
    """Create a synthetic sharp, well-exposed, uncluttered 800x800 image."""
    img = np.full((800, 800, 3), 128, dtype=np.uint8)
    for y in range(200, 600, 20):
        for x in range(200, 600, 20):
            if ((y // 20) + (x // 20)) % 2 == 0:
                img[y : y + 20, x : x + 20] = 50
            else:
                img[y : y + 20, x : x + 20] = 200
    return img


def test_baseline_passes_checks(baseline_image):
    """Verify that the sharp baseline image passes all core checks."""
    cfg = CoachConfig()
    res = check_resolution(baseline_image, cfg)
    assert res.passed is True

    blur = check_blur(baseline_image, cfg)
    assert blur.passed is True

    bright = check_brightness(baseline_image, cfg)
    assert bright.passed is True

    clutter = check_clutter(baseline_image, cfg)
    assert clutter.passed is True

    report = assess_photo(baseline_image, config=cfg)
    assert report.ok is True
    assert len(report.tips) == 0


def test_gaussian_blur_fails(baseline_image):
    """Verify that GaussianBlur(31x31) fails blur check."""
    blurred = cv2.GaussianBlur(baseline_image, (31, 31), 0)
    res = check_blur(blurred)
    assert res.passed is False
    assert "blurry" in res.tip.lower()


def test_sharp_object_on_flat_background_passes_blur():
    """Verify a sharp textured object on a large flat background passes blur, while blurred fails."""
    cfg = CoachConfig()
    img = np.full((800, 800, 3), 128, dtype=np.uint8)
    for y in range(250, 550, 15):
        for x in range(250, 550, 15):
            if ((y // 15) + (x // 15)) % 2 == 0:
                img[y : y + 15, x : x + 15] = 20
            else:
                img[y : y + 15, x : x + 15] = 230

    res_sharp = check_blur(img, cfg)
    assert res_sharp.passed is True
    assert res_sharp.value > cfg.blur_min

    blurred = cv2.GaussianBlur(img, (31, 31), 0)
    res_blurred = check_blur(blurred, cfg)
    assert res_blurred.passed is False
    assert res_blurred.value < cfg.blur_min
    assert "blurry" in res_blurred.tip.lower()


def test_white_background_dark_object_passes_brightness():
    """Verify white background with a dark textured object passes brightness."""
    cfg = CoachConfig()
    img = np.full((800, 800, 3), 250, dtype=np.uint8)
    for y in range(250, 550, 15):
        for x in range(250, 550, 15):
            img[y : y + 15, x : x + 15] = 30 if ((y // 15) + (x // 15)) % 2 == 0 else 70

    res = check_brightness(img, cfg)
    assert res.passed is True
    assert res.tip == ""


def test_black_background_bright_object_passes_brightness():
    """Verify black background with a bright textured object passes brightness."""
    cfg = CoachConfig()
    img = np.full((800, 800, 3), 10, dtype=np.uint8)
    for y in range(250, 550, 15):
        for x in range(250, 550, 15):
            img[y : y + 15, x : x + 15] = 190 if ((y // 15) + (x // 15)) % 2 == 0 else 240

    res = check_brightness(img, cfg)
    assert res.passed is True
    assert res.tip == ""


def test_uniformly_dark_fails_brightness():
    """Verify that a uniformly dark image fails brightness."""
    cfg = CoachConfig()
    dark_img = np.full((800, 800, 3), 30, dtype=np.uint8)
    res = check_brightness(dark_img, cfg)
    assert res.passed is False
    assert "dark" in res.tip.lower()


def test_uniformly_near_white_fails_brightness():
    """Verify that a uniformly near-white image fails brightness."""
    cfg = CoachConfig()
    bright_img = np.full((800, 800, 3), 245, dtype=np.uint8)
    res = check_brightness(bright_img, cfg)
    assert res.passed is False
    assert "bright" in res.tip.lower()


def test_low_res_fails(baseline_image):
    """Verify that a 300x300 image fails the resolution check."""
    low_res = cv2.resize(baseline_image, (300, 300))
    res = check_resolution(low_res)
    assert res.passed is False
    assert "resolution" in res.tip.lower()


def test_border_clutter_check(baseline_image):
    """Verify plain border passes clutter, while noisy border fails clutter."""
    cfg = CoachConfig()
    plain_res = check_clutter(baseline_image, cfg)
    assert plain_res.passed is True

    # High-edge dense border pattern
    noisy = baseline_image.copy()
    for y in range(0, 120, 6):
        noisy[y, :] = 255 if (y // 6) % 2 == 0 else 0
    for x in range(0, 120, 6):
        noisy[:, x] = 255 if (x // 6) % 2 == 0 else 0

    noisy_res = check_clutter(noisy, cfg)
    assert noisy_res.passed is False
    assert "messy" in noisy_res.tip.lower()


def test_framing_checks():
    """Verify framing check with centered mask, tiny mask, and edge-touching mask."""
    cfg = CoachConfig()

    mask_mid = np.zeros((800, 800), dtype=np.uint8)
    mask_mid[200:600, 200:600] = 255
    res_mid = check_framing(mask_mid, cfg)
    assert res_mid.passed is True
    assert res_mid.tip == ""

    mask_tiny = np.zeros((800, 800), dtype=np.uint8)
    mask_tiny[375:425, 375:425] = 255
    res_tiny = check_framing(mask_tiny, cfg)
    assert res_tiny.passed is False
    assert "small" in res_tiny.tip.lower()

    mask_edge = np.zeros((800, 800), dtype=np.uint8)
    mask_edge[0:400, 200:600] = 255
    res_edge = check_framing(mask_edge, cfg)
    assert res_edge.passed is False
    assert "cut off" in res_edge.tip.lower()


def test_assess_photo_1x1_no_crash():
    """Verify that assess_photo handles a 1x1 image without crashing."""
    img_1x1 = np.zeros((1, 1, 3), dtype=np.uint8)
    report = assess_photo(img_1x1)
    assert isinstance(report, CoachReport)
    assert report.ok is False
    assert len(report.tips) > 0


def test_assess_photo_invalid_inputs():
    """Verify that invalid inputs return a failed report and do not raise."""
    rep_none = assess_photo(None)
    assert rep_none.ok is False

    rep_empty = assess_photo(np.array([]))
    assert rep_empty.ok is False
