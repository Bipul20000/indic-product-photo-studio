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
    # Add a high-contrast pattern in the center (well within border margin)
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


def test_dark_image_fails(baseline_image):
    """Verify that multiplying the image by 0.2 fails brightness check (too dark)."""
    dark = (baseline_image * 0.2).astype(np.uint8)
    res = check_brightness(dark)
    assert res.passed is False
    assert "dark" in res.tip.lower()


def test_bright_image_fails(baseline_image):
    """Verify that pushing pixel values near white fails brightness check (too bright)."""
    bright = np.clip(baseline_image.astype(float) + 110, 0, 255).astype(np.uint8)
    res = check_brightness(bright)
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
    # Plain border on baseline
    plain_res = check_clutter(baseline_image)
    assert plain_res.passed is True

    # High-edge border
    noisy = baseline_image.copy()
    for y in range(0, 120, 4):
        noisy[y, :] = 255 if (y // 4) % 2 == 0 else 0

    noisy_res = check_clutter(noisy)
    assert noisy_res.passed is False
    assert "messy" in noisy_res.tip.lower()


def test_framing_checks():
    """Verify framing check with centered mask, tiny mask, and edge-touching mask."""
    cfg = CoachConfig()

    # Mask in the middle (area ~ 25%, well inside borders)
    mask_mid = np.zeros((800, 800), dtype=np.uint8)
    mask_mid[200:600, 200:600] = 255
    res_mid = check_framing(mask_mid, cfg)
    assert res_mid.passed is True
    assert res_mid.tip == ""

    # Tiny mask (area < min_product_area 15%)
    mask_tiny = np.zeros((800, 800), dtype=np.uint8)
    mask_tiny[375:425, 375:425] = 255
    res_tiny = check_framing(mask_tiny, cfg)
    assert res_tiny.passed is False
    assert "small" in res_tiny.tip.lower()

    # Mask touching the edge
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
