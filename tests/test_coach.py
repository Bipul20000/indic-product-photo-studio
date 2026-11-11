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
    """Verify framing check: centered mask passes, corner-touching mask passes,
    wide strip on left edge fails with side=left, and tiny mask fails for too small."""
    cfg = CoachConfig()

    mask_mid = np.zeros((800, 800), dtype=np.uint8)
    mask_mid[200:600, 200:600] = 255
    res_mid = check_framing(mask_mid, cfg)
    assert res_mid.passed is True
    assert res_mid.tip == ""

    # A mask touching one corner pixel passes
    mask_corner = np.zeros((800, 800), dtype=np.uint8)
    mask_corner[200:600, 200:600] = 255
    mask_corner[0, 0] = 255
    res_corner = check_framing(mask_corner, cfg)
    assert res_corner.passed is True
    assert res_corner.tip == ""

    # A mask covering a wide strip of the left edge fails with side=left
    mask_edge = np.zeros((800, 800), dtype=np.uint8)
    mask_edge[200:600, 0:3] = 255
    res_edge = check_framing(mask_edge, cfg)
    assert res_edge.passed is False
    assert "cut off" in res_edge.tip.lower()
    assert "left" in res_edge.tip.lower()

    # Tiny mask fails for too small
    mask_tiny = np.zeros((800, 800), dtype=np.uint8)
    mask_tiny[375:425, 375:425] = 255
    res_tiny = check_framing(mask_tiny, cfg)
    assert res_tiny.passed is False
    assert "small" in res_tiny.tip.lower()


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


def test_mask_based_exposure_dark_on_white():
    """A dark object on a white background passes exposure when mask is provided."""
    img = np.full((800, 800, 3), 255, dtype=np.uint8)
    mask = np.zeros((800, 800), dtype=np.uint8)
    mask[200:600, 200:600] = 255
    img[200:600, 200:600] = 80

    cfg = CoachConfig(
        dark_feature="product_p95",
        dark_threshold=50.0,
        bright_feature="product_clip_hi",
        bright_threshold=0.20,
    )
    res = check_brightness(img, mask=mask, config=cfg)
    assert res.passed is True
    assert res.tip == ""
    assert res.used_fallback is False


def test_mask_based_exposure_washed_out_fails():
    """A washed-out object with product pixels at 255 fails bright check."""
    img = np.full((800, 800, 3), 120, dtype=np.uint8)
    mask = np.zeros((800, 800), dtype=np.uint8)
    mask[200:600, 200:600] = 255
    img[200:600, 200:600] = 255  # 100% clipped product

    cfg = CoachConfig(
        dark_feature="product_p95",
        dark_threshold=50.0,
        bright_feature="product_clip_hi",
        bright_threshold=0.20,
    )
    res = check_brightness(img, mask=mask, config=cfg)
    assert res.passed is False
    assert "bright" in res.tip.lower()


def test_mask_based_clutter_plain_vs_busy():
    """Plain background passes clutter, while background with dense edges fails."""
    mask = np.zeros((800, 800), dtype=np.uint8)
    mask[200:600, 200:600] = 255
    cfg = CoachConfig(clutter_feature="bg_edge_density", clutter_threshold=0.08)

    # Plain background
    img_plain = np.full((800, 800, 3), 200, dtype=np.uint8)
    res_plain = check_clutter(img_plain, mask=mask, config=cfg)
    assert res_plain.passed is True
    assert res_plain.used_fallback is False

    # Busy background
    img_busy = np.full((800, 800, 3), 200, dtype=np.uint8)
    for y in range(0, 160, 8):
        img_busy[y, :] = 0
    for x in range(0, 160, 8):
        img_busy[:, x] = 0

    res_busy = check_clutter(img_busy, mask=mask, config=cfg)
    assert res_busy.passed is False
    assert "messy" in res_busy.tip.lower()


def test_mask_fallback_when_mask_none_or_small():
    """Product-based features fall back to whole-image version when mask is None or < 2%."""
    img = np.full((800, 800, 3), 150, dtype=np.uint8)
    cfg = CoachConfig(
        dark_feature="product_p95",
        bright_feature="product_clip_hi",
        clutter_feature="bg_edge_density",
    )

    # Case 1: mask is None
    res_b_none = check_brightness(img, mask=None, config=cfg)
    assert res_b_none.used_fallback is True
    res_c_none = check_clutter(img, mask=None, config=cfg)
    assert res_c_none.used_fallback is True

    # Case 2: mask has product area < 2% (e.g. 50x50 in 800x800 = 0.39%)
    mask_small = np.zeros((800, 800), dtype=np.uint8)
    mask_small[375:425, 375:425] = 255
    res_b_small = check_brightness(img, mask=mask_small, config=cfg)
    assert res_b_small.used_fallback is True
    res_c_small = check_clutter(img, mask=mask_small, config=cfg)
    assert res_c_small.used_fallback is True


def test_threshold_selection_rule_and_cap_budget():
    """Verify threshold selection rule (recall -> FPR -> median) and cap budget loop."""
    from scripts.tune_coach import choose_qualifying_cap, select_threshold

    # 1. Higher recall wins under the cap
    # Good values: [20, 40], Bad values: [5, 15]
    # At T=10: bad < 10 is {5} (rec=0.5), good < 10 is 0 (fpr=0.0)
    # At T=25: bad < 25 is {5, 15} (rec=1.0), good < 25 is {20} (fpr=0.5)
    # With fpr_cap=0.50, T=25 has higher recall (1.0 > 0.5), so it wins.
    t, fpr, rec = select_threshold([20.0, 40.0], [5.0, 15.0], direction="lower", fpr_cap=0.50)
    assert rec == 1.0
    assert t > 15.0

    # 2. Tie in recall goes to the lowest FPR
    # Good values: [20, 40], Bad values: [5, 10]
    # Unique: [5, 10, 20, 40]. Midpoints: [7.5, 15.0, 30.0]
    # At T=15.0: bad < 15 is {5, 10} (rec=1.0), good < 15 is 0 (fpr=0.0)
    # At T=30.0: bad < 30 is {5, 10} (rec=1.0), good < 30 is {20} (fpr=0.5)
    # With fpr_cap=0.50, both have recall 1.0, but T=15.0 has lower FPR (0.0 < 0.5), so T=15.0 wins.
    t, fpr, rec = select_threshold([20.0, 40.0], [5.0, 10.0], direction="lower", fpr_cap=0.50)
    assert rec == 1.0
    assert fpr == 0.0
    assert t == 15.0

    # 3. Tie in recall AND lowest FPR goes to median of tied thresholds
    # Good values: [50, 60], Bad values: [10, 20]
    # all_vals injects points [30, 34, 38] between bad and good
    # Midpoints between 20 and 50 are 25.0, 32.0, 36.0, 44.0 (all rec=1.0, fpr=0.0)
    # Tied thresholds: [25.0, 32.0, 36.0, 44.0] -> median is (32.0 + 36.0)/2 = 34.0
    all_v = [10.0, 20.0, 30.0, 34.0, 38.0, 50.0, 60.0]
    t, fpr, rec = select_threshold([50.0, 60.0], [10.0, 20.0], direction="lower", fpr_cap=0.05, all_vals=all_v)
    assert rec == 1.0
    assert fpr == 0.0
    assert t == 34.0

    # 4. Cap budget loop picks the largest qualifying cap
    cap_results = [
        {"cap": 0.05, "combined_far": 0.16},
        {"cap": 0.04, "combined_far": 0.13},
        {"cap": 0.03, "combined_far": 0.09},  # Largest cap <= 0.10
        {"cap": 0.02, "combined_far": 0.06},
        {"cap": 0.01, "combined_far": 0.03},
        {"cap": 0.00, "combined_far": 0.01},
    ]
    chosen = choose_qualifying_cap(cap_results, max_combined_far=0.10)
    assert chosen == 0.03

    # Fallback to cap 0.0 when none qualifies
    no_qualify = [
        {"cap": 0.05, "combined_far": 0.35},
        {"cap": 0.00, "combined_far": 0.30},
    ]
    chosen_fb = choose_qualifying_cap(no_qualify, max_combined_far=0.10)
    assert chosen_fb == 0.0


