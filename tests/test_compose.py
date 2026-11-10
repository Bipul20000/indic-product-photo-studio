import cv2
import numpy as np
import pytest

from studio.compose.compose import (
    PRESETS,
    add_soft_shadow,
    composite,
    make_white_background,
    place_product,
)


@pytest.fixture
def synthetic_rgba():
    """Create a synthetic 200x200 RGBA image with a 100x100 product in the center."""
    rgba = np.zeros((200, 200, 4), dtype=np.uint8)
    rgba[50:150, 50:150, 0] = 60   # B
    rgba[50:150, 50:150, 1] = 120  # G
    rgba[50:150, 50:150, 2] = 180  # R
    rgba[50:150, 50:150, 3] = 255  # Alpha
    return rgba


def test_place_product_output_size_matches_preset(synthetic_rgba):
    """place_product output size matches all defined canvas presets."""
    for preset_name, (w, h) in PRESETS.items():
        canvas_rgba, mask = place_product(synthetic_rgba, preset=preset_name)
        assert canvas_rgba.shape == (h, w, 4), f"Mismatch for {preset_name}: got {canvas_rgba.shape}"
        assert mask.shape == (h, w), f"Mask mismatch for {preset_name}: got {mask.shape}"


def test_place_product_bbox_centered(synthetic_rgba):
    """The placed product bounding box is centered when vertical_offset is zero."""
    canvas_rgba, _ = place_product(synthetic_rgba, preset="square_1000", fill=0.80, vertical_offset=0.0)
    pts = np.argwhere(canvas_rgba[:, :, 3] > 0)
    assert len(pts) > 0

    y_min, x_min = pts.min(axis=0)
    y_max, x_max = pts.max(axis=0) + 1

    left_dist = x_min
    right_dist = 1000 - x_max
    top_dist = y_min
    bottom_dist = 1000 - y_max

    assert abs(left_dist - right_dist) <= 1, f"Horizontal not centered: left={left_dist}, right={right_dist}"
    assert abs(top_dist - bottom_dist) <= 1, f"Vertical not centered: top={top_dist}, bottom={bottom_dist}"


def test_composite_product_region_equals_original(synthetic_rgba):
    """Product interior in composite equals the resized original pixels within small tolerance."""
    canvas_rgba, mask = place_product(synthetic_rgba, preset="square_1000", fill=0.80, vertical_offset=0.0)
    bg = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    out = composite(bg, canvas_rgba, mask, shadow=True)

    solid = canvas_rgba[:, :, 3] == 255
    assert np.any(solid)

    diff = np.max(np.abs(out[solid].astype(int) - canvas_rgba[solid, :3].astype(int)))
    assert diff <= 1, f"Product pixel difference too large: max diff = {diff}"


def test_shadow_only_darkens_outside_product(synthetic_rgba):
    """Shadow only darkens background pixels outside the product, leaving interior untouched."""
    canvas_rgba, mask = place_product(synthetic_rgba, preset="square_1000", fill=0.80, vertical_offset=0.0)
    bg = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    out = composite(bg, canvas_rgba, mask, shadow=True)

    # Product interior is not darkened by shadow
    solid = canvas_rgba[:, :, 3] == 255
    diff = np.max(np.abs(out[solid].astype(int) - canvas_rgba[solid, :3].astype(int)))
    assert diff <= 1

    # Outside product directly below (y shift = 12), shadow darkens pure white (255)
    pts = np.argwhere(solid)
    y_max = pts[:, 0].max()
    x_mid = pts[:, 1].mean().astype(int)

    shadow_y = y_max + 8
    if shadow_y < 1000:
        shadow_pixel = out[shadow_y, x_mid]
        assert np.all(shadow_pixel < 255), f"Expected darkened shadow pixel, got {shadow_pixel}"


def test_output_corners_pure_white(synthetic_rgba):
    """Corners of the output image remain pure white (255, 255, 255)."""
    canvas_rgba, mask = place_product(synthetic_rgba, preset="square_1000", fill=0.80)
    bg = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    out = composite(bg, canvas_rgba, mask, shadow=True)

    assert np.all(out[0, 0] == 255), f"Top-left corner not white: {out[0, 0]}"
    assert np.all(out[0, 999] == 255), f"Top-right corner not white: {out[0, 999]}"
    assert np.all(out[999, 0] == 255), f"Bottom-left corner not white: {out[999, 0]}"
    assert np.all(out[999, 999] == 255), f"Bottom-right corner not white: {out[999, 999]}"


def test_place_product_empty_input():
    """Empty or zero-alpha inputs gracefully return zero arrays without error."""
    empty_rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    canvas_rgba, mask = place_product(empty_rgba, preset="square_1000")
    assert canvas_rgba.shape == (1000, 1000, 4)
    assert mask.shape == (1000, 1000)
    assert np.all(canvas_rgba == 0)
    assert np.all(mask == 0)
