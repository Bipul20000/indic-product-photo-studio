"""Tests for environmental lighting and ambient color harmonization."""

import numpy as np
import pytest

from studio.compose.harmonize import harmonize_product


def test_harmonize_preserves_shape_and_alpha():
    """Verify harmonize_product returns correct shape and identical alpha channel."""
    h, w = 100, 100
    prod_rgba = np.full((h, w, 4), 200, dtype=np.uint8)
    # Circle mask in alpha
    y, x = np.ogrid[:h, :w]
    circle = (x - 50) ** 2 + (y - 50) ** 2 <= 30**2
    prod_rgba[~circle, 3] = 0

    bg_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    # Golden background
    bg_bgr[:, :] = [30, 140, 255]

    harmonized = harmonize_product(prod_rgba, bg_bgr, strength=0.2, rim_strength=0.3)

    assert harmonized.shape == (h, w, 4)
    # Alpha must remain 100% identical
    assert np.array_equal(harmonized[:, :, 3], prod_rgba[:, :, 3])


def test_harmonize_zero_strength_is_noop():
    """Verify that strength=0 and rim_strength=0 returns identical image."""
    h, w = 80, 80
    prod_rgba = np.random.randint(0, 256, (h, w, 4), dtype=np.uint8)
    bg_bgr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    out = harmonize_product(prod_rgba, bg_bgr, strength=0.0, rim_strength=0.0)
    assert np.array_equal(out, prod_rgba)


def test_harmonize_ambient_shift():
    """Verify that a cool product shifts slightly toward warm background chromaticity."""
    h, w = 120, 120
    # Pure cool blue product: B=255, G=50, R=50
    prod_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    prod_rgba[30:90, 30:90, :3] = [255, 50, 50]
    prod_rgba[30:90, 30:90, 3] = 255

    # Pure warm red/orange background: B=20, G=100, R=240
    bg_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    bg_bgr[:, :] = [20, 100, 240]

    out = harmonize_product(prod_rgba, bg_bgr, strength=0.3, rim_strength=0.0)
    # In center of product, Red should increase due to warm ambient shift
    center_r = out[60, 60, 2]
    orig_r = prod_rgba[60, 60, 2]
    assert center_r > orig_r
