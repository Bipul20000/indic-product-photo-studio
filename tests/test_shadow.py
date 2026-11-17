"""Tests for realistic dual-layer contact and ambient shadow synthesis."""

import numpy as np
import pytest

from studio.compose.shadow import add_realistic_shadow


def test_realistic_shadow_darkens_under_product_base():
    """Verify that contact shadow creates realistic darkening right under product footprint."""
    h, w = 150, 150
    canvas_bgr = np.full((h, w, 3), 255, dtype=np.uint8)  # White canvas

    # Product mask located at y: 30..100, x: 50..100
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[30:100, 50:100] = 255

    shadowed = add_realistic_shadow(
        canvas_bgr,
        mask,
        contact_opacity=0.70,
        diffuse_opacity=0.30,
        diffuse_offset_y=12,
        contact_offset_y=3,
    )

    assert shadowed.shape == (h, w, 3)
    # Area right under product base (e.g. y=103, x=75) must be darkened
    contact_val = shadowed[103, 75, 0]
    assert contact_val < 200, "Base contact area should be darkened significantly"

    # Far away area (e.g. y=10, x=10) should remain untouched (255)
    assert shadowed[10, 10, 0] == 255


def test_realistic_shadow_empty_mask():
    """Verify that an empty mask returns unmodified canvas copy."""
    h, w = 60, 60
    canvas_bgr = np.full((h, w, 3), 180, dtype=np.uint8)
    empty_mask = np.zeros((h, w), dtype=np.uint8)

    out = add_realistic_shadow(canvas_bgr, empty_mask)
    assert np.array_equal(out, canvas_bgr)
