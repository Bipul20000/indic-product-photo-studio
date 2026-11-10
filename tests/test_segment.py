import cv2
import numpy as np
import pytest

from studio.segment.segmenter import (
    SegmentResult,
    get_rembg_session,
    postprocess_mask,
    segment_product,
)


def test_postprocess_mask_keeps_large_component_and_drops_specks():
    """Post-processing keeps components >= 10% max area and drops small specks."""
    alpha = np.zeros((200, 200), dtype=np.uint8)
    # Primary large component (area ~ 5000 px)
    cv2.circle(alpha, (80, 80), 40, 255, -1)
    # Multi-part component >= 10% (area ~ 1250 px, ~25% of large)
    cv2.circle(alpha, (150, 150), 20, 255, -1)
    # Speck < 10% (area ~ 28 px)
    cv2.circle(alpha, (20, 20), 3, 255, -1)

    result = postprocess_mask(alpha)

    assert not result.failed
    assert result.mask[80, 80] == 255, "Primary object should be retained"
    assert result.mask[150, 150] == 255, "Secondary multi-part component (>=10%) should be retained"
    assert result.mask[20, 20] == 0, "Small noise speck (<10%) should be discarded"


def test_postprocess_mask_handles_empty_input():
    """All-zero, empty, or degenerate alpha inputs gracefully return failed SegmentResult."""
    # All zeros
    alpha_zeros = np.zeros((100, 100), dtype=np.uint8)
    res1 = postprocess_mask(alpha_zeros)
    assert res1.failed is True
    assert res1.area_fraction == 0.0
    assert np.all(res1.mask == 0)

    # Empty 0x0
    alpha_empty = np.zeros((0, 0), dtype=np.uint8)
    res2 = postprocess_mask(alpha_empty)
    assert res2.failed is True
    assert res2.area_fraction == 0.0

    # None input
    res3 = postprocess_mask(None)
    assert res3.failed is True


def test_postprocess_mask_fills_small_holes():
    """Morphological closing fills small holes inside foreground."""
    alpha = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(alpha, (100, 100), 50, 255, -1)
    # Puncture 2px hole in center
    cv2.circle(alpha, (100, 100), 2, 0, -1)
    assert alpha[100, 100] == 0

    result = postprocess_mask(alpha)
    assert result.mask[100, 100] == 255, "Small internal hole should be filled"


def test_postprocess_mask_touches_border():
    """Touches border is accurately detected when mask reaches within border_margin_px."""
    alpha = np.zeros((100, 100), dtype=np.uint8)
    # Mask touching top-left border
    alpha[0:30, 0:30] = 255

    res = postprocess_mask(alpha, border_margin_px=4)
    assert res.touches_border is True

    # Mask in center only
    alpha_center = np.zeros((100, 100), dtype=np.uint8)
    alpha_center[30:70, 30:70] = 255
    res_center = postprocess_mask(alpha_center, border_margin_px=4)
    assert res_center.touches_border is False


@pytest.mark.slow
def test_segment_product_with_model():
    """Test actual model segmentation if available offline."""
    try:
        session = get_rembg_session("isnet-general-use")
    except Exception as exc:
        pytest.skip(f"Model cannot be loaded offline: {exc}")

    # Create synthetic test image: a colored circle on neutral background
    img = np.full((300, 300, 3), 200, dtype=np.uint8)
    cv2.circle(img, (150, 150), 60, (50, 120, 200), -1)

    result = segment_product(img, model="isnet-general-use")
    assert isinstance(result, SegmentResult)
    assert result.mask.shape == (300, 300)
    assert result.rgba.shape == (300, 300, 4)
    assert not result.failed
    assert result.area_fraction > 0.05
