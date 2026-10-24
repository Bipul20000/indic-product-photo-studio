from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np

from studio.coach.config import CoachConfig


@dataclass
class CheckResult:
    """Result of an individual photo quality check."""

    name: str
    passed: bool
    value: float
    tip: str


def check_resolution(img: np.ndarray, config: Optional[CoachConfig] = None) -> CheckResult:
    """Verify that the photo resolution meets the minimum required dimension."""
    if config is None:
        config = CoachConfig()

    h, w = img.shape[:2]
    min_side = min(h, w)
    passed = bool(min_side >= config.min_side_px)
    tip = (
        ""
        if passed
        else f"Photo resolution is too low ({min_side}px). Please take a photo with at least {config.min_side_px}px on the shortest side."
    )
    return CheckResult(name="resolution", passed=passed, value=float(min_side), tip=tip)


def check_blur(img: np.ndarray, config: Optional[CoachConfig] = None) -> CheckResult:
    """Verify image sharpness using Laplacian variance on resized grayscale."""
    if config is None:
        config = CoachConfig()

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return CheckResult(name="blur", passed=False, value=0.0, tip="Image has zero dimensions.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 and img.shape[2] == 3 else img

    longest = max(h, w)
    if longest != config.blur_resize_side and longest > 0:
        scale = config.blur_resize_side / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        gray = cv2.resize(gray, (new_w, new_h), interpolation=interp)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(lap.var())
    passed = bool(variance >= config.blur_min)
    tip = (
        ""
        if passed
        else "Photo is blurry. Hold the phone steady, tap on the product to focus, and take it again."
    )
    return CheckResult(name="blur", passed=passed, value=round(variance, 2), tip=tip)


def check_brightness(img: np.ndarray, config: Optional[CoachConfig] = None) -> CheckResult:
    """Verify image exposure: not too dark, not too bright, and not overly clipped."""
    if config is None:
        config = CoachConfig()

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return CheckResult(name="brightness", passed=False, value=0.0, tip="Image has zero dimensions.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 and img.shape[2] == 3 else img
    mean_val = float(np.mean(gray))
    clipped_frac = float(np.mean((gray <= 5) | (gray >= 250)))

    if mean_val < config.dark_max_mean:
        return CheckResult(
            name="brightness",
            passed=False,
            value=round(mean_val, 2),
            tip="Photo is too dark. Move next to a window or turn on more light.",
        )

    if mean_val > config.bright_min_mean:
        return CheckResult(
            name="brightness",
            passed=False,
            value=round(mean_val, 2),
            tip="Photo is too bright. Move away from direct harsh light or glare.",
        )

    if clipped_frac > config.clipped_max_fraction:
        return CheckResult(
            name="brightness",
            passed=False,
            value=round(clipped_frac, 4),
            tip="Lighting has harsh glare or deep shadows. Adjust lighting for more even exposure.",
        )

    return CheckResult(
        name="brightness",
        passed=True,
        value=round(mean_val, 2),
        tip="",
    )


def check_clutter(img: np.ndarray, config: Optional[CoachConfig] = None) -> CheckResult:
    """Verify background cleanliness via Canny edge density in the outer 15% border band."""
    if config is None:
        config = CoachConfig()

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return CheckResult(name="clutter", passed=False, value=0.0, tip="Image has zero dimensions.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 and img.shape[2] == 3 else img

    # Use basic Canny edge detector
    edges = cv2.Canny(gray, 50, 150)

    band_h = max(1, int(round(h * config.border_band_fraction)))
    band_w = max(1, int(round(w * config.border_band_fraction)))

    border_mask = np.ones((h, w), dtype=bool)
    if h > 2 * band_h and w > 2 * band_w:
        border_mask[band_h : h - band_h, band_w : w - band_w] = False

    total_border_pixels = int(np.count_nonzero(border_mask))
    if total_border_pixels == 0:
        density = 0.0
    else:
        edge_pixels_in_border = int(np.count_nonzero((edges > 0) & border_mask))
        density = float(edge_pixels_in_border / total_border_pixels)

    passed = bool(density <= config.clutter_max)
    tip = (
        ""
        if passed
        else "Background looks messy. Place the product on a plain sheet or table."
    )
    return CheckResult(name="clutter", passed=passed, value=round(density, 4), tip=tip)


def check_framing(mask: np.ndarray, config: Optional[CoachConfig] = None) -> CheckResult:
    """Verify product framing: area coverage and not cut off at image borders."""
    if config is None:
        config = CoachConfig()

    h, w = mask.shape[:2]
    total_px = h * w
    if total_px == 0:
        return CheckResult(name="framing", passed=False, value=0.0, tip="Product mask is empty.")

    binary = mask > 0
    product_px = int(np.count_nonzero(binary))
    area_frac = float(product_px / total_px)

    margin = config.border_margin_px
    if h <= 2 * margin or w <= 2 * margin:
        touches_border = bool(np.any(binary))
    else:
        touches_border = bool(
            np.any(binary[:margin, :])
            or np.any(binary[-margin:, :])
            or np.any(binary[:, :margin])
            or np.any(binary[:, -margin:])
        )

    if touches_border:
        return CheckResult(
            name="framing",
            passed=False,
            value=round(area_frac, 4),
            tip="Product is cut off at the edge. Move the phone back so the entire product is in the frame.",
        )

    if area_frac < config.min_product_area:
        return CheckResult(
            name="framing",
            passed=False,
            value=round(area_frac, 4),
            tip="Product is too small in the frame. Move the phone closer to fill more of the frame.",
        )

    if area_frac > config.max_product_area:
        return CheckResult(
            name="framing",
            passed=False,
            value=round(area_frac, 4),
            tip="Product is too close or taking up too much of the photo. Move the phone back a bit.",
        )

    return CheckResult(
        name="framing",
        passed=True,
        value=round(area_frac, 4),
        tip="",
    )
