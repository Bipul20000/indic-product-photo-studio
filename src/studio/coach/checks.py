from dataclasses import dataclass
from typing import Dict, Optional
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
    """Verify image sharpness using 8x8 tile scores based on config.blur_method."""
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

    method = config.blur_method.lower() if config.blur_method else "tile_p90"

    row_chunks = np.array_split(gray, 8, axis=0)
    tile_scores = []

    for r_chunk in row_chunks:
        tiles = np.array_split(r_chunk, 8, axis=1)
        for tile in tiles:
            if tile.size == 0:
                continue
            lap_var = float(cv2.Laplacian(tile, cv2.CV_64F).var())

            if method == "tile_norm_p90":
                lum_var = float(np.var(tile))
                tile_scores.append(lap_var / (lum_var + 1.0))
            elif method == "scale_ratio":
                th, tw = tile.shape[:2]
                down_w = max(1, tw // 2)
                down_h = max(1, th // 2)
                tile_down = cv2.resize(tile, (down_w, down_h), interpolation=cv2.INTER_AREA)
                lap_down_var = float(cv2.Laplacian(tile_down, cv2.CV_64F).var())
                tile_scores.append(lap_var / (lap_down_var + 1e-6))
            else:  # "tile_p90"
                tile_scores.append(lap_var)

    if not tile_scores:
        blur_score = 0.0
    else:
        blur_score = float(np.percentile(tile_scores, 90))

    passed = bool(blur_score >= config.blur_min)
    tip = (
        ""
        if passed
        else "Photo is blurry. Hold the phone steady, tap on the product to focus, and take it again."
    )
    return CheckResult(name="blur", passed=passed, value=round(blur_score, 2), tip=tip)


def check_brightness(img: np.ndarray, config: Optional[CoachConfig] = None) -> CheckResult:
    """Verify image exposure using 95th and 5th percentiles of luminance."""
    if config is None:
        config = CoachConfig()

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return CheckResult(name="brightness", passed=False, value=0.0, tip="Image has zero dimensions.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 and img.shape[2] == 3 else img
    p95 = float(np.percentile(gray, 95))
    p5 = float(np.percentile(gray, 5))

    if p95 < config.dark_max_p95:
        return CheckResult(
            name="brightness",
            passed=False,
            value=round(p95, 2),
            tip="Photo is too dark. Move next to a window or turn on more light.",
        )

    if p5 > config.bright_min_p5:
        return CheckResult(
            name="brightness",
            passed=False,
            value=round(p5, 2),
            tip="Photo is too bright. Move away from direct harsh light or glare.",
        )

    return CheckResult(
        name="brightness",
        passed=True,
        value=round(p95, 2),
        tip="",
    )


def check_clutter(img: np.ndarray, config: Optional[CoachConfig] = None) -> CheckResult:
    """Verify background cleanliness via blurred Canny edge density in outer 15% border band."""
    if config is None:
        config = CoachConfig()

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return CheckResult(name="clutter", passed=False, value=0.0, tip="Image has zero dimensions.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 and img.shape[2] == 3 else img

    # Resize copy to 512 on longest side
    longest = max(h, w)
    if longest != 512 and longest > 0:
        scale = 512.0 / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        gray_512 = cv2.resize(gray, (new_w, new_h), interpolation=interp)
    else:
        gray_512 = gray

    h512, w512 = gray_512.shape[:2]

    # GaussianBlur 5x5 to suppress fine textures
    blurred = cv2.GaussianBlur(gray_512, (5, 5), 0)

    # Canny edges
    edges = cv2.Canny(blurred, 50, 150)

    band_h = max(1, int(round(h512 * config.border_band_fraction)))
    band_w = max(1, int(round(w512 * config.border_band_fraction)))

    border_mask = np.ones((h512, w512), dtype=bool)
    if h512 > 2 * band_h and w512 > 2 * band_w:
        border_mask[band_h : h512 - band_h, band_w : w512 - band_w] = False

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


def compute_side_contact(mask: np.ndarray, border_margin_px: int = 4) -> Dict[str, float]:
    """Compute the contact fraction for each edge (top, bottom, left, right).

    Looking at the outermost border_margin_px rows/columns, a position along that edge
    line is treated as present if any pixel in that strip is mask (> 0).
    """
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}

    m = max(1, min(border_margin_px, h, w))

    # Top: rows 0:m across columns
    top_present = np.any(mask[:m, :] > 0, axis=0)
    top_frac = float(np.count_nonzero(top_present) / w)

    # Bottom: rows -m: across columns
    bottom_present = np.any(mask[-m:, :] > 0, axis=0)
    bottom_frac = float(np.count_nonzero(bottom_present) / w)

    # Left: cols 0:m across rows
    left_present = np.any(mask[:, :m] > 0, axis=1)
    left_frac = float(np.count_nonzero(left_present) / h)

    # Right: cols -m: across rows
    right_present = np.any(mask[:, -m:] > 0, axis=1)
    right_frac = float(np.count_nonzero(right_present) / h)

    return {
        "top": round(top_frac, 4),
        "bottom": round(bottom_frac, 4),
        "left": round(left_frac, 4),
        "right": round(right_frac, 4),
    }


def check_framing(mask: np.ndarray, config: Optional[CoachConfig] = None) -> CheckResult:
    """Verify product framing: per-side border cut-off and area coverage."""
    if config is None:
        config = CoachConfig()

    h, w = mask.shape[:2]
    total_px = h * w
    if total_px == 0:
        return CheckResult(name="framing", passed=False, value=0.0, tip="Product mask is empty.")

    binary = mask > 0
    product_px = int(np.count_nonzero(binary))
    if product_px == 0:
        return CheckResult(name="framing", passed=False, value=0.0, tip="Product mask is empty.")

    area_frac = float(product_px / total_px)
    side_contact = compute_side_contact(mask, border_margin_px=config.border_margin_px)

    # Cut off only if at least one side has contact fraction >= cutoff_side_fraction
    cut_sides = [side for side, frac in side_contact.items() if frac >= config.cutoff_side_fraction]
    if cut_sides:
        max_side = max(side_contact, key=side_contact.get)
        max_frac = side_contact[max_side]
        tip = f"Your product is cut off at the {max_side}. Step back a little so the whole product is in the frame."
        return CheckResult(
            name="framing",
            passed=False,
            value=round(max_frac, 4),
            tip=tip,
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
