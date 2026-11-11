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
    used_fallback: bool = False


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


def check_brightness(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    config: Optional[CoachConfig] = None,
) -> CheckResult:
    """Verify image exposure using configurable dark and bright feature candidates with mask support."""
    if isinstance(mask, CoachConfig) and config is None:
        config = mask
        mask = None
    if config is None:
        config = CoachConfig()

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return CheckResult(name="brightness", passed=False, value=0.0, tip="Image has zero dimensions.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 and img.shape[2] == 3 else img
    total_px = h * w

    # Determine if mask is valid and >= 2% of image
    has_valid_mask = (
        mask is not None
        and isinstance(mask, np.ndarray)
        and mask.size > 0
        and len(mask.shape) >= 2
        and (int(np.count_nonzero(mask > 0)) / total_px) >= 0.02
    )

    # Dark check
    dark_feat = config.dark_feature
    dark_t = config.dark_threshold if config.dark_threshold is not None else config.dark_max_p95
    used_fb_dark = False

    if dark_feat == "global_p95":
        dark_val = float(np.percentile(gray, 95))
        dark_failed = dark_val < dark_t
    elif dark_feat in ("product_p95", "product_p50"):
        if has_valid_mask:
            p = 95 if dark_feat == "product_p95" else 50
            dark_val = float(np.percentile(gray[mask > 0], p))
            dark_failed = dark_val < dark_t
        else:
            dark_val = float(np.percentile(gray, 95))
            dark_failed = dark_val < config.dark_max_p95
            used_fb_dark = True
    else:
        dark_val = float(np.percentile(gray, 95))
        dark_failed = dark_val < dark_t

    # Bright check
    bright_feat = config.bright_feature
    bright_t = config.bright_threshold if config.bright_threshold is not None else config.bright_min_p5
    used_fb_bright = False

    if bright_feat == "global_p5":
        bright_val = float(np.percentile(gray, 5))
        bright_failed = bright_val > bright_t
    elif bright_feat == "product_p5":
        if has_valid_mask:
            bright_val = float(np.percentile(gray[mask > 0], 5))
            bright_failed = bright_val > bright_t
        else:
            bright_val = float(np.percentile(gray, 5))
            bright_failed = bright_val > config.bright_min_p5
            used_fb_bright = True
    elif bright_feat == "product_clip_hi":
        if has_valid_mask:
            prod_px = gray[mask > 0]
            bright_val = float(np.count_nonzero(prod_px >= 250) / prod_px.size)
            bright_failed = bright_val > bright_t
        else:
            bright_val = float(np.percentile(gray, 5))
            bright_failed = bright_val > config.bright_min_p5
            used_fb_bright = True
    else:
        bright_val = float(np.percentile(gray, 5))
        bright_failed = bright_val > bright_t

    used_fb = used_fb_dark or used_fb_bright

    if dark_failed:
        return CheckResult(
            name="brightness",
            passed=False,
            value=round(dark_val, 4),
            tip="Photo is too dark. Move next to a window or turn on more light.",
            used_fallback=used_fb_dark,
        )

    if bright_failed:
        return CheckResult(
            name="brightness",
            passed=False,
            value=round(bright_val, 4),
            tip="Photo is too bright. Move away from direct harsh light or glare.",
            used_fallback=used_fb_bright,
        )

    return CheckResult(
        name="brightness",
        passed=True,
        value=round(dark_val, 4),
        tip="",
        used_fallback=used_fb,
    )


# Alias check_exposure to check_brightness
check_exposure = check_brightness


def check_clutter(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    config: Optional[CoachConfig] = None,
) -> CheckResult:
    """Verify background cleanliness via configurable candidate features with mask support."""
    if isinstance(mask, CoachConfig) and config is None:
        config = mask
        mask = None
    if config is None:
        config = CoachConfig()

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return CheckResult(name="clutter", passed=False, value=0.0, tip="Image has zero dimensions.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 and img.shape[2] == 3 else img
    total_px = h * w

    # Determine if mask is valid and >= 2% of image
    has_valid_mask = (
        mask is not None
        and isinstance(mask, np.ndarray)
        and mask.size > 0
        and len(mask.shape) >= 2
        and (int(np.count_nonzero(mask > 0)) / total_px) >= 0.02
    )

    # Resize copy to 512 on longest side
    longest = max(h, w)
    if longest != 512 and longest > 0:
        scale = 512.0 / float(longest)
        w512 = max(1, int(round(w * scale)))
        h512 = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        gray_512 = cv2.resize(gray, (w512, h512), interpolation=interp)
    else:
        gray_512 = gray
        w512, h512 = w, h

    blurred = cv2.GaussianBlur(gray_512, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Compute global border density (outer 15% band)
    band_h = max(1, int(round(h512 * config.border_band_fraction)))
    band_w = max(1, int(round(w512 * config.border_band_fraction)))
    border_mask = np.ones((h512, w512), dtype=bool)
    if h512 > 2 * band_h and w512 > 2 * band_w:
        border_mask[band_h : h512 - band_h, band_w : w512 - band_w] = False

    tot_border = int(np.count_nonzero(border_mask))
    global_border_val = (
        float(np.count_nonzero((edges > 0) & border_mask) / tot_border)
        if tot_border > 0
        else 0.0
    )

    clutter_feat = config.clutter_feature
    clutter_t = config.clutter_threshold if config.clutter_threshold is not None else config.clutter_max

    if clutter_feat == "global_border":
        passed = bool(global_border_val <= clutter_t)
        return CheckResult(
            name="clutter",
            passed=passed,
            value=round(global_border_val, 4),
            tip="" if passed else "Background looks messy. Place the product on a plain sheet or table.",
            used_fallback=False,
        )

    # Product-based clutter feature requested
    if not has_valid_mask:
        passed = bool(global_border_val <= config.clutter_max)
        return CheckResult(
            name="clutter",
            passed=passed,
            value=round(global_border_val, 4),
            tip="" if passed else "Background looks messy. Place the product on a plain sheet or table.",
            used_fallback=True,
        )

    # Valid mask: compute background outside mask dilated by 15 px
    mask_512 = cv2.resize(mask, (w512, h512), interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    dilated = cv2.dilate((mask_512 > 0).astype(np.uint8), kernel)
    bg_mask = (dilated == 0)
    bg_px = int(np.count_nonzero(bg_mask))

    # Skip and pass if background is under 10% of image
    if bg_px / (h512 * w512) < 0.10:
        return CheckResult(name="clutter", passed=True, value=0.0, tip="", used_fallback=False)

    if clutter_feat == "bg_edge_density":
        density = float(np.count_nonzero((edges > 0) & bg_mask) / bg_px)
        passed = bool(density <= clutter_t)
        return CheckResult(
            name="clutter",
            passed=passed,
            value=round(density, 4),
            tip="" if passed else "Background looks messy. Place the product on a plain sheet or table.",
            used_fallback=False,
        )
    elif clutter_feat == "bg_lum_std":
        std = float(np.std(blurred[bg_mask]))
        passed = bool(std <= clutter_t)
        return CheckResult(
            name="clutter",
            passed=passed,
            value=round(std, 2),
            tip="" if passed else "Background looks messy. Place the product on a plain sheet or table.",
            used_fallback=False,
        )
    else:
        passed = bool(global_border_val <= clutter_t)
        return CheckResult(
            name="clutter",
            passed=passed,
            value=round(global_border_val, 4),
            tip="" if passed else "Background looks messy. Place the product on a plain sheet or table.",
            used_fallback=False,
        )


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
