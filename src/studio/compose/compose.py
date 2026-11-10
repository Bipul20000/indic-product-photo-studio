from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

from studio.coach.coach import CoachReport, assess_with_segmentation
from studio.coach.config import CoachConfig
from studio.segment.segmenter import SegmentResult

PRESETS: Dict[str, Tuple[int, int]] = {
    "square_1000": (1000, 1000),
    "square_2000": (2000, 2000),
    "portrait_4x5": (1080, 1350),
}


def _resolve_preset(preset: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
    """Resolve preset name or dimensions tuple into (width, height)."""
    if isinstance(preset, str):
        if preset not in PRESETS:
            raise KeyError(f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")
        return PRESETS[preset]
    elif isinstance(preset, (tuple, list)) and len(preset) == 2:
        return int(preset[0]), int(preset[1])
    else:
        raise ValueError(f"Invalid preset format: {preset}")


def place_product(
    rgba: np.ndarray,
    preset: Union[str, Tuple[int, int]] = "square_1000",
    fill: float = 0.80,
    vertical_offset: float = 0.03,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop product to bbox, resize to fill shorter canvas side, and center with vertical offset.

    Args:
        rgba: 4-channel BGRA image of the segmented product.
        preset: Preset identifier string or (width, height) tuple.
        fill: Fraction of shorter canvas side that longest product side should occupy (default 0.80).
        vertical_offset: Downward vertical offset fraction of canvas height (default 0.03).

    Returns:
        Tuple of (canvas_rgba, product_mask_on_canvas).
    """
    canvas_w, canvas_h = _resolve_preset(preset)

    if rgba is None or not isinstance(rgba, np.ndarray) or rgba.size == 0 or len(rgba.shape) < 3 or rgba.shape[2] < 4:
        return (
            np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8),
            np.zeros((canvas_h, canvas_w), dtype=np.uint8),
        )

    alpha = rgba[:, :, 3]
    pts = np.argwhere(alpha > 0)
    if pts.size == 0:
        return (
            np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8),
            np.zeros((canvas_h, canvas_w), dtype=np.uint8),
        )

    # 1. Crop to bbox
    y_min, x_min = pts.min(axis=0)
    y_max, x_max = pts.max(axis=0) + 1
    crop_h = y_max - y_min
    crop_w = x_max - x_min
    cropped = rgba[y_min:y_max, x_min:x_max]

    # 2. Resize so longest product side is fill * shorter canvas side
    shorter_canvas_side = min(canvas_w, canvas_h)
    target_longest = fill * shorter_canvas_side
    longest_crop_side = max(crop_h, crop_w)
    scale = target_longest / max(longest_crop_side, 1)

    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=interp)

    # 3. Center horizontally and vertically (with vertical_offset downward)
    x_start = (canvas_w - new_w) // 2
    y_offset_px = int(round(vertical_offset * canvas_h))
    y_start = (canvas_h - new_h) // 2 + y_offset_px

    x_start = max(0, min(canvas_w - new_w, x_start))
    y_start = max(0, min(canvas_h - new_h, y_start))

    canvas_rgba = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    canvas_rgba[y_start : y_start + new_h, x_start : x_start + new_w] = resized

    product_mask_on_canvas = (canvas_rgba[:, :, 3] >= 128).astype(np.uint8) * 255
    return canvas_rgba, product_mask_on_canvas


def add_soft_shadow(
    canvas_bgr: np.ndarray,
    product_mask_on_canvas: np.ndarray,
    opacity: float = 0.30,
    blur_sigma: float = 18.0,
    offset_y: int = 12,
) -> np.ndarray:
    """Create a soft drop shadow from the product mask and multiply onto canvas.

    Args:
        canvas_bgr: Background canvas in BGR uint8 format.
        product_mask_on_canvas: 2D uint8 mask on the canvas.
        opacity: Maximum shadow opacity factor (default 0.30).
        blur_sigma: Gaussian blur sigma for the shadow (default 18).
        offset_y: Downward vertical pixel shift (default 12).

    Returns:
        Shadowed canvas BGR array.
    """
    if canvas_bgr is None or product_mask_on_canvas is None:
        return canvas_bgr

    h, w = canvas_bgr.shape[:2]
    if h == 0 or w == 0:
        return canvas_bgr

    if not np.any(product_mask_on_canvas > 0):
        return canvas_bgr.copy()

    # Shift down
    m_mat = np.float32([[1, 0, 0], [0, 1, offset_y]])
    shifted = cv2.warpAffine(
        product_mask_on_canvas,
        m_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Blur
    ksize = int(6 * blur_sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(shifted, (ksize, ksize), blur_sigma)

    # Multiply
    shadow_factor = 1.0 - (blurred.astype(np.float32) / 255.0) * opacity
    shadowed = np.clip(canvas_bgr.astype(np.float32) * shadow_factor[:, :, np.newaxis], 0, 255).astype(np.uint8)
    return shadowed


def composite(
    background_bgr: np.ndarray,
    rgba_on_canvas: np.ndarray,
    mask: Optional[np.ndarray] = None,
    shadow: bool = True,
    opacity: float = 0.30,
    blur_sigma: float = 18.0,
    offset_y: int = 12,
) -> np.ndarray:
    """Composite original product pixels over background using feathered alpha and optional drop shadow.

    Args:
        background_bgr: Canvas background in BGR format.
        rgba_on_canvas: Placed product RGBA on canvas.
        mask: Optional binary product mask on canvas (derived from alpha if omitted).
        shadow: Whether to generate and apply soft drop shadow under product.
        opacity: Shadow opacity (default 0.30).
        blur_sigma: Shadow blur sigma (default 18.0).
        offset_y: Shadow downward offset (default 12).

    Returns:
        Final composited image in BGR format.
    """
    if background_bgr is None or rgba_on_canvas is None:
        raise ValueError("background_bgr and rgba_on_canvas must be provided.")

    h, w = rgba_on_canvas.shape[:2]
    if background_bgr.shape[:2] != (h, w):
        bg = cv2.resize(background_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
    else:
        bg = background_bgr.copy()

    if shadow:
        shadow_mask = mask if mask is not None else (rgba_on_canvas[:, :, 3] >= 128).astype(np.uint8) * 255
        bg = add_soft_shadow(bg, shadow_mask, opacity=opacity, blur_sigma=blur_sigma, offset_y=offset_y)

    prod_bgr = rgba_on_canvas[:, :, :3].astype(np.float32)
    alpha = (rgba_on_canvas[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]

    out = prod_bgr * alpha + bg.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_white_background(
    img_bgr: np.ndarray,
    preset: Union[str, Tuple[int, int]] = "square_1000",
    config: Optional[CoachConfig] = None,
) -> Tuple[np.ndarray, CoachReport, SegmentResult]:
    """Segment product, place on pure white canvas with soft shadow, and composite.

    Args:
        img_bgr: Input photo in BGR format.
        preset: Canvas preset identifier or (width, height) tuple (default 'square_1000').
        config: Optional CoachConfig configuration.

    Returns:
        Tuple of (output_bgr_image, CoachReport, SegmentResult).
    """
    report, seg = assess_with_segmentation(img_bgr, config=config)
    canvas_w, canvas_h = _resolve_preset(preset)

    if seg.failed:
        blank = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
        return blank, report, seg

    canvas_rgba, canvas_mask = place_product(seg.rgba, preset=preset)
    bg = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    out_bgr = composite(bg, canvas_rgba, canvas_mask, shadow=True)
    return out_bgr, report, seg
