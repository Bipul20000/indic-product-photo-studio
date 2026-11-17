"""Environmental lighting and ambient color harmonization for product compositing.

Bridges the "sticker effect" by transferring ambient background chromaticity
onto product boundaries (rim light / Fresnel bounce) and harmonizing global color temperature.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


def harmonize_product(
    rgba_on_canvas: np.ndarray,
    background_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    strength: float = 0.12,
    rim_width: int = 12,
    rim_strength: float = 0.28,
) -> np.ndarray:
    """Harmonize foreground product lighting and color temperature with background.

    Args:
        rgba_on_canvas: 4-channel BGRA image of product placed on canvas.
        background_bgr: 3-channel BGR image of the background canvas.
        mask: Optional 2D uint8 mask on canvas (derived from alpha if omitted).
        strength: Global color temperature shift strength [0.0, 1.0]. Default 0.12.
        rim_width: Pixel width of inner rim zone where background light bounces. Default 12.
        rim_strength: Ambient bounce intensity along product contour [0.0, 1.0]. Default 0.28.

    Returns:
        Harmonized 4-channel BGRA array with natural ambient light spill and color balance.
    """
    if rgba_on_canvas is None or background_bgr is None:
        return rgba_on_canvas

    if strength <= 0.0 and rim_strength <= 0.0:
        return rgba_on_canvas.copy()

    h, w = rgba_on_canvas.shape[:2]
    if background_bgr.shape[:2] != (h, w):
        bg = cv2.resize(background_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
    else:
        bg = background_bgr

    alpha = rgba_on_canvas[:, :, 3]
    prod_mask = mask if mask is not None else (alpha >= 128).astype(np.uint8) * 255

    pts = np.argwhere(prod_mask > 0)
    if pts.size == 0:
        return rgba_on_canvas.copy()

    prod_bgr = rgba_on_canvas[:, :, :3].copy()

    # 1. Extract ambient background profile surrounding the product
    # Dilate mask by 25px and subtract product mask to sample the surrounding light
    k_surround = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    surround_mask = cv2.dilate(prod_mask, k_surround)
    surround_ring = (surround_mask > 0) & (prod_mask == 0)

    if np.any(surround_ring):
        ambient_bgr = np.mean(bg[surround_ring], axis=0)  # (3,) float
    else:
        ambient_bgr = np.mean(bg, axis=(0, 1))

    # 2. Subtle Global Color Temperature Harmonization (in LAB space)
    if strength > 0.0:
        prod_lab = cv2.cvtColor(prod_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        ambient_1px = np.uint8([[ambient_bgr]])
        ambient_lab = cv2.cvtColor(ambient_1px, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)

        # Shift a* and b* (chromaticity) slightly toward ambient illuminant
        mask_bool = prod_mask > 0
        if np.any(mask_bool):
            prod_mean_a = np.mean(prod_lab[mask_bool, 1])
            prod_mean_b = np.mean(prod_lab[mask_bool, 2])

            shift_a = (ambient_lab[1] - prod_mean_a) * strength
            shift_b = (ambient_lab[2] - prod_mean_b) * strength

            prod_lab[mask_bool, 1] = np.clip(prod_lab[mask_bool, 1] + shift_a, 0, 255)
            prod_lab[mask_bool, 2] = np.clip(prod_lab[mask_bool, 2] + shift_b, 0, 255)

            prod_bgr = cv2.cvtColor(prod_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # 3. Ambient Rim Bounce (Fresnel Light Bleed on Outer Edges)
    if rim_strength > 0.0 and rim_width > 0:
        # Inner rim: pixels within rim_width of the contour
        k_rim = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rim_width * 2 + 1, rim_width * 2 + 1))
        eroded_mask = cv2.erode(prod_mask, k_rim)
        rim_raw = (prod_mask.astype(np.float32) - eroded_mask.astype(np.float32)) / 255.0

        # Feather the rim inward with Gaussian blur
        k_blur = max(3, (rim_width // 2) * 2 + 1)
        rim_feathered = cv2.GaussianBlur(rim_raw, (k_blur, k_blur), rim_width / 3.0)
        rim_weight = np.clip(rim_feathered * rim_strength, 0.0, 1.0)[:, :, np.newaxis]

        # Light bounce from smoothed background
        bg_soft = cv2.GaussianBlur(bg, (21, 21), 7.0).astype(np.float32)
        blended_bgr = prod_bgr.astype(np.float32) * (1.0 - rim_weight) + bg_soft * rim_weight
        prod_bgr = np.clip(blended_bgr, 0, 255).astype(np.uint8)

    harmonized_rgba = np.dstack([prod_bgr, alpha])
    return harmonized_rgba
