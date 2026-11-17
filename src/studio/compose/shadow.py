"""Realistic dual-layer ground contact and ambient occlusion shadow synthesis.

Generates:
1. A tight, dense Contact Shadow (Ambient Occlusion) right at the base of the product
   where it touches the surface.
2. A soft, feathered Diffuse Shadow extending downward for natural depth.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


def add_realistic_shadow(
    canvas_bgr: np.ndarray,
    product_mask_on_canvas: np.ndarray,
    contact_opacity: float = 0.65,
    diffuse_opacity: float = 0.28,
    diffuse_blur_sigma: float = 20.0,
    diffuse_offset_y: int = 14,
    contact_blur_sigma: float = 4.0,
    contact_offset_y: int = 3,
) -> np.ndarray:
    """Apply dual-layer realistic contact shadow and diffuse drop shadow to background.

    Args:
        canvas_bgr: Background canvas in BGR uint8 format.
        product_mask_on_canvas: 2D uint8 mask on the canvas (0 or 255).
        contact_opacity: Maximum opacity for the tight contact occlusion shadow (default 0.65).
        diffuse_opacity: Maximum opacity for the soft diffuse shadow (default 0.28).
        diffuse_blur_sigma: Blur sigma for diffuse shadow (default 20.0).
        diffuse_offset_y: Downward pixel shift for diffuse shadow (default 14).
        contact_blur_sigma: Blur sigma for tight contact shadow (default 4.0).
        contact_offset_y: Downward pixel shift for tight contact shadow (default 3).

    Returns:
        Shadowed canvas BGR array with realistic ground contact.
    """
    if canvas_bgr is None or product_mask_on_canvas is None:
        return canvas_bgr

    h, w = canvas_bgr.shape[:2]
    if h == 0 or w == 0:
        return canvas_bgr

    pts = np.argwhere(product_mask_on_canvas > 0)
    if pts.size == 0:
        return canvas_bgr.copy()

    y_min, x_min = pts.min(axis=0)
    y_max, x_max = pts.max(axis=0) + 1
    prod_h = y_max - y_min

    # 1. Base Diffuse Shadow (broad soft drop shadow)
    m_diffuse = np.float32([[1, 0, 0], [0, 1, diffuse_offset_y]])
    shifted_diffuse = cv2.warpAffine(
        product_mask_on_canvas,
        m_diffuse,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    k_diff = int(6 * diffuse_blur_sigma + 1) | 1
    blurred_diffuse = cv2.GaussianBlur(shifted_diffuse, (k_diff, k_diff), diffuse_blur_sigma)
    diffuse_factor = 1.0 - (blurred_diffuse.astype(np.float32) / 255.0) * diffuse_opacity

    # 2. Tight Contact Shadow (Ambient Occlusion at base)
    # Extract only the lowest 15% footprint of the product mask
    base_threshold_y = y_max - max(10, int(0.15 * prod_h))
    contact_mask = np.zeros_like(product_mask_on_canvas)
    contact_mask[base_threshold_y:y_max, :] = product_mask_on_canvas[base_threshold_y:y_max, :]

    # Vertically compress and horizontally dilate slightly to mimic surface contact
    m_contact = np.float32([[1, 0, 0], [0, 1, contact_offset_y]])
    shifted_contact = cv2.warpAffine(
        contact_mask,
        m_contact,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # Anisotropic footprint: compress vertically
    k_cont_w = int(contact_blur_sigma * 6 + 1) | 1
    k_cont_h = int(contact_blur_sigma * 3 + 1) | 1
    blurred_contact = cv2.GaussianBlur(shifted_contact, (k_cont_w, k_cont_h), contact_blur_sigma)
    contact_factor = 1.0 - (blurred_contact.astype(np.float32) / 255.0) * contact_opacity

    # 3. Combined Shadow Multiplier
    total_shadow = np.clip(diffuse_factor * contact_factor, 0.0, 1.0)[:, :, np.newaxis]
    shadowed_bgr = np.clip(canvas_bgr.astype(np.float32) * total_shadow, 0, 255).astype(np.uint8)
    return shadowed_bgr
