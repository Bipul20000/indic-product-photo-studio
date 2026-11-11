from typing import Tuple, Union

import cv2
import numpy as np
from PIL import Image

from studio.compose.compose import PRESETS


def gen_size(preset: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
    """Determine generation dimensions (width, height) for SD 1.5 inpainting.

    Square presets map to (512, 512).
    Portrait 4x5 presets map to (512, 640).

    Args:
        preset: Preset identifier string or (width, height) tuple.

    Returns:
        (width, height) tuple.
    """
    if isinstance(preset, str):
        if preset in ("square_1000", "square_2000"):
            return (512, 512)
        elif preset == "portrait_4x5":
            return (512, 640)
        elif preset in PRESETS:
            pw, ph = PRESETS[preset]
            if pw == ph:
                return (512, 512)
            return (512, int(round(512 * ph / pw)))
        else:
            return (512, 512)
    elif isinstance(preset, (tuple, list)) and len(preset) == 2:
        pw, ph = int(preset[0]), int(preset[1])
        if pw == ph:
            return (512, 512)
        elif (pw, ph) == (1080, 1350) or abs((pw / ph) - (4 / 5)) < 0.02:
            return (512, 640)
        else:
            return (512, int(round(512 * ph / pw)))
    else:
        return (512, 512)


def build_inpaint_inputs(
    canvas_bgr: np.ndarray,
    product_mask: np.ndarray,
    size: Tuple[int, int],
    erode_px: int = 6,
) -> Tuple[Image.Image, Image.Image, np.ndarray]:
    """Prepare scaled image, inpaint mask, and product mask for inpainting.

    Resizes canvas and mask to size. Erodes the product mask by erode_px so
    the generated background slightly overlaps the product boundary, eliminating
    white halos.

    Args:
        canvas_bgr: Full-resolution canvas image in BGR format.
        product_mask: Binary mask where product = 255, background = 0.
        size: Target (width, height) for inpainting.
        erode_px: Pixel radius to erode product mask boundaries (default 6).

    Returns:
        Tuple of:
            init_image: PIL RGB image at target size.
            inpaint_mask: PIL L image (255 = generate/background, 0 = keep/product).
            small_product_mask: Eroded binary product mask (uint8 array, 255 product, 0 bg).
    """
    target_w, target_h = size
    h, w = canvas_bgr.shape[:2]

    # Resize canvas to inpaint size
    interp = cv2.INTER_AREA if (target_w <= w and target_h <= h) else cv2.INTER_LINEAR
    resized_canvas = cv2.resize(canvas_bgr, (target_w, target_h), interpolation=interp)
    init_rgb = cv2.cvtColor(resized_canvas, cv2.COLOR_BGR2RGB)
    init_image = Image.fromarray(init_rgb)

    # Resize mask
    resized_mask = cv2.resize(product_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    binary_mask = (resized_mask >= 128).astype(np.uint8) * 255

    # Erode product mask by erode_px
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        small_product_mask = cv2.erode(binary_mask, kernel, iterations=1)
    else:
        small_product_mask = binary_mask

    # Inpaint mask: 255 = area to generate (background), 0 = keep (product)
    inpaint_arr = (255 - small_product_mask).astype(np.uint8)
    inpaint_mask = Image.fromarray(inpaint_arr, mode="L")

    return init_image, inpaint_mask, small_product_mask


def make_inpaint_condition(
    init_image: Image.Image,
    inpaint_mask: Image.Image,
) -> np.ndarray:
    """Format inpaint conditioning image following diffusers ControlNet inpaint specs.

    Scales image to [0, 1] float32, sets masked pixels (inpaint_mask > 0.5) to -1.0,
    and returns 1x3xHxW array.

    Args:
        init_image: PIL RGB image.
        inpaint_mask: PIL L mask image (255 = area to inpaint).

    Returns:
        np.ndarray of shape (1, 3, H, W) with dtype float32.
    """
    image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
    mask = np.array(inpaint_mask.convert("L")).astype(np.float32) / 255.0

    # Diffusers specification: pixels where mask > 0.5 set to -1.0
    image[mask > 0.5] = -1.0
    # Shape: (1, 3, H, W)
    condition = np.expand_dims(image.transpose(2, 0, 1), axis=0)
    return condition
