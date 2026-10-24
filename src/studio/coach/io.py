from pathlib import Path
from typing import Union
import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass


def load_image(path: Union[str, Path], max_longest_side: int = 1600) -> np.ndarray:
    """Load an image from disk into a BGR uint8 NumPy array.

    - Registers HEIC support
    - Applies EXIF orientation transpose
    - Converts RGB to BGR
    - Downscales so the longest dimension is at most max_longest_side (default 1600 px)

    Args:
        path: File path to image.
        max_longest_side: Maximum length for the longest image dimension.

    Returns:
        np.ndarray: BGR image array with uint8 dtype.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Image not found at path: {path}")

    with Image.open(path_obj) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        rgb_arr = np.array(img, dtype=np.uint8)

    bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

    h, w = bgr_arr.shape[:2]
    longest = max(h, w)
    if longest > max_longest_side:
        scale = max_longest_side / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        bgr_arr = cv2.resize(bgr_arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return bgr_arr
