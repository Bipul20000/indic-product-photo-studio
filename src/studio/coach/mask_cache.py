from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from studio.coach.io import load_image
from studio.segment.segmenter import segment_product

CACHE_DIR = Path("data/coach_eval/mask_cache")


def get_mask(
    path_or_image: Union[str, Path, np.ndarray],
    key: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> np.ndarray:
    """Retrieve binary segmentation mask from disk cache, or compute and save it.

    Args:
        path_or_image: Image filepath (str/Path) or BGR uint8 NumPy array.
        key: Cache key (typically the image filename). Derived from path if omitted.
        cache_dir: Directory for cached mask png files (defaults to data/coach_eval/mask_cache).

    Returns:
        2D binary mask as uint8 array (0 / 255).
    """
    target_dir = cache_dir or CACHE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    if key is None:
        if isinstance(path_or_image, (str, Path)):
            key = Path(path_or_image).name
        else:
            raise ValueError("key must be provided when path_or_image is a NumPy array.")

    # Ensure key ends cleanly as <image_filename>.png
    filename = key if key.endswith(".png") else f"{key}.png"
    cache_file = target_dir / filename

    if isinstance(path_or_image, (str, Path)):
        img = load_image(path_or_image)
    else:
        img = path_or_image

    if cache_file.exists():
        cached = cv2.imread(str(cache_file), cv2.IMREAD_GRAYSCALE)
        if cached is not None and cached.shape == img.shape[:2]:
            return cached

    seg = segment_product(img)
    mask = seg.mask if seg.mask is not None and seg.mask.size > 0 else np.zeros(img.shape[:2], dtype=np.uint8)

    cv2.imwrite(str(cache_file), mask)
    return mask
