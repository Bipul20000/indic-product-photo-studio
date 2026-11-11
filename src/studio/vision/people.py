"""Shared person detection using YOLOv8n with OpenCV HOG fallback."""

import os
from typing import Any, List, Optional, Tuple, Union
import warnings

import cv2
import numpy as np
from PIL import Image

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

# Lazy model references
_YOLO_MODEL = None
_YOLO_TRIED = False
_USE_HOG = False
_HOG_WARNING_PRINTED = False


def get_person_detector():
    """Lazily load YOLOv8n or fallback to OpenCV HOG."""
    global _YOLO_MODEL, _YOLO_TRIED, _USE_HOG, _HOG_WARNING_PRINTED
    if not _YOLO_TRIED:
        _YOLO_TRIED = True
        try:
            from ultralytics import YOLO

            _YOLO_MODEL = YOLO("yolov8n.pt")
        except Exception as e:
            if not _HOG_WARNING_PRINTED:
                print(f"WARNING: Could not load ultralytics YOLO ({e}). Falling back to OpenCV HOG detector.")
                _HOG_WARNING_PRINTED = True
            _USE_HOG = True
    return _YOLO_MODEL, _USE_HOG


def _to_bgr_array(pil_or_bgr_image: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Ensure image is a 3-channel BGR numpy array."""
    if isinstance(pil_or_bgr_image, Image.Image):
        rgb = np.array(pil_or_bgr_image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(pil_or_bgr_image, np.ndarray):
        if len(pil_or_bgr_image.shape) == 2:
            return cv2.cvtColor(pil_or_bgr_image, cv2.COLOR_GRAY2BGR)
        elif pil_or_bgr_image.shape[2] == 4:
            return cv2.cvtColor(pil_or_bgr_image, cv2.COLOR_BGRA2BGR)
        return pil_or_bgr_image
    else:
        raise TypeError(f"Expected PIL Image or np.ndarray, got {type(pil_or_bgr_image)}")


def detect_people(
    pil_or_bgr_image: Union[Image.Image, np.ndarray],
    conf: float = 0.35,
    min_box_area: float = 0.01,
    ignore_mask: Optional[np.ndarray] = None,
    model: Optional[Any] = None,
    use_hog: bool = False,
) -> List[Tuple[float, float, float, float]]:
    """Detect people in an image and return bounding boxes as (x1, y1, x2, y2).

    Args:
        pil_or_bgr_image: PIL Image or BGR numpy array.
        conf: Confidence threshold for person detection (default 0.35).
        min_box_area: Minimum fraction of total image area for a box (default 0.01).
        ignore_mask: Optional mask (2D array, 255/1 for product). If > 50% of the box
            area lies inside the product mask, the detection is ignored.
        model: Optional YOLO model override (useful for testing/mocking).
        use_hog: If True, force using OpenCV HOG detector.

    Returns:
        List of bounding boxes (x1, y1, x2, y2) in pixel coordinates.
    """
    img_bgr = _to_bgr_array(pil_or_bgr_image)
    img_h, img_w = img_bgr.shape[:2]
    img_area = img_h * img_w
    if img_area <= 0:
        return []

    # Prepare ignore_mask if provided
    mask_arr = None
    if ignore_mask is not None:
        mask_arr = np.array(ignore_mask)
        if mask_arr.ndim > 2:
            mask_arr = mask_arr[:, :, 0]
        if mask_arr.shape[:2] != (img_h, img_w):
            mask_arr = cv2.resize(mask_arr.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    boxes_out: List[Tuple[float, float, float, float]] = []

    # Run YOLO or HOG
    if model is None and not use_hog:
        m, hog_flag = get_person_detector()
        if hog_flag or m is None:
            use_hog = True
        else:
            model = m

    if not use_hog and model is not None:
        results = model(img_bgr, verbose=False)
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                box_conf = float(box.conf[0].item())
                cls_name = model.names.get(cls_id, "") if hasattr(model, "names") else ""
                if (cls_id == 0 or cls_name == "person") and box_conf >= conf:
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    box_area = w * h
                    if (box_area / img_area) < min_box_area:
                        continue

                    # Check ignore mask overlap (> 50% inside product mask)
                    if mask_arr is not None and box_area > 0:
                        ix1 = max(0, min(img_w, int(round(x1))))
                        iy1 = max(0, min(img_h, int(round(y1))))
                        ix2 = max(0, min(img_w, int(round(x2))))
                        iy2 = max(0, min(img_h, int(round(y2))))
                        if ix2 > ix1 and iy2 > iy1:
                            sub_mask = mask_arr[iy1:iy2, ix1:ix2]
                            overlap_pixels = int(np.sum(sub_mask > 0))
                            overlap_fraction = overlap_pixels / float(box_area)
                            if overlap_fraction > 0.50:
                                continue

                    boxes_out.append((x1, y1, x2, y2))
    else:
        # Fallback to OpenCV HOG
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        rects, weights = hog.detectMultiScale(img_bgr, winStride=(8, 8), padding=(4, 4), scale=1.05)
        for (rx, ry, rw, rh), weight in zip(rects, weights):
            w_val = float(weight[0]) if hasattr(weight, "__iter__") else float(weight)
            if w_val < conf:
                continue
            box_area = float(rw * rh)
            if (box_area / img_area) < min_box_area:
                continue

            x1, y1, x2, y2 = float(rx), float(ry), float(rx + rw), float(ry + rh)

            # Check ignore mask overlap
            if mask_arr is not None and box_area > 0:
                ix1 = max(0, min(img_w, int(round(x1))))
                iy1 = max(0, min(img_h, int(round(y1))))
                ix2 = max(0, min(img_w, int(round(x2))))
                iy2 = max(0, min(img_h, int(round(y2))))
                if ix2 > ix1 and iy2 > iy1:
                    sub_mask = mask_arr[iy1:iy2, ix1:ix2]
                    overlap_pixels = int(np.sum(sub_mask > 0))
                    overlap_fraction = overlap_pixels / float(box_area)
                    if overlap_fraction > 0.50:
                        continue

            boxes_out.append((x1, y1, x2, y2))

    return boxes_out


def contains_person(
    pil_or_bgr_image: Union[Image.Image, np.ndarray],
    conf: float = 0.35,
    min_box_area: float = 0.01,
    ignore_mask: Optional[np.ndarray] = None,
    model: Optional[Any] = None,
) -> bool:
    """Return True if image contains any person matching confidence and area thresholds."""
    boxes = detect_people(
        pil_or_bgr_image=pil_or_bgr_image,
        conf=conf,
        min_box_area=min_box_area,
        ignore_mask=ignore_mask,
        model=model,
    )
    return len(boxes) > 0


# Compatibility helpers for build_lora_dataset and older tests
def check_person_yolo(
    img_bgr: np.ndarray,
    model: Any,
    conf_thresh: float = 0.35,
    min_area_fraction: float = 0.015,
    ignore_mask: Optional[np.ndarray] = None,
) -> bool:
    boxes = detect_people(
        img_bgr,
        conf=conf_thresh,
        min_box_area=min_area_fraction,
        ignore_mask=ignore_mask,
        model=model,
        use_hog=False,
    )
    return len(boxes) > 0


def check_person_hog(
    img_bgr: np.ndarray,
    min_area_fraction: float = 0.015,
    ignore_mask: Optional[np.ndarray] = None,
) -> bool:
    boxes = detect_people(
        img_bgr,
        conf=0.35,
        min_box_area=min_area_fraction,
        ignore_mask=ignore_mask,
        use_hog=True,
    )
    return len(boxes) > 0


def detect_person(
    img_bgr: np.ndarray,
    model: Optional[Any] = None,
    use_hog: bool = False,
    conf_thresh: float = 0.35,
    min_area_fraction: float = 0.015,
    ignore_mask: Optional[np.ndarray] = None,
) -> bool:
    boxes = detect_people(
        img_bgr,
        conf=conf_thresh,
        min_box_area=min_area_fraction,
        ignore_mask=ignore_mask,
        model=model,
        use_hog=use_hog,
    )
    return len(boxes) > 0
