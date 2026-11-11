from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from studio.coach.checks import compute_side_contact
from studio.coach.config import CoachConfig


@dataclass
class SegmentResult:
    """Result of product segmentation and mask post-processing."""

    mask: np.ndarray  # uint8 0/255 binary mask, same (H, W) as image
    rgba: np.ndarray  # uint8 BGRA 4-channel image with feathered alpha
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area_fraction: float  # product area / total image area
    touches_border: bool  # touches border per cutoff_side_fraction
    failed: bool  # True if area_fraction < 0.005 or segmentation failed
    side_contact: Dict[str, float] = field(
        default_factory=lambda: {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}
    )
    model_used: str = "isnet-general-use"


_SESSIONS: Dict[str, Any] = {}


def _ensure_rembg_home() -> None:
    """Point REMBG_HOME to project models directory if not already configured."""
    if "REMBG_HOME" not in os.environ and "U2NET_HOME" not in os.environ:
        project_root = Path(__file__).resolve().parents[3]
        models_dir = project_root / "data" / "models" / "rembg"
        if models_dir.exists():
            os.environ["REMBG_HOME"] = str(models_dir)


def get_rembg_session(model: str = "isnet-general-use") -> Any:
    """Retrieve or lazily create and cache a rembg session with fallback.

    Args:
        model: Model identifier (default 'isnet-general-use', fallback 'u2net').

    Returns:
        A loaded rembg session object.
    """
    _ensure_rembg_home()

    if model in _SESSIONS:
        return _SESSIONS[model]

    from rembg import new_session

    session = None
    try:
        session = new_session(model)
        _SESSIONS[model] = session
    except Exception as exc:
        if model != "u2net":
            try:
                session = new_session("u2net")
                _SESSIONS["u2net"] = session
                _SESSIONS[model] = session
            except Exception:
                raise exc
        else:
            raise exc

    return session


def postprocess_mask(
    alpha: np.ndarray,
    img_bgr: Optional[np.ndarray] = None,
    border_margin_px: int = 4,
    cutoff_side_fraction: float = 0.08,
) -> SegmentResult:
    """Post-process a raw alpha mask into a clean binary mask and feathered RGBA.

    1. Threshold at 128.
    2. Keep largest CC + any CC with area >= 10% of largest (multi-part products).
    3. Morphological close (5x5 ellipse) to fill small holes.
    4. Compute area fraction. If < 0.005, return empty mask and failed=True.
    5. Compute per-side border contact (top, bottom, left, right) and touches_border.
    6. Compute bounding box.
    7. 2 px Gaussian feather on final alpha used for rgba; mask stays hard binary.

    Args:
        alpha: 2D uint8 predicted mask or alpha array from rembg.
        img_bgr: Optional original BGR image for creating the RGBA result.
        border_margin_px: Border margin in pixels for boundary contact check.
        cutoff_side_fraction: Per-side contact fraction threshold for cut-off.

    Returns:
        SegmentResult with binary mask, BGRA image, bbox, metrics, and failed flag.
    """
    empty_side_contact = {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}

    if alpha is None or not isinstance(alpha, np.ndarray) or alpha.size == 0 or len(alpha.shape) < 2:
        return SegmentResult(
            mask=np.zeros((0, 0), dtype=np.uint8),
            rgba=np.zeros((0, 0, 4), dtype=np.uint8),
            bbox=(0, 0, 0, 0),
            area_fraction=0.0,
            touches_border=False,
            failed=True,
            side_contact=empty_side_contact,
        )

    h, w = alpha.shape[:2]
    total_px = h * w
    if total_px == 0:
        return SegmentResult(
            mask=np.zeros((0, 0), dtype=np.uint8),
            rgba=np.zeros((0, 0, 4), dtype=np.uint8),
            bbox=(0, 0, 0, 0),
            area_fraction=0.0,
            touches_border=False,
            failed=True,
            side_contact=empty_side_contact,
        )

    # 1. Threshold at 128
    binary = (alpha >= 128).astype(np.uint8) * 255

    # 2. Connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        empty_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        return SegmentResult(
            mask=empty_mask,
            rgba=empty_rgba,
            bbox=(0, 0, 0, 0),
            area_fraction=0.0,
            touches_border=False,
            failed=True,
            side_contact=empty_side_contact,
        )

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = int(np.max(areas))
    if max_area == 0:
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        empty_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        return SegmentResult(
            mask=empty_mask,
            rgba=empty_rgba,
            bbox=(0, 0, 0, 0),
            area_fraction=0.0,
            touches_border=False,
            failed=True,
            side_contact=empty_side_contact,
        )

    min_area_thresh = 0.10 * max_area
    keep_labels = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area_thresh]
    filtered_mask = np.isin(labels, keep_labels).astype(np.uint8) * 255

    # 3. Morphological close to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)

    # Check area fraction
    fg_px = int(np.count_nonzero(closed_mask))
    area_fraction = float(fg_px / total_px)

    # If no foreground found (< 0.005), mark failed=True
    if area_fraction < 0.005:
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        empty_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        return SegmentResult(
            mask=empty_mask,
            rgba=empty_rgba,
            bbox=(0, 0, 0, 0),
            area_fraction=round(area_fraction, 4),
            touches_border=False,
            failed=True,
            side_contact=empty_side_contact,
        )

    # Bounding box (x, y, w, h)
    bx, by, bw, bh = cv2.boundingRect(closed_mask)
    bbox = (int(bx), int(by), int(bw), int(bh))

    # Per-side contact and touches_border check
    side_contact = compute_side_contact(closed_mask, border_margin_px=border_margin_px)
    touches_border = bool(any(frac >= cutoff_side_fraction for frac in side_contact.values()))

    # 4. 2 px Gaussian feather on final alpha used for rgba
    feathered_alpha = cv2.GaussianBlur(closed_mask, (0, 0), sigmaX=2.0, sigmaY=2.0)

    if img_bgr is not None and img_bgr.shape[:2] == (h, w):
        if len(img_bgr.shape) == 2:
            base_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4:
            base_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
        else:
            base_bgr = img_bgr
        rgba = np.dstack([base_bgr, feathered_alpha])
    else:
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 3] = feathered_alpha

    return SegmentResult(
        mask=closed_mask,
        rgba=rgba,
        bbox=bbox,
        area_fraction=round(area_fraction, 4),
        touches_border=touches_border,
        failed=False,
        side_contact=side_contact,
    )


def _run_single_segmentation(
    img_bgr: np.ndarray,
    model: str = "isnet-general-use",
    config: Optional[CoachConfig] = None,
) -> SegmentResult:
    """Execute segmentation with a single model."""
    border_margin_px = config.border_margin_px if config else 4
    cutoff_side_fraction = config.cutoff_side_fraction if config else 0.08
    h, w = img_bgr.shape[:2]

    session = get_rembg_session(model)
    from rembg import remove

    if len(img_bgr.shape) == 2:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    elif img_bgr.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGB)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    raw_mask = remove(img_rgb, session=session, only_mask=True)
    if not isinstance(raw_mask, np.ndarray):
        raw_mask = np.array(raw_mask)

    res = postprocess_mask(
        raw_mask,
        img_bgr=img_bgr,
        border_margin_px=border_margin_px,
        cutoff_side_fraction=cutoff_side_fraction,
    )
    res.model_used = model
    return res


def segment_product(
    img_bgr: np.ndarray,
    model: str = "isnet-general-use",
    config: Optional[CoachConfig] = None,
) -> SegmentResult:
    """Segment the product from the background using rembg with robust post-processing.

    If primary model returns area_fraction < 0.005, retries once with 'u2net'
    and keeps whichever result has the larger area_fraction (ignoring results > 0.95).
    Never raises an exception on corrupt or invalid inputs.

    Args:
        img_bgr: Input photo as BGR uint8 NumPy array.
        model: Primary model name for rembg (default 'isnet-general-use').
        config: Optional CoachConfig configuration thresholds.

    Returns:
        SegmentResult with mask, rgba, bbox, area_fraction, touches_border, failed, side_contact, model_used.
    """
    empty_side_contact = {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}

    if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0 or len(img_bgr.shape) < 2:
        return SegmentResult(
            mask=np.zeros((0, 0), dtype=np.uint8),
            rgba=np.zeros((0, 0, 4), dtype=np.uint8),
            bbox=(0, 0, 0, 0),
            area_fraction=0.0,
            touches_border=False,
            failed=True,
            side_contact=empty_side_contact,
            model_used=model,
        )

    h, w = img_bgr.shape[:2]

    try:
        res1 = _run_single_segmentation(img_bgr, model=model, config=config)
    except Exception:
        res1 = SegmentResult(
            mask=np.zeros((h, w), dtype=np.uint8),
            rgba=np.zeros((h, w, 4), dtype=np.uint8),
            bbox=(0, 0, 0, 0),
            area_fraction=0.0,
            touches_border=False,
            failed=True,
            side_contact=empty_side_contact,
            model_used=model,
        )

    # Segmenter fallback: if primary model returns area_fraction < 0.005, retry with u2net
    if res1.area_fraction < 0.005 and model != "u2net":
        try:
            res2 = _run_single_segmentation(img_bgr, model="u2net", config=config)
        except Exception:
            res2 = SegmentResult(
                mask=np.zeros((h, w), dtype=np.uint8),
                rgba=np.zeros((h, w, 4), dtype=np.uint8),
                bbox=(0, 0, 0, 0),
                area_fraction=0.0,
                touches_border=False,
                failed=True,
                side_contact=empty_side_contact,
                model_used="u2net",
            )

        def _score(r: SegmentResult) -> float:
            if r.failed or r.area_fraction > 0.95:
                return -1.0
            return r.area_fraction

        score1 = _score(res1)
        score2 = _score(res2)
        if score2 > score1:
            return res2
        else:
            return res1

    return res1
