from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple
import numpy as np

from studio.coach.checks import (
    CheckResult,
    check_blur,
    check_brightness,
    check_clutter,
    check_framing,
    check_resolution,
)
from studio.coach.config import CoachConfig


@dataclass
class CoachReport:
    """Consolidated assessment report containing verdicts and actionable tips."""

    ok: bool
    results: List[CheckResult] = field(default_factory=list)
    tips: List[str] = field(default_factory=list)


def assess_photo(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    config: Optional[CoachConfig] = None,
) -> CoachReport:
    """Assess image quality and framing for seller product photos.

    Runs resolution, blur, brightness, and clutter checks on `img`.
    If `mask` is provided, runs product framing check.

    This function is guaranteed to never raise on invalid or corrupt input,
    returning a failed CoachReport instead.

    Args:
        img: BGR uint8 NumPy array of the photo.
        mask: Optional binary product mask array (2D).
        config: Optional CoachConfig configuration thresholds.

    Returns:
        CoachReport: Assessment results with overall ok flag, check results, and tips.
    """
    if config is None:
        config = CoachConfig()

    try:
        if img is None or not isinstance(img, np.ndarray) or img.size == 0 or len(img.shape) < 2:
            fail_res = CheckResult(
                name="input_validity",
                passed=False,
                value=0.0,
                tip="Invalid image data. Please provide a valid photo.",
            )
            return CoachReport(ok=False, results=[fail_res], tips=[fail_res.tip])

        results: List[CheckResult] = []

        # Core image quality checks
        results.append(check_resolution(img, config))
        results.append(check_blur(img, config))
        results.append(check_brightness(img, mask=mask, config=config))
        results.append(check_clutter(img, mask=mask, config=config))

        # Framing check only if mask is provided
        if mask is not None:
            if isinstance(mask, np.ndarray) and mask.size > 0 and len(mask.shape) >= 2:
                results.append(check_framing(mask, config))
            else:
                results.append(
                    CheckResult(
                        name="framing",
                        passed=False,
                        value=0.0,
                        tip="Invalid product mask provided.",
                    )
                )

        ok = all(r.passed for r in results)
        tips = [r.tip for r in results if not r.passed and r.tip]
        return CoachReport(ok=ok, results=results, tips=tips)

    except Exception as exc:
        err_res = CheckResult(
            name="error",
            passed=False,
            value=0.0,
            tip=f"Error evaluating image: {str(exc)}",
        )
        return CoachReport(ok=False, results=[err_res], tips=[err_res.tip])


def assess_with_segmentation(
    img: np.ndarray,
    config: Optional[CoachConfig] = None,
    model: str = "isnet-general-use",
) -> tuple[CoachReport, Any]:
    """Segment image, run quality checks with framing assessment, and return report + segment result.

    If segmentation failed, adds a tip:
    'We could not find the product. Place it clearly in the center of a plain background.'
    """
    if config is None:
        config = CoachConfig()

    from studio.segment.segmenter import segment_product

    seg = segment_product(img, model=model, config=config)

    if seg.failed:
        report = assess_photo(img, mask=None, config=config)
        fail_check = CheckResult(
            name="framing",
            passed=False,
            value=0.0,
            tip="We could not find the product. Place it clearly in the center of a plain background.",
        )
        report.results.append(fail_check)
        report.ok = False
        if fail_check.tip not in report.tips:
            report.tips.append(fail_check.tip)
        return report, seg

    report = assess_photo(img, mask=seg.mask, config=config)
    return report, seg

