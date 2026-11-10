from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional, Union

import numpy as np

from studio.coach.coach import CoachReport, assess_with_segmentation
from studio.coach.config import CoachConfig
from studio.coach.io import load_image
from studio.compose.compose import _resolve_preset, composite, place_product
from studio.segment.segmenter import SegmentResult


@dataclass
class PipelineResult:
    """Consolidated result of the studio image pipeline execution."""

    ok: bool
    report: CoachReport
    image: Optional[np.ndarray]
    segment: SegmentResult
    seconds: float


def run_pipeline(
    image_path: Union[str, Path],
    scene: str = "white",
    marketplace: str = "square_1000",
    force: bool = False,
    config: Optional[CoachConfig] = None,
) -> PipelineResult:
    """Run the photo studio generation pipeline.

    Args:
        image_path: Path to the input product photo.
        scene: Scene type (currently only 'white' is implemented).
        marketplace: Preset canvas name (e.g. 'square_1000', 'square_2000', 'portrait_4x5').
        force: If True, generate the composited image even if photo fails coach checks.
        config: Optional CoachConfig configuration thresholds.

    Returns:
        PipelineResult containing ok status, coach report, output image, segment result, and seconds.

    Raises:
        NotImplementedError: If scene is not 'white' ("needs Phase 3").
    """
    if scene != "white":
        raise NotImplementedError("needs Phase 3")

    start_time = time.time()
    img_bgr = load_image(image_path)
    report, seg = assess_with_segmentation(img_bgr, config=config)

    if not report.ok and not force:
        elapsed = time.time() - start_time
        return PipelineResult(
            ok=False,
            report=report,
            image=None,
            segment=seg,
            seconds=round(elapsed, 3),
        )

    canvas_w, canvas_h = _resolve_preset(marketplace)
    if seg.failed:
        out_image = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    else:
        canvas_rgba, canvas_mask = place_product(seg.rgba, preset=marketplace)
        bg = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
        out_image = composite(bg, canvas_rgba, canvas_mask, shadow=True)

    elapsed = time.time() - start_time
    return PipelineResult(
        ok=report.ok,
        report=report,
        image=out_image,
        segment=seg,
        seconds=round(elapsed, 3),
    )
