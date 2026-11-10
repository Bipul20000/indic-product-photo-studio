"""Product segmentation module using rembg with morphological refinement."""

from studio.segment.segmenter import (
    SegmentResult,
    get_rembg_session,
    postprocess_mask,
    segment_product,
)

__all__ = [
    "SegmentResult",
    "get_rembg_session",
    "postprocess_mask",
    "segment_product",
]
