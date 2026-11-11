"""Shoot Coach package for phone photo quality checking."""

from studio.coach.checks import CheckResult, check_framing, compute_side_contact
from studio.coach.coach import CoachReport, assess_photo, assess_with_segmentation
from studio.coach.config import CoachConfig
from studio.coach.io import load_image

__all__ = [
    "assess_photo",
    "assess_with_segmentation",
    "check_framing",
    "compute_side_contact",
    "CoachConfig",
    "CoachReport",
    "CheckResult",
    "load_image",
]
