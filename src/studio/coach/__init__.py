"""Shoot Coach package for phone photo quality checking."""

from studio.coach.checks import CheckResult
from studio.coach.coach import CoachReport, assess_photo
from studio.coach.config import CoachConfig
from studio.coach.io import load_image

__all__ = [
    "assess_photo",
    "CoachConfig",
    "CoachReport",
    "CheckResult",
    "load_image",
]
