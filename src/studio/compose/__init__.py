"""Compositing module for product placement, drop shadows, and canvas presets."""

from studio.compose.compose import (
    PRESETS,
    add_soft_shadow,
    composite,
    make_white_background,
    place_product,
)

__all__ = [
    "PRESETS",
    "add_soft_shadow",
    "composite",
    "make_white_background",
    "place_product",
]
