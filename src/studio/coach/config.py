from dataclasses import dataclass


@dataclass
class CoachConfig:
    """Configuration thresholds for Shoot Coach photo quality checks."""

    blur_min: float = 100.0
    dark_max_mean: float = 60.0
    bright_min_mean: float = 200.0
    clipped_max_fraction: float = 0.25
    min_side_px: int = 600
    clutter_max: float = 0.12
    min_product_area: float = 0.15
    max_product_area: float = 0.85
    border_margin_px: int = 4
    border_band_fraction: float = 0.15
    blur_resize_side: int = 1024
