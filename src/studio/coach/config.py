from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional


def _find_thresholds_file() -> Optional[Path]:
    """Locate configs/coach_thresholds.json in current directory or project root."""
    p = Path("configs/coach_thresholds.json")
    if p.exists():
        return p
    p_repo = Path(__file__).resolve().parents[3] / "configs" / "coach_thresholds.json"
    if p_repo.exists():
        return p_repo
    return None


@dataclass
class CoachConfig:
    """Configuration thresholds for Shoot Coach photo quality checks."""

    blur_min: float = 100.0
    blur_method: str = "tile_p90"
    dark_max_p95: float = 90.0
    bright_min_p5: float = 150.0
    min_side_px: int = 600
    clutter_max: float = 0.12
    min_product_area: float = 0.15
    max_product_area: float = 0.85
    border_margin_px: int = 4
    cutoff_side_fraction: float = 0.08
    border_band_fraction: float = 0.15
    blur_resize_side: int = 1024
    load_saved_thresholds: bool = True

    def __post_init__(self):
        """Load threshold overrides from configs/coach_thresholds.json if available."""
        if not self.load_saved_thresholds:
            return

        thresh_file = _find_thresholds_file()
        if thresh_file and thresh_file.exists():
            try:
                with open(thresh_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.items():
                    if hasattr(self, k):
                        setattr(self, k, v)
            except Exception:
                pass
