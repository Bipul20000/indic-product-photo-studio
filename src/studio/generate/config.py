from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

REQUIRED_SCENES = [
    "diwali",
    "raksha_bandhan",
    "holi",
    "wedding",
    "indian_home",
    "minimal",
]


def _find_scenes_file(path: Union[str, Path] = "configs/scenes.yaml") -> Path:
    """Locate scenes.yaml configuration file."""
    p = Path(path)
    if p.exists():
        return p
    repo_p = Path(__file__).resolve().parents[3] / "configs" / "scenes.yaml"
    if repo_p.exists():
        return repo_p
    return p


def load_scenes(path: Union[str, Path] = "configs/scenes.yaml") -> Dict[str, Any]:
    """Load and validate scenes configuration YAML file.

    Validates that:
    1. A non-empty negative prompt exists.
    2. All six required scenes exist with non-empty prompts.

    Args:
        path: Path to scenes.yaml file.

    Returns:
        Validated scenes configuration dictionary.

    Raises:
        FileNotFoundError: If configuration file does not exist.
        ValueError: If validation fails for required fields or scenes.
    """
    resolved_path = _find_scenes_file(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Scenes config file not found at: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Invalid scenes.yaml format: top-level structure must be a dictionary.")

    negative = data.get("negative")
    if not isinstance(negative, str) or not negative.strip():
        raise ValueError("scenes.yaml must define a non-empty 'negative' prompt string.")

    scenes = data.get("scenes")
    if not isinstance(scenes, dict):
        raise ValueError("scenes.yaml must define a 'scenes' dictionary.")

    for name in REQUIRED_SCENES:
        if name not in scenes:
            raise ValueError(f"Required scene '{name}' missing from scenes.yaml.")
        scene_info = scenes[name]
        if not isinstance(scene_info, dict) or "prompt" not in scene_info:
            raise ValueError(f"Scene '{name}' must be a dict with a 'prompt' field.")
        prompt = scene_info.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Scene '{name}' must have a non-empty string prompt.")

    return data


@dataclass
class GenConfig:
    """Configuration for ControlNet Inpaint background generation."""

    base_models: List[str] = field(
        default_factory=lambda: [
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "runwayml/stable-diffusion-v1-5",
        ]
    )
    controlnet_id: str = "lllyasviel/control_v11p_sd15_inpaint"
    steps: int = 20
    guidance: float = 7.0
    controlnet_scale: float = 0.5
    eta: float = 1.0
    lora_path: Optional[str] = None
    lora_scale: float = 0.8
    erode_px: int = 6
