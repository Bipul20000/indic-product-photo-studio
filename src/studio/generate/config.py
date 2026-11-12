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
    lora_scale: Optional[float] = None
    erode_px: int = 6


def _find_lora_scales_file(path: Union[str, Path] = "configs/lora_scales.yaml") -> Path:
    """Locate lora_scales.yaml configuration file."""
    p = Path(path)
    if p.exists():
        return p
    repo_p = Path(__file__).resolve().parents[3] / "configs" / "lora_scales.yaml"
    if repo_p.exists():
        return repo_p
    return p


def load_lora_scales(path: Union[str, Path] = "configs/lora_scales.yaml") -> Dict[str, float]:
    """Load per-scene LoRA scale mapping from configs/lora_scales.yaml."""
    resolved_path = _find_lora_scales_file(path)
    if not resolved_path.exists():
        return {
            "diwali": 0.8,
            "raksha_bandhan": 0.8,
            "holi": 0.0,
            "wedding": 0.8,
            "indian_home": 0.8,
        }
    with open(resolved_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {str(k): float(v) for k, v in data.items() if isinstance(v, (int, float))}


def effective_lora_scale(
    scene: str,
    override: Optional[float] = None,
    scales_path: Union[str, Path] = "configs/lora_scales.yaml",
) -> float:
    """Determine the effective LoRA scale for a given scene.

    Args:
        scene: Scene name (e.g. 'diwali', 'holi', 'minimal').
        override: Optional global override for LoRA scale across all scenes.
        scales_path: Path to lora_scales.yaml.

    Returns:
        Effective float scale. If override is specified, returns override.
        Otherwise looks up scene in lora_scales.yaml. If scene is unknown, returns 0.0.
    """
    if override is not None:
        return float(override)

    scales = load_lora_scales(scales_path)
    return float(scales.get(scene, 0.0))


def _find_lora_themes_file(path: Union[str, Path] = "configs/lora_themes.yaml") -> Path:
    """Locate lora_themes.yaml configuration file."""
    p = Path(path)
    if p.exists():
        return p
    repo_p = Path(__file__).resolve().parents[3] / "configs" / "lora_themes.yaml"
    if repo_p.exists():
        return repo_p
    return p


def load_theme_triggers(path: Union[str, Path] = "configs/lora_themes.yaml") -> Dict[str, str]:
    """Load theme triggers from configs/lora_themes.yaml."""
    resolved_path = _find_lora_themes_file(path)
    if not resolved_path.exists():
        return {}
    with open(resolved_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    triggers = {}
    for theme_name, theme_info in data.items():
        if isinstance(theme_info, dict) and "trigger" in theme_info:
            triggers[theme_name] = theme_info["trigger"]
    return triggers


def apply_trigger(
    scene: str,
    prompt: str,
    lora_used: bool = True,
    scale: Optional[float] = None,
) -> str:
    """Prepend the theme trigger to the scene prompt when LoRA is used and scale > 0.

    Loads the trigger per theme from configs/lora_themes.yaml.
    The theme names equal the scene names diwali, raksha_bandhan, holi,
    wedding, and indian_home. 'minimal' and 'white' get no trigger.
    Trigger is added only when the effective scale for that scene is > 0.

    Args:
        scene: Scene name (e.g. 'diwali', 'minimal', 'white').
        prompt: Scene prompt string.
        lora_used: Boolean indicating if LoRA is used in generation.
        scale: Optional explicit scale. If None, effective_lora_scale(scene) is used.

    Returns:
        Prompt with trigger prepended if applicable, otherwise original prompt.
    """
    if not lora_used:
        return prompt

    if scene in ("minimal", "white"):
        return prompt

    eff_scale = effective_lora_scale(scene) if scale is None else float(scale)
    if eff_scale <= 0.0:
        return prompt

    triggers = load_theme_triggers()
    trigger = triggers.get(scene)
    if not trigger:
        return prompt

    if not prompt:
        return trigger

    return f"{trigger}, {prompt}"

