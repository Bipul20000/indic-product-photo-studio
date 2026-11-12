#!/usr/bin/env python3
"""Generate sample text-to-image outputs comparing base SD 1.5 vs festive LoRA.

Renders a 5-theme grid (diwali, raksha_bandhan, holi, wedding, indian_home) with
4 base model images and 4 LoRA images per theme.
Supports --fake flag for offline layout verification on CPU without models.
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from studio.generate.config import GenConfig

THEMES = ["diwali", "raksha_bandhan", "holi", "wedding", "indian_home"]

THEME_PALETTES = {
    "diwali": ((255, 145, 30), (60, 20, 10)),
    "raksha_bandhan": ((255, 200, 210), (140, 30, 50)),
    "holi": ((140, 220, 255), (255, 150, 210)),
    "wedding": ((240, 205, 120), (150, 20, 30)),
    "indian_home": ((235, 215, 190), (110, 75, 45)),
}


def load_theme_configs(yaml_path: Union[str, Path] = "configs/lora_themes.yaml") -> Dict[str, Dict]:
    """Load trigger and description for each theme from configs/lora_themes.yaml."""
    p = Path(yaml_path)
    if not p.exists():
        repo_p = Path(__file__).resolve().parents[1] / "configs" / "lora_themes.yaml"
        if repo_p.exists():
            p = repo_p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def make_fake_gradient(theme: str, seed: int, is_lora: bool, size: int = 192) -> np.ndarray:
    """Generate deterministic synthetic gradient for layout testing."""
    c1, c2 = THEME_PALETTES.get(theme, ((200, 200, 200), (50, 50, 50)))
    if is_lora:
        # Boost saturation / warmth for LoRA fake representation
        c1 = (min(255, int(c1[0] * 1.1)), min(255, int(c1[1] * 0.95)), min(255, int(c1[2] * 1.05)))
        c2 = (min(255, int(c2[0] * 1.2)), min(255, int(c2[1] * 0.9)), min(255, int(c2[2] * 0.9)))

    s_shift = (seed * 23) % 40 - 20
    c1 = tuple(max(0, min(255, val + s_shift)) for val in c1)
    c2 = tuple(max(0, min(255, val - s_shift)) for val in c2)

    y = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None, None]
    top = np.array(c1, dtype=np.float32)
    bottom = np.array(c2, dtype=np.float32)
    grad = (1.0 - y) * top + y * bottom
    img_bgr = cv2.cvtColor(np.clip(grad, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Label seed and type
    label = f"LoRA s={seed}" if is_lora else f"base s={seed}"
    cv2.putText(img_bgr, label, (8, size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img_bgr


def render_overview_grid(
    theme_images: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]],
    thumb_size: int = 192,
) -> np.ndarray:
    """Assemble a side-by-side grid of 5 themes x (4 base + 4 LoRA) thumbnails.

    Args:
        theme_images: Mapping from theme name to tuple of (4 base images, 4 LoRA images).
        thumb_size: Pixel size for each square thumbnail (default 192).

    Returns:
        BGR image of the complete contact sheet.
    """
    n_themes = len(THEMES)
    header_h = 44
    label_w = 140
    margin = 8
    col_gap = 4
    mid_sep = 12

    # 4 base columns, separator, 4 lora columns
    content_w = 4 * thumb_size + 3 * col_gap + mid_sep + 4 * thumb_size + 3 * col_gap
    total_w = label_w + content_w + margin * 2
    total_h = header_h + n_themes * thumb_size + (n_themes - 1) * col_gap + margin * 2

    canvas = np.full((total_h, total_w, 3), 245, dtype=np.uint8)

    # 1. Draw header background and titles
    base_start_x = label_w + margin
    base_end_x = base_start_x + 4 * thumb_size + 3 * col_gap
    lora_start_x = base_end_x + mid_sep
    lora_end_x = lora_start_x + 4 * thumb_size + 3 * col_gap

    # Base header
    cv2.rectangle(canvas, (base_start_x, margin), (base_end_x, margin + header_h - 6), (230, 230, 230), -1)
    cv2.putText(
        canvas,
        "base (SD 1.5)",
        (base_start_x + (base_end_x - base_start_x) // 2 - 50, margin + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )

    # LoRA header
    cv2.rectangle(canvas, (lora_start_x, margin), (lora_end_x, margin + header_h - 6), (220, 235, 255), -1)
    cv2.putText(
        canvas,
        "LoRA (festive_lora)",
        (lora_start_x + (lora_end_x - lora_start_x) // 2 - 75, margin + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (160, 50, 20),
        2,
        cv2.LINE_AA,
    )

    # 2. Place rows
    for r_idx, theme in enumerate(THEMES):
        y_top = margin + header_h + r_idx * (thumb_size + col_gap)

        # Row label
        theme_display = theme.replace("_", " ").title()
        cv2.putText(
            canvas,
            theme_display,
            (margin + 4, y_top + thumb_size // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

        base_imgs, lora_imgs = theme_images.get(theme, ([], []))

        # Base images
        for c_idx in range(4):
            x_left = base_start_x + c_idx * (thumb_size + col_gap)
            if c_idx < len(base_imgs):
                img = cv2.resize(base_imgs[c_idx], (thumb_size, thumb_size))
                canvas[y_top : y_top + thumb_size, x_left : x_left + thumb_size] = img
            cv2.rectangle(canvas, (x_left, y_top), (x_left + thumb_size - 1, y_top + thumb_size - 1), (180, 180, 180), 1)

        # LoRA images
        for c_idx in range(4):
            x_left = lora_start_x + c_idx * (thumb_size + col_gap)
            if c_idx < len(lora_imgs):
                img = cv2.resize(lora_imgs[c_idx], (thumb_size, thumb_size))
                canvas[y_top : y_top + thumb_size, x_left : x_left + thumb_size] = img
            cv2.rectangle(canvas, (x_left, y_top), (x_left + thumb_size - 1, y_top + thumb_size - 1), (180, 180, 180), 1)

    return canvas


def generate_samples(
    fake: bool = False,
    lora_path: str = "models/festive_lora",
    lora_scale: float = 0.8,
    out_path: str = "outputs/lora_samples.jpg",
) -> Optional[Path]:
    """Run sample generation for all 5 themes comparing base and LoRA."""
    theme_cfg = load_theme_configs()
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    theme_images: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {}

    if fake:
        print("Generating fake sample gradients for layout testing...")
        for theme in THEMES:
            base_list = [make_fake_gradient(theme, seed, is_lora=False) for seed in range(4)]
            lora_list = [make_fake_gradient(theme, seed, is_lora=True) for seed in range(4)]
            theme_images[theme] = (base_list, lora_list)
    else:
        # Check if LoRA weights exist
        lora_p = Path(lora_path)
        actual_weights = None
        if lora_p.is_file():
            actual_weights = lora_p
        elif (lora_p / "pytorch_lora_weights.safetensors").is_file():
            actual_weights = lora_p / "pytorch_lora_weights.safetensors"

        if not actual_weights or not actual_weights.exists():
            print(f"LoRA weights not found at: {lora_path}. Skipping samples generation.")
            return None

        import torch
        if not torch.cuda.is_available():
            print("CUDA is not available. LoRA sampling requires a GPU. Skipping samples generation.")
            return None

        from diffusers import DDIMScheduler, StableDiffusionPipeline

        base_model_id = GenConfig().base_models[0]
        print(f"Loading base pipeline: {base_model_id}...")
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")

        # Generate base images
        for theme in THEMES:
            t_data = theme_cfg.get(theme, {})
            trigger = t_data.get("trigger", "")
            desc = t_data.get("description", "")
            prompt = f"{trigger}, {desc}" if trigger else desc

            print(f"Generating base images for theme '{theme}' (prompt: {prompt})...")
            b_list = []
            for seed in range(4):
                gen = torch.Generator("cuda").manual_seed(seed)
                out = pipe(prompt=prompt, num_inference_steps=20, guidance_scale=7.0, generator=gen)
                bgr = cv2.cvtColor(np.array(out.images[0]), cv2.COLOR_RGB2BGR)
                b_list.append(bgr)

            theme_images[theme] = (b_list, [])

        # Load LoRA weights
        print(f"Loading LoRA weights from {actual_weights}...")
        pipe.load_lora_weights(str(lora_p if lora_p.is_dir() else lora_p.parent))

        # Generate LoRA images
        for theme in THEMES:
            t_data = theme_cfg.get(theme, {})
            trigger = t_data.get("trigger", "")
            desc = t_data.get("description", "")
            prompt = f"{trigger}, {desc}" if trigger else desc

            print(f"Generating LoRA images for theme '{theme}'...")
            l_list = []
            for seed in range(4):
                gen = torch.Generator("cuda").manual_seed(seed)
                out = pipe(
                    prompt=prompt,
                    num_inference_steps=20,
                    guidance_scale=7.0,
                    generator=gen,
                    cross_attention_kwargs={"scale": lora_scale},
                )
                bgr = cv2.cvtColor(np.array(out.images[0]), cv2.COLOR_RGB2BGR)
                l_list.append(bgr)

            base_list = theme_images[theme][0]
            theme_images[theme] = (base_list, l_list)

    grid = render_overview_grid(theme_images, thumb_size=192)
    cv2.imwrite(str(out_p), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Sample contact sheet saved to: {out_p}")
    return out_p


def parse_args():
    parser = argparse.ArgumentParser(description="Generate sample images comparing base vs LoRA.")
    parser.add_argument("--fake", action="store_true", help="Draw fake gradients for local layout testing")
    parser.add_argument("--lora", type=str, default="models/festive_lora", help="Path to LoRA directory or safetensors")
    parser.add_argument("--lora-scale", type=float, default=0.8, help="Cross-attention LoRA scale")
    parser.add_argument("--out", type=str, default="outputs/lora_samples.jpg", help="Output contact sheet image path")
    return parser.parse_args()


def main():
    args = parse_args()
    generate_samples(
        fake=args.fake,
        lora_path=args.lora,
        lora_scale=args.lora_scale,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
