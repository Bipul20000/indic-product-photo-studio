from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Optional, Union

import cv2
import numpy as np

from studio.coach.coach import CoachReport, assess_with_segmentation
from studio.coach.config import CoachConfig
from studio.coach.io import load_image
from studio.compose.compose import _resolve_preset, composite, place_product
from studio.generate.backgrounds import BackgroundGenerator, FakeBackgroundGenerator
from studio.generate.config import GenConfig, load_scenes
from studio.generate.prep import build_inpaint_inputs, gen_size
from studio.segment.segmenter import SegmentResult
from studio.vision.people import contains_person


@dataclass
class PipelineResult:
    """Consolidated result of the studio image pipeline execution."""

    ok: bool
    report: CoachReport
    image: Optional[np.ndarray]
    segment: SegmentResult
    seconds: float
    scene: str = "white"
    seed: int = 0
    prompt: str = ""
    background: Optional[np.ndarray] = None
    attempts: int = 1
    person_flag: bool = False


def run_pipeline(
    image_path: Union[str, Path, np.ndarray],
    scene: str = "white",
    marketplace: str = "square_1000",
    force: bool = False,
    config: Optional[CoachConfig] = None,
    seed: int = 0,
    product_name: Optional[str] = None,
    generator: Optional[Any] = None,
    gen_config: Optional[GenConfig] = None,
    fill: float = 0.80,
    controlnet_scale: Optional[float] = None,
) -> PipelineResult:
    """Run the photo studio generation pipeline for white background or AI scenes.

    Args:
        image_path: Path to input photo or BGR NumPy array.
        scene: Scene type ('white', 'diwali', 'raksha_bandhan', 'holi', 'wedding', 'indian_home', 'minimal').
        marketplace: Preset canvas name or (width, height) tuple.
        force: If True, generate the composited image even if photo fails coach checks.
        config: Optional CoachConfig configuration thresholds.
        seed: Random seed for generative background.
        product_name: Name of product to substitute into prompt (default "the product").
        generator: Optional BackgroundGenerator or FakeBackgroundGenerator instance.
        gen_config: Optional GenConfig for generation settings.
        fill: Fraction of shorter canvas side that product occupies (default 0.80).
        controlnet_scale: Optional override for ControlNet conditioning scale.

    Returns:
        PipelineResult containing ok status, coach report, output image, segment result, seconds,
        scene, seed, prompt, background array, attempts, and person_flag.
    """
    start_time = time.time()

    if isinstance(image_path, (str, Path)):
        img_bgr = load_image(image_path)
    else:
        img_bgr = image_path

    report, seg = assess_with_segmentation(img_bgr, config=config)

    if not report.ok and not force:
        elapsed = time.time() - start_time
        return PipelineResult(
            ok=False,
            report=report,
            image=None,
            segment=seg,
            seconds=round(elapsed, 3),
            scene=scene,
            seed=seed,
            prompt="",
            background=None,
            attempts=1,
            person_flag=False,
        )

    canvas_w, canvas_h = _resolve_preset(marketplace)

    # If segmentation completely failed, return blank canvas
    if seg.failed:
        blank = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
        elapsed = time.time() - start_time
        return PipelineResult(
            ok=False,
            report=report,
            image=blank,
            segment=seg,
            seconds=round(elapsed, 3),
            scene=scene,
            seed=seed,
            prompt="",
            background=blank,
            attempts=1,
            person_flag=False,
        )

    # Place product on preset canvas with configured fill
    canvas_rgba, canvas_mask = place_product(seg.rgba, preset=marketplace, fill=fill)

    if scene == "white":
        bg = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
        out_image = composite(bg, canvas_rgba, canvas_mask, shadow=True, opacity=0.30)
        elapsed = time.time() - start_time
        return PipelineResult(
            ok=report.ok,
            report=report,
            image=out_image,
            segment=seg,
            seconds=round(elapsed, 3),
            scene="white",
            seed=seed,
            prompt="",
            background=bg,
            attempts=1,
            person_flag=False,
        )

    # AI Scene Background Generation
    g_cfg = gen_config or GenConfig()
    inpaint_w, inpaint_h = gen_size(marketplace)

    # Prepare inputs for ControlNet inpainting
    init_image, inpaint_mask, small_prod_mask = build_inpaint_inputs(
        canvas_rgba[:, :, :3],
        canvas_mask,
        size=(inpaint_w, inpaint_h),
        erode_px=g_cfg.erode_px,
    )

    # Format scene prompt
    scenes_data = load_scenes()
    scenes_dict = scenes_data.get("scenes", {})
    if scene not in scenes_dict:
        raise ValueError(f"Unknown scene '{scene}'. Available scenes: {list(scenes_dict.keys())}")

    prod_str = product_name.strip() if product_name and product_name.strip() else "the product"
    raw_prompt = scenes_dict[scene]["prompt"]
    prompt = raw_prompt.replace("{product}", prod_str)
    negative = scenes_data.get("negative", "")

    # Lazy-load generator if not provided
    if generator is None:
        generator = BackgroundGenerator(config=g_cfg)

    # Generate background with retry loop on people (up to 3 retries, 1-4 attempts)
    attempts = 1
    person_flag = False
    current_seed = seed
    gen_pil = None

    while True:
        gen_kwargs = {
            "init_image": init_image,
            "inpaint_mask": inpaint_mask,
            "prompt": prompt,
            "negative": negative,
            "seed": current_seed,
        }
        if controlnet_scale is not None:
            gen_kwargs["controlnet_scale"] = controlnet_scale

        gen_pil = generator.generate(**gen_kwargs)

        # Check for people in generated background, ignoring product mask
        has_person = contains_person(
            gen_pil,
            conf=0.35,
            min_box_area=0.01,
            ignore_mask=small_prod_mask,
        )

        if has_person and attempts <= 3:
            current_seed = seed + 1000 * attempts
            attempts += 1
            continue

        person_flag = bool(has_person)
        break

    # Convert PIL RGB to OpenCV BGR
    gen_bgr_small = cv2.cvtColor(np.array(gen_pil), cv2.COLOR_RGB2BGR)

    # Upscale generated background to preset size using bicubic interpolation
    bg_upscaled = cv2.resize(gen_bgr_small, (canvas_w, canvas_h), interpolation=cv2.INTER_CUBIC)

    # Composite original product pixels over generated background with soft shadow at opacity 0.25
    out_image = composite(
        background_bgr=bg_upscaled,
        rgba_on_canvas=canvas_rgba,
        mask=canvas_mask,
        shadow=True,
        opacity=0.25,
    )

    elapsed = time.time() - start_time
    return PipelineResult(
        ok=report.ok,
        report=report,
        image=out_image,
        segment=seg,
        seconds=round(elapsed, 3),
        scene=scene,
        seed=current_seed,
        prompt=prompt,
        background=bg_upscaled,
        attempts=attempts,
        person_flag=person_flag,
    )
