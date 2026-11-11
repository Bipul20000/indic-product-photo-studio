"""Plain functions and data structures for the Gradio photo studio prototype.

Contains check_photo, generate_one, and generate_all functions that return structured
data (not Gradio components) for easy testability and headless execution.
"""

from collections import namedtuple
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from studio.coach.coach import assess_with_segmentation
from studio.coach.config import CoachConfig
from studio.coach.io import load_image
from studio.compose.compose import PRESETS, _resolve_preset
from studio.generate.backgrounds import FakeBackgroundGenerator
from studio.generate.config import REQUIRED_SCENES, GenConfig
from studio.pipeline import run_pipeline


class CheckRow(dict):
    """Row representing a quality check with dictionary, attribute, and index access."""

    def __init__(self, check: str, passed: bool, value: Any):
        super().__init__(check=check, passed=passed, value=value)
        self.check = check
        self.passed = passed
        self.value = value

    def __getitem__(self, item: Union[str, int]) -> Any:
        if isinstance(item, int):
            return [self.check, self.passed, self.value][item]
        return super().__getitem__(item)


class CheckPhotoResult(tuple):
    """Result from checking photo with Shoot Coach and segmentation."""

    def __new__(cls, verdict: str, tips: str, table: List[CheckRow], mask_overlay: Optional[np.ndarray]):
        return super().__new__(cls, (verdict, tips, table, mask_overlay))

    def __init__(self, verdict: str, tips: str, table: List[CheckRow], mask_overlay: Optional[np.ndarray]):
        self.verdict = verdict
        self.tips = tips
        self.table = table
        self.mask_overlay = mask_overlay

    def __getitem__(self, key: Union[str, int]) -> Any:
        if isinstance(key, str):
            mapping = {
                "verdict": self.verdict,
                "tips": self.tips,
                "table": self.table,
                "mask_overlay": self.mask_overlay,
            }
            if key in mapping:
                return mapping[key]
        return super().__getitem__(key)


class GenerateOneResult(tuple):
    """Result from generating a single scene background."""

    def __new__(cls, image: Optional[np.ndarray], verdict: str, tips: str, seconds: float):
        return super().__new__(cls, (image, verdict, tips, seconds))

    def __init__(self, image: Optional[np.ndarray], verdict: str, tips: str, seconds: float):
        self.image = image
        self.verdict = verdict
        self.tips = tips
        self.seconds = seconds

    def __getitem__(self, key: Union[str, int]) -> Any:
        if isinstance(key, str):
            mapping = {
                "image": self.image,
                "verdict": self.verdict,
                "tips": self.tips,
                "seconds": self.seconds,
            }
            if key in mapping:
                return mapping[key]
        return super().__getitem__(key)


class GenerateAllResult(tuple):
    """Result from generating all scene backgrounds."""

    def __new__(cls, gallery: List[Tuple[np.ndarray, str]], verdict: str, tips: str, seconds: float):
        return super().__new__(cls, (gallery, verdict, tips, seconds))

    def __init__(self, gallery: List[Tuple[np.ndarray, str]], verdict: str, tips: str, seconds: float):
        self.gallery = gallery
        self.verdict = verdict
        self.tips = tips
        self.seconds = seconds

    def __getitem__(self, key: Union[str, int]) -> Any:
        if isinstance(key, str):
            mapping = {
                "gallery": self.gallery,
                "verdict": self.verdict,
                "tips": self.tips,
                "seconds": self.seconds,
            }
            if key in mapping:
                return mapping[key]
        return super().__getitem__(key)


def check_photo(
    image_path: Union[str, Path, np.ndarray, None],
    config: Optional[CoachConfig] = None,
) -> CheckPhotoResult:
    """Run Shoot Coach checks and segmentation on a photo.

    Args:
        image_path: Path to the image file or BGR numpy array.
        config: Optional CoachConfig for check thresholds.

    Returns:
        CheckPhotoResult containing:
            verdict: 'OK' or 'RETAKE' (or 'ERROR' on failure)
            tips: Plain text tips from Coach
            table: List of CheckRow(check, passed, value)
            mask_overlay: RGB numpy array with product mask overlay in red (40% alpha)
    """
    try:
        if image_path is None:
            return CheckPhotoResult("ERROR", "Please upload an image first.", [], None)

        if isinstance(image_path, (str, Path)):
            p = Path(image_path)
            if not p.exists() or not p.is_file():
                return CheckPhotoResult("ERROR", f"Image file not found: {image_path}", [], None)
            img_bgr = load_image(image_path)
        elif isinstance(image_path, np.ndarray):
            img_bgr = image_path
        else:
            return CheckPhotoResult("ERROR", f"Invalid image input type: {type(image_path)}", [], None)

        if img_bgr is None or img_bgr.size == 0:
            return CheckPhotoResult("ERROR", "Failed to decode image.", [], None)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        report, seg = assess_with_segmentation(img_bgr, config=config)

        # Build table
        table = []
        for r in report.results:
            val = round(r.value, 2) if isinstance(r.value, float) else r.value
            table.append(CheckRow(check=r.name, passed=r.passed, value=val))

        # Build mask overlay (red, 40% alpha)
        mask_overlay = img_rgb.copy()
        if seg is not None and not seg.failed and seg.mask is not None:
            # Mask is 2D uint8 (0 or 255)
            mask_bool = (seg.mask > 127)
            red_color = np.array([255, 0, 0], dtype=np.uint8)
            mask_overlay[mask_bool] = (0.6 * img_rgb[mask_bool] + 0.4 * red_color).astype(np.uint8)

        verdict = "OK" if report.ok else "RETAKE"
        tips = "\n".join(report.tips) if report.tips else ""

        return CheckPhotoResult(verdict=verdict, tips=tips, table=table, mask_overlay=mask_overlay)
    except Exception as exc:
        return CheckPhotoResult("ERROR", f"Error checking photo: {exc}", [], None)


def generate_one(
    image_path: Union[str, Path, np.ndarray, None],
    scene: str = "diwali",
    marketplace: str = "square_1000",
    product_name: Optional[str] = None,
    seed: int = 0,
    force_retake: bool = False,
    generator: Optional[Any] = None,
    gen_config: Optional[GenConfig] = None,
    config: Optional[CoachConfig] = None,
) -> GenerateOneResult:
    """Run generation pipeline for a single scene with Shoot Coach gating.

    Args:
        image_path: Path to the image file or BGR numpy array.
        scene: Scene identifier (white, diwali, raksha_bandhan, holi, wedding, indian_home, minimal).
        marketplace: Preset name from compose.PRESETS (e.g. square_1000).
        product_name: Optional product name for prompt synthesis.
        seed: Random seed for generation.
        force_retake: If True, generate even if Shoot Coach says retake.
        generator: BackgroundGenerator or FakeBackgroundGenerator instance.
        gen_config: Optional GenConfig settings.
        config: Optional CoachConfig settings.

    Returns:
        GenerateOneResult containing:
            image: RGB numpy array if generation succeeded, None if retake requested.
            verdict: 'OK' or 'RETAKE'
            tips: Coach tips if retake
            seconds: Processing time in seconds
    """
    try:
        if image_path is None:
            return GenerateOneResult(None, "ERROR", "Please upload an image first.", 0.0)

        if isinstance(image_path, (str, Path)):
            p = Path(image_path)
            if not p.exists() or not p.is_file():
                return GenerateOneResult(None, "ERROR", f"Image file not found: {image_path}", 0.0)

        if generator is None:
            generator = FakeBackgroundGenerator(config=gen_config)

        res = run_pipeline(
            image_path=image_path,
            scene=scene,
            marketplace=marketplace,
            force=force_retake,
            product_name=product_name if product_name else None,
            seed=int(seed),
            generator=generator,
            gen_config=gen_config,
            config=config,
        )

        verdict = "OK" if (res.report and res.report.ok) else "RETAKE"
        tips = "\n".join(res.report.tips) if (res.report and res.report.tips) else ""

        if res.image is None:
            return GenerateOneResult(None, verdict, tips, res.seconds)

        img_rgb = cv2.cvtColor(res.image, cv2.COLOR_BGR2RGB)
        return GenerateOneResult(img_rgb, verdict, tips, res.seconds)
    except Exception as exc:
        return GenerateOneResult(None, "ERROR", f"Error generating image: {exc}", 0.0)


def generate_all(
    image_path: Union[str, Path, np.ndarray, None],
    marketplace: str = "square_1000",
    product_name: Optional[str] = None,
    seed: int = 0,
    force_retake: bool = False,
    generator: Optional[Any] = None,
    gen_config: Optional[GenConfig] = None,
    config: Optional[CoachConfig] = None,
    scenes: Optional[List[str]] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> GenerateAllResult:
    """Generate all generative scenes for the uploaded photo with the same seed.

    Args:
        image_path: Path to the image file or BGR numpy array.
        marketplace: Preset name from compose.PRESETS.
        product_name: Optional product name.
        seed: Random seed for generation.
        force_retake: If True, generate even if Shoot Coach says retake.
        generator: Generator instance.
        gen_config: Optional GenConfig.
        config: Optional CoachConfig.
        scenes: List of scene names (defaults to REQUIRED_SCENES).
        progress: Optional callable progress(fraction, desc=...) from gr.Progress().

    Returns:
        GenerateAllResult containing:
            gallery: List of (RGB numpy array, scene_name) tuples.
            verdict: 'OK' or 'RETAKE'
            tips: Coach tips if retake
            seconds: Total seconds taken across scenes
    """
    try:
        if image_path is None:
            return GenerateAllResult([], "ERROR", "Please upload an image first.", 0.0)

        if isinstance(image_path, (str, Path)):
            p = Path(image_path)
            if not p.exists() or not p.is_file():
                return GenerateAllResult([], "ERROR", f"Image file not found: {image_path}", 0.0)

        if scenes is None:
            scenes = list(REQUIRED_SCENES)

        if generator is None:
            generator = FakeBackgroundGenerator(config=gen_config)

        total_seconds = 0.0
        gallery: List[Tuple[np.ndarray, str]] = []
        last_report = None
        total_scenes = len(scenes)

        for idx, sc in enumerate(scenes):
            if progress is not None:
                try:
                    progress(idx / total_scenes, desc=f"Generating scene {idx+1}/{total_scenes}: {sc}")
                except Exception:
                    pass

            res = run_pipeline(
                image_path=image_path,
                scene=sc,
                marketplace=marketplace,
                force=force_retake,
                product_name=product_name if product_name else None,
                seed=int(seed),
                generator=generator,
                gen_config=gen_config,
                config=config,
            )
            total_seconds += res.seconds
            last_report = res.report

            # If coach rejected and force_retake is False, stop immediately
            if res.image is None:
                verdict = "RETAKE"
                tips = "\n".join(res.report.tips) if (res.report and res.report.tips) else ""
                return GenerateAllResult([], verdict, tips, round(total_seconds, 3))

            img_rgb = cv2.cvtColor(res.image, cv2.COLOR_BGR2RGB)
            gallery.append((img_rgb, sc))

        if progress is not None:
            try:
                progress(1.0, desc="All scenes generated!")
            except Exception:
                pass

        verdict = "OK" if (last_report and last_report.ok) else "RETAKE"
        tips = "\n".join(last_report.tips) if (last_report and last_report.tips) else ""
        return GenerateAllResult(gallery, verdict, tips, round(total_seconds, 3))
    except Exception as exc:
        return GenerateAllResult([], "ERROR", f"Error generating scenes: {exc}", 0.0)
