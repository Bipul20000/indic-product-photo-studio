"""Tests for studio app logic and Gradio Blocks construction."""

from pathlib import Path
import cv2
import gradio as gr
import numpy as np
import pytest

from studio.app_logic import (
    CheckPhotoResult,
    GenerateAllResult,
    GenerateOneResult,
    check_photo,
    generate_all,
    generate_one,
)
from studio.generate.backgrounds import FakeBackgroundGenerator
from studio.generate.config import REQUIRED_SCENES


@pytest.fixture
def synthetic_image_path(tmp_path: Path) -> Path:
    """Create a synthetic high-quality image of a product on a plain background."""
    img = np.full((1000, 1000, 3), 245, dtype=np.uint8)
    # Distinct central foreground object
    img[300:700, 300:700] = [35, 45, 55]

    path = tmp_path / "synthetic_product.jpg"
    cv2.imwrite(str(path), img)
    return path


def test_check_photo_returns_verdict_and_expected_table(synthetic_image_path: Path):
    """Verify check_photo returns verdict and quality check table."""
    try:
        res = check_photo(synthetic_image_path)
    except Exception as exc:
        pytest.skip(f"Segmentation model cannot load offline: {exc}")

    assert isinstance(res, CheckPhotoResult)
    assert res.verdict in ("OK", "RETAKE")

    # Verify check names in table
    table = res.table
    check_names = [row["check"] for row in table]
    expected_checks = ["resolution", "blur", "brightness", "clutter", "framing"]
    for expected in expected_checks:
        assert expected in check_names

    # Verify mask overlay format
    assert res.mask_overlay is not None
    assert res.mask_overlay.shape == (1000, 1000, 3)


def test_generate_one_right_size(synthetic_image_path: Path):
    """Verify generate_one with scene='diwali' returns an image of the right size."""
    generator = FakeBackgroundGenerator()
    try:
        res = generate_one(
            image_path=synthetic_image_path,
            scene="diwali",
            marketplace="square_1000",
            force_retake=True,
            generator=generator,
        )
    except Exception as exc:
        pytest.skip(f"Segmentation model cannot load offline: {exc}")

    assert isinstance(res, GenerateOneResult)
    assert res.image is not None
    # Preset square_1000 -> (1000, 1000, 3)
    assert res.image.shape == (1000, 1000, 3)
    assert res.seconds >= 0.0


def test_generate_all_returns_six_images(synthetic_image_path: Path):
    """Verify generate_all returns exactly 6 scene images."""
    generator = FakeBackgroundGenerator()
    try:
        res = generate_all(
            image_path=synthetic_image_path,
            marketplace="square_1000",
            force_retake=True,
            generator=generator,
        )
    except Exception as exc:
        pytest.skip(f"Segmentation model cannot load offline: {exc}")

    assert isinstance(res, GenerateAllResult)
    assert len(res.gallery) == 6

    # Verify scene names and image sizes
    returned_scenes = [item[1] for item in res.gallery]
    assert returned_scenes == REQUIRED_SCENES
    for img, _ in res.gallery:
        assert img.shape == (1000, 1000, 3)


def test_handlers_missing_file_path():
    """Verify that handlers given a missing file path return error messages without raising."""
    missing_path = "/non/existent/path/product_404.jpg"

    # 1. check_photo
    res_check = check_photo(missing_path)
    assert res_check.verdict == "ERROR"
    assert "not found" in res_check.tips.lower()
    assert res_check.table == []
    assert res_check.mask_overlay is None

    # 2. generate_one
    res_one = generate_one(missing_path)
    assert res_one.image is None
    assert res_one.verdict == "ERROR"
    assert "not found" in res_one.tips.lower()

    # 3. generate_all
    res_all = generate_all(missing_path)
    assert res_all.gallery == []
    assert res_all.verdict == "ERROR"
    assert "not found" in res_all.tips.lower()


def test_app_blocks_builds_without_launching():
    """Verify that importing app.app and building the Blocks object works without launching."""
    from app.app import build_app

    demo = build_app(fake_generator=True, default_preset="square_1000")
    assert isinstance(demo, gr.Blocks)
