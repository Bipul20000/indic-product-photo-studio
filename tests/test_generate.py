import subprocess
import sys
import numpy as np
from PIL import Image
import pytest

from studio.generate.backgrounds import FakeBackgroundGenerator
from studio.generate.config import GenConfig, REQUIRED_SCENES, load_scenes
from studio.generate.prep import build_inpaint_inputs, gen_size, make_inpaint_condition
from studio.pipeline import run_pipeline


def test_scenes_yaml_loads():
    """scenes.yaml loads with all six required scenes and a non-empty negative prompt."""
    scenes_data = load_scenes("configs/scenes.yaml")
    assert isinstance(scenes_data, dict)

    negative = scenes_data.get("negative")
    assert isinstance(negative, str) and len(negative.strip()) > 0

    scenes = scenes_data.get("scenes")
    assert isinstance(scenes, dict)

    for name in REQUIRED_SCENES:
        assert name in scenes, f"Scene '{name}' missing"
        prompt = scenes[name].get("prompt")
        assert isinstance(prompt, str) and len(prompt.strip()) > 0
        assert "{product}" in prompt, f"Scene '{name}' prompt missing '{{product}}' placeholder"


def test_gen_size_presets():
    """gen_size maps presets correctly to 512-based inpaint dimensions."""
    assert gen_size("square_1000") == (512, 512)
    assert gen_size("square_2000") == (512, 512)
    assert gen_size("portrait_4x5") == (512, 640)
    assert gen_size((1080, 1350)) == (512, 640)
    assert gen_size((1000, 1000)) == (512, 512)


def test_build_inpaint_inputs():
    """build_inpaint_inputs erodes product mask and sets background to 255."""
    # 1000x1000 canvas with a 400x400 center product square
    canvas_bgr = np.full((1000, 1000, 3), 200, dtype=np.uint8)
    product_mask = np.zeros((1000, 1000), dtype=np.uint8)
    product_mask[300:700, 300:700] = 255

    init_img, inpaint_mask, small_prod_mask = build_inpaint_inputs(
        canvas_bgr=canvas_bgr,
        product_mask=product_mask,
        size=(512, 512),
        erode_px=6,
    )

    assert init_img.size == (512, 512)
    assert inpaint_mask.size == (512, 512)
    assert small_prod_mask.shape == (512, 512)

    inpaint_arr = np.array(inpaint_mask)

    # Background (e.g. top-left corner) is 255 (to be generated)
    assert inpaint_arr[10, 10] == 255

    # Center interior of product is 0 (keep)
    assert inpaint_arr[256, 256] == 0
    assert small_prod_mask[256, 256] == 255

    # Eroded product boundary should have turned from 0 to 255 in inpaint mask
    # Rescaled product mask starts at ~153.6. Near row 156 was inside original product,
    # but after 6px erosion it should be part of inpaint area (255)
    assert inpaint_arr[156, 256] == 255


def test_make_inpaint_condition():
    """make_inpaint_condition sets masked pixels to -1.0 and others to [0, 1]."""
    init_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    inpaint_mask = Image.new("L", (64, 64), color=0)

    # Set top half of inpaint mask to 255 (masked for inpainting)
    for y in range(32):
        for x in range(64):
            inpaint_mask.putpixel((x, y), 255)

    condition = make_inpaint_condition(init_img, inpaint_mask)

    assert isinstance(condition, np.ndarray)
    assert condition.shape == (1, 3, 64, 64)
    assert condition.dtype == np.float32

    # Top half (masked) must be exactly -1.0
    masked_region = condition[0, :, :32, :]
    assert np.all(masked_region == -1.0)

    # Bottom half (unmasked) must be scaled to [0, 1]
    unmasked_region = condition[0, :, 32:, :]
    assert np.all(unmasked_region >= 0.0)
    assert np.all(unmasked_region <= 1.0)
    assert np.allclose(unmasked_region, 128.0 / 255.0, atol=1e-2)


def test_run_pipeline_fake_generator():
    """run_pipeline generates scene background with FakeBackgroundGenerator and preserves product."""
    # Synthetic dark object on white background
    img = np.full((800, 800, 3), 255, dtype=np.uint8)
    img[250:550, 250:550] = [40, 60, 180]  # object pixels in BGR

    fake_gen = FakeBackgroundGenerator()
    res = run_pipeline(
        image_path=img,
        scene="diwali",
        marketplace="square_1000",
        force=True,
        seed=42,
        product_name="handcrafted wooden box",
        generator=fake_gen,
    )

    assert res.image is not None
    assert res.image.shape == (1000, 1000, 3)
    assert res.scene == "diwali"
    assert res.seed == 42
    assert "handcrafted wooden box" in res.prompt
    assert res.background is not None
    assert res.background.shape == (1000, 1000, 3)

    # Pixels in product center should match original product colors closely
    center_color = res.image[500, 500]
    assert np.allclose(center_color, [40, 60, 180], atol=15)


def test_import_without_torch_subprocess():
    """Importing studio.generate and studio.pipeline must never require torch at module level."""
    code = """
import sys
# Explicitly block torch and diffusers imports
sys.modules['torch'] = None
sys.modules['diffusers'] = None

import studio
import studio.generate
import studio.pipeline

print('IMPORT_SUCCESS')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Import failed with stderr:\n{result.stderr}"
    assert "IMPORT_SUCCESS" in result.stdout


def test_product_family_rule_and_diverse_pick():
    """Verify family extraction regex and diverse even spacing selection."""
    from scripts.run_scenes import extract_family, pick_diverse_products

    # 1. Family rule
    assert extract_family("Ceramic_Mug_orig.jpg") == "ceramic_mug"
    assert extract_family("Ceramic_Mug_2_orig.jpg") == "ceramic_mug"
    assert extract_family("Ceramic_Mug_3_orig.jpg") == "ceramic_mug"
    assert extract_family("Diwali_diya_49793170_orig.jpg") == "diwali_diya"
    assert extract_family("Diwali_diya_49793178_orig.jpg") == "diwali_diya"
    assert extract_family("Leather_Handbag_1_2_orig.jpg") == "leather_handbag"
    assert extract_family("table_lamp_1970s_02_orig.jpg") == "table_lamp_1970s"

    # 2. Deduplication to first file of each family
    made_up_files = [
        "brass_diya_1_orig.jpg",
        "brass_diya_2_orig.jpg",
        "ceramic_mug_orig.jpg",
        "ceramic_mug_2_orig.jpg",
        "ceramic_mug_3_orig.jpg",
        "leather_bag_10_orig.jpg",
        "leather_bag_20_orig.jpg",
        "rakhi_bracelet_orig.jpg",
        "sneakers_shoes_1_orig.jpg",
        "water_bottle_orig.jpg",
    ]

    seen = set()
    unique_family_files = []
    for f in made_up_files:
        fam = extract_family(f)
        if fam not in seen:
            seen.add(fam)
            unique_family_files.append((f, fam))

    assert len(unique_family_files) == 6
    expected_families = ["brass_diya", "ceramic_mug", "leather_bag", "rakhi_bracelet", "sneakers_shoes", "water_bottle"]
    assert [fam for _, fam in unique_family_files] == expected_families
    assert unique_family_files[0][0] == "brass_diya_1_orig.jpg"
    assert unique_family_files[1][0] == "ceramic_mug_orig.jpg"

    # 3. Even spacing selection
    # From 6 families, pick 3 -> indices linspace(0, 5, 3) = [0.0, 2.5, 5.0].round() = [0, 2, 5]
    picked_3 = pick_diverse_products(unique_family_files, 3)
    assert len(picked_3) == 3
    assert picked_3[0][0] == "brass_diya_1_orig.jpg"
    assert picked_3[1][0] == "leather_bag_10_orig.jpg"
    assert picked_3[2][0] == "water_bottle_orig.jpg"

    # Picking more items than available families returns all without duplicates
    picked_all = pick_diverse_products(unique_family_files, 10)
    assert len(picked_all) == 6


def test_contains_person_mocked_detector():
    """Verify contains_person min_box_area filter and ignore_mask overlap logic."""
    from unittest.mock import MagicMock
    from studio.vision.people import contains_person, detect_people

    # 1000x1000 image (area = 1,000,000)
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)

    def make_mock(box_xyxy, conf_val=0.8):
        box_mock = MagicMock()
        box_mock.cls = [MagicMock(item=lambda: 0)]
        box_mock.conf = [MagicMock(item=lambda: conf_val)]
        box_mock.xyxy = [MagicMock(tolist=lambda: box_xyxy)]

        res_mock = MagicMock()
        res_mock.boxes = [box_mock]

        m = MagicMock()
        m.names = {0: "person"}
        m.return_value = [res_mock]
        return m

    # 1. Box with area < 0.01 (e.g. 50x50 = 2,500 px = 0.25%) -> False
    m_small = make_mock([100, 100, 150, 150])
    assert contains_person(img, conf=0.35, min_box_area=0.01, model=m_small) is False

    # 2. Box with area >= 0.01 (e.g. 200x200 = 40,000 px = 4.0%) -> True
    m_large = make_mock([100, 100, 300, 300])
    assert contains_person(img, conf=0.35, min_box_area=0.01, model=m_large) is True

    # 3. Box with ignore_mask: box is [100, 100, 300, 300] (200x200 = 40,000 px)
    # Mask covers [100:300, 100:300] completely (100% overlap > 50%) -> False
    mask_full = np.zeros((1000, 1000), dtype=np.uint8)
    mask_full[100:300, 100:300] = 255
    assert contains_person(img, conf=0.35, min_box_area=0.01, ignore_mask=mask_full, model=m_large) is False

    # Mask covers only [100:150, 100:200] (50x100 = 5,000 px = 12.5% overlap <= 50%) -> True
    mask_partial = np.zeros((1000, 1000), dtype=np.uint8)
    mask_partial[100:150, 100:200] = 255
    assert contains_person(img, conf=0.35, min_box_area=0.01, ignore_mask=mask_partial, model=m_large) is True


def test_pipeline_retry_loop_on_person(monkeypatch):
    """Verify that generation path retries up to 3 times if person is detected, recording attempts."""
    # 1000x1000 image with a white product square in the center
    fake_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    fake_img[300:700, 300:700] = 255

    generator = FakeBackgroundGenerator()

    # Mock contains_person to return True twice, then False
    call_count = 0

    def mock_contains_person(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return True
        return False

    monkeypatch.setattr("studio.pipeline.contains_person", mock_contains_person)

    result = run_pipeline(
        image_path=fake_img,
        scene="diwali",
        force=True,
        generator=generator,
    )

    assert result.attempts == 3
    assert result.person_flag is False
    assert call_count == 3


def test_sweep_config_grid_has_six_entries():
    """Verify sweep config grid has exactly 6 parameter combinations."""
    from scripts.sweep_gen import get_sweep_configs, SWEEP_CONFIGS

    configs = get_sweep_configs()
    assert len(configs) == 6
    expected = [
        (1.0, 0.80),
        (1.0, 0.65),
        (0.7, 0.80),
        (0.7, 0.65),
        (0.5, 0.80),
        (0.5, 0.65),
    ]
    assert configs == expected
    assert len(SWEEP_CONFIGS) == 6


