"""Tests for Phase 7a per-scene LoRA scale and paired comparison."""

from pathlib import Path
import subprocess
import sys
import pytest

from scripts.compare_lora import format_metrics_table, parse_args
from studio.generate.config import apply_trigger, effective_lora_scale, load_lora_scales


def test_effective_lora_scale():
    # 1. With the config defaults
    scales = load_lora_scales()
    assert scales["diwali"] == 0.8
    assert scales["raksha_bandhan"] == 0.8
    assert scales["holi"] == 0.0
    assert scales["wedding"] == 0.8
    assert scales["indian_home"] == 0.8

    assert effective_lora_scale("diwali") == 0.8
    assert effective_lora_scale("raksha_bandhan") == 0.8
    assert effective_lora_scale("holi") == 0.0
    assert effective_lora_scale("wedding") == 0.8
    assert effective_lora_scale("indian_home") == 0.8

    # 2. With override
    assert effective_lora_scale("diwali", override=0.5) == 0.5
    assert effective_lora_scale("holi", override=0.5) == 0.5
    assert effective_lora_scale("unknown_scene", override=0.6) == 0.6

    # 3. For unknown scenes (default to 0.0)
    assert effective_lora_scale("unknown_scene") == 0.0
    assert effective_lora_scale("minimal") == 0.0
    assert effective_lora_scale("white") == 0.0


def test_apply_trigger_scales():
    prompt = "a product on a table"

    # Scale > 0: trigger added
    assert apply_trigger("diwali", prompt, lora_used=True, scale=0.8) == f"indiandiwali, {prompt}"
    assert apply_trigger("holi", prompt, lora_used=True, scale=0.7) == f"indianholi, {prompt}"

    # Scale == 0: trigger NOT added
    assert apply_trigger("diwali", prompt, lora_used=True, scale=0.0) == prompt
    # In config, holi scale is 0.0, so default call should not add trigger
    assert apply_trigger("holi", prompt, lora_used=True) == prompt

    # lora_used == False: trigger NOT added even if scale > 0
    assert apply_trigger("diwali", prompt, lora_used=False, scale=0.8) == prompt


def test_compare_lora_args():
    # Default arguments
    args = parse_args([])
    assert args.lora == "models/festive_lora"
    assert args.n_products == 6
    assert args.scenes == "diwali,raksha_bandhan,holi,wedding,indian_home"
    assert args.seeds == "0,1"
    assert args.out == "outputs/compare"
    assert not args.fake_generator

    # Custom arguments
    args_custom = parse_args([
        "--lora", "models/custom_lora",
        "--n-products", "3",
        "--scenes", "diwali,holi",
        "--seeds", "10,20",
        "--out", "outputs/custom_compare",
        "--fake-generator",
    ])
    assert args_custom.lora == "models/custom_lora"
    assert args_custom.n_products == 3
    assert args_custom.scenes == "diwali,holi"
    assert args_custom.seeds == "10,20"
    assert args_custom.out == "outputs/custom_compare"
    assert args_custom.fake_generator is True


def test_format_metrics_table_fake_data():
    fake_metrics = {
        "diwali": {
            "base": {"clip_score": 0.285, "attempts": 1.10, "flagged": 0, "count": 6, "seconds": 2.45},
            "lora": {"clip_score": 0.320, "attempts": 1.15, "flagged": 1, "count": 6, "seconds": 2.61},
        },
        "holi": {
            "base": {"clip_score": 0.250, "attempts": 1.00, "flagged": 0, "count": 6, "seconds": 2.10},
            "lora": {"clip_score": 0.250, "attempts": 1.00, "flagged": 0, "count": 6, "seconds": 2.12},
        },
    }
    sanity_diffs = {"holi": 0.0000}

    table = format_metrics_table(fake_metrics, sanity_diffs)
    assert "Scene" in table
    assert "diwali" in table
    assert "holi" in table
    assert "0.285" in table
    assert "0.320" in table
    assert "0/6" in table
    assert "1/6" in table
    assert "Sanity check (scale 0.0 scenes):" in table
    assert "holi: mean absolute pixel difference = 0.0000" in table
    assert "CLIP score measures text alignment only; it does not measure authenticity." in table


def test_compare_lora_help_from_temp_dir(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "compare_lora.py"

    proc = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Paired Base vs LoRA evaluation script" in proc.stdout
