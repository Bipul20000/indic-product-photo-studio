"""Tests for Phase 5 LoRA training scripts and support (no GPU, no network)."""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from scripts.prepare_lora_training import prepare_dataset
from scripts.train_lora import build_train_command, has_checkpoint
from scripts.lora_samples import generate_samples, load_theme_configs, make_fake_gradient, render_overview_grid
from studio.generate.config import GenConfig, apply_trigger, load_theme_triggers


# ---------------------------------------------------------------------------
# 1. prepare_lora_training on a small fake data/lora_train in a temp dir
# ---------------------------------------------------------------------------


def test_prepare_lora_training_temp_dir(tmp_path):
    src_dir = tmp_path / "lora_train"
    dst_dir = tmp_path / "lora_train_hf" / "train"

    # Create fake themes
    theme_a = src_dir / "diwali"
    theme_b = src_dir / "holi"
    theme_a.mkdir(parents=True)
    theme_b.mkdir(parents=True)

    # Theme A: 2 images
    (theme_a / "001.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)
    (theme_a / "001.txt").write_text("caption for diwali 1", encoding="utf-8")
    (theme_a / "002.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)
    (theme_a / "002.txt").write_text("caption for diwali 2", encoding="utf-8")

    # Theme B: 1 image
    (theme_b / "001.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)
    (theme_b / "001.txt").write_text("caption for holi 1", encoding="utf-8")

    counts = prepare_dataset(src_dir=src_dir, dst_dir=dst_dir)

    assert counts == {"diwali": 2, "holi": 1}

    # Verify copied files
    assert (dst_dir / "diwali_001.jpg").exists()
    assert (dst_dir / "diwali_002.jpg").exists()
    assert (dst_dir / "holi_001.jpg").exists()

    # Verify metadata.jsonl
    metadata_p = dst_dir / "metadata.jsonl"
    assert metadata_p.exists()

    lines = [json.loads(line) for line in metadata_p.read_text(encoding="utf-8").strip().split("\n")]
    assert len(lines) == 3
    assert lines[0] == {"file_name": "diwali_001.jpg", "text": "caption for diwali 1"}
    assert lines[1] == {"file_name": "diwali_002.jpg", "text": "caption for diwali 2"}
    assert lines[2] == {"file_name": "holi_001.jpg", "text": "caption for holi 1"}


def test_prepare_lora_training_missing_dir(tmp_path):
    missing_dir = tmp_path / "non_existent_lora_train"
    with pytest.raises(FileNotFoundError, match="LoRA training source directory not found"):
        prepare_dataset(src_dir=missing_dir, dst_dir=tmp_path / "out")


# ---------------------------------------------------------------------------
# 2. train command builder & resume logic & dry-run
# ---------------------------------------------------------------------------


def test_build_train_command_flags(tmp_path):
    out_dir = tmp_path / "models" / "festive_lora"
    out_dir.mkdir(parents=True)

    cmd = build_train_command(
        train_data_dir="data/lora_train_hf",
        output_dir=str(out_dir),
        max_steps=1500,
        vendor_script="scripts/vendor/train_text_to_image_lora.py",
        pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5",
    )

    cmd_str = " ".join(cmd)
    assert cmd[0] == "accelerate"
    assert cmd[1] == "launch"
    assert "--mixed_precision fp16" in cmd_str
    assert "scripts/vendor/train_text_to_image_lora.py" in cmd
    assert "--pretrained_model_name_or_path stable-diffusion-v1-5/stable-diffusion-v1-5" in cmd_str
    assert "--train_data_dir data/lora_train_hf" in cmd_str
    assert "--image_column image" in cmd_str
    assert "--caption_column text" in cmd_str
    assert "--resolution 512" in cmd_str
    assert "--random_flip" in cmd
    assert "--train_batch_size 1" in cmd_str
    assert "--gradient_accumulation_steps 4" in cmd_str
    assert "--gradient_checkpointing" in cmd
    assert "--max_train_steps 1500" in cmd_str
    assert "--learning_rate 1e-4" in cmd_str
    assert "--lr_scheduler cosine" in cmd_str
    assert "--lr_warmup_steps 100" in cmd_str
    assert "--rank 16" in cmd_str
    assert "--seed 42" in cmd_str
    assert "--checkpointing_steps 250" in cmd_str
    assert "--checkpoints_total_limit 2" in cmd_str
    assert "--dataloader_num_workers 2" in cmd_str
    assert "--report_to none" in cmd_str
    assert f"--output_dir {out_dir}" in cmd_str
    assert "--resume_from_checkpoint" not in cmd


def test_build_train_command_resume_checkpoint(tmp_path):
    out_dir = tmp_path / "models" / "festive_lora"
    out_dir.mkdir(parents=True)

    # No checkpoint initially
    assert not has_checkpoint(out_dir)
    cmd_no_resume = build_train_command(output_dir=str(out_dir))
    assert "--resume_from_checkpoint" not in cmd_no_resume

    # Create a checkpoint directory
    ckpt_dir = out_dir / "checkpoint-250"
    ckpt_dir.mkdir()
    assert has_checkpoint(out_dir)

    cmd_resume = build_train_command(output_dir=str(out_dir))
    assert "--resume_from_checkpoint" in cmd_resume
    idx = cmd_resume.index("--resume_from_checkpoint")
    assert cmd_resume[idx + 1] == "latest"


def test_train_lora_dry_run(capsys, monkeypatch):
    from scripts.train_lora import main

    monkeypatch.setattr("sys.argv", ["train_lora.py", "--dry-run", "--max-steps", "1200"])
    ret = main()
    assert ret == 0

    captured = capsys.readouterr()
    assert "accelerate launch" in captured.out
    assert "--max_train_steps 1200" in captured.out
    assert "--output_dir models/festive_lora" in captured.out


# ---------------------------------------------------------------------------
# 3. apply_trigger for each theme, for minimal and white, and lora_used=False
# ---------------------------------------------------------------------------


def test_apply_trigger_each_theme():
    triggers = load_theme_triggers()
    expected_triggers = {
        "diwali": "indiandiwali",
        "raksha_bandhan": "indianrakhi",
        "holi": "indianholi",
        "wedding": "indianwedding",
        "indian_home": "indianhome",
    }

    for theme, expected_trig in expected_triggers.items():
        assert triggers.get(theme) == expected_trig
        prompt = "a festive setting with lamps"
        res = apply_trigger(scene=theme, prompt=prompt, lora_used=True, scale=0.8)
        assert res == f"{expected_trig}, {prompt}"


def test_apply_trigger_minimal_and_white():
    prompt = "a product on a clean background"
    # minimal should not get trigger
    res_minimal = apply_trigger(scene="minimal", prompt=prompt, lora_used=True)
    assert res_minimal == prompt

    # white should not get trigger
    res_white = apply_trigger(scene="white", prompt=prompt, lora_used=True)
    assert res_white == prompt


def test_apply_trigger_lora_not_used():
    prompt = "a festive scene with diyas"
    # When lora_used=False, prompt is unaltered regardless of scene
    for scene in ["diwali", "raksha_bandhan", "holi", "wedding", "indian_home", "minimal", "white"]:
        res = apply_trigger(scene=scene, prompt=prompt, lora_used=False)
        assert res == prompt


# ---------------------------------------------------------------------------
# 4. lora_samples with --fake
# ---------------------------------------------------------------------------


def test_lora_samples_fake(tmp_path):
    out_file = tmp_path / "lora_samples_test.jpg"
    result_path = generate_samples(fake=True, out_path=str(out_file))

    assert result_path is not None
    assert out_file.exists()
    assert out_file.stat().st_size > 0

    # Read back image and verify dimensions
    img = cv2.imread(str(out_file))
    assert img is not None
    assert len(img.shape) == 3
    # Height: 44 (header) + 5 * 192 + 4 * 4 + 16 = 1036
    # Width: 140 (label) + 8 * 192 + 6 * 4 + 12 + 16 = 1728
    assert img.shape[0] == 1036
    assert img.shape[1] == 1728
