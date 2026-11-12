"""Tests for Phase 7b blind evaluation tool and statistical analysis."""

import csv
import json
from pathlib import Path
import threading
import cv2
import gradio as gr
import numpy as np
import pytest

from app.blind_test import append_response, build_app, compute_winner
from scripts.analyze_blind_test import analyze_responses, compute_stats
from scripts.make_blind_pairs import compute_pixel_diff, make_blind_pairs


# ---------------------------------------------------------------------------
# 1. Pairing logic with a fake manifest & directory
# ---------------------------------------------------------------------------


def test_pairing_logic_with_fake_manifest(tmp_path):
    compare_dir = tmp_path / "compare"
    base_dir = compare_dir / "base"
    lora_dir = compare_dir / "lora"

    scenes = ["diwali", "raksha_bandhan", "holi", "wedding", "indian_home"]
    for sc in scenes:
        (base_dir / sc).mkdir(parents=True)
        (lora_dir / sc).mkdir(parents=True)

    manifest_rows = []

    # Helper to create a dummy image with given solid color
    def create_img(path: Path, color_val: int):
        img = np.full((50, 50, 3), color_val, dtype=np.uint8)
        cv2.imwrite(str(path), img)

    # 1. diwali: product A, s0 -> diff = 30 (should be kept)
    create_img(base_dir / "diwali" / "prodA_s0.jpg", 100)
    create_img(lora_dir / "diwali" / "prodA_s0.jpg", 130)
    manifest_rows.append({"product": "prodA", "scene": "diwali", "seed": "0", "variant": "base"})
    manifest_rows.append({"product": "prodA", "scene": "diwali", "seed": "0", "variant": "lora"})

    # 2. diwali: product A, s1 -> diff = 0.5 (should be dropped because < 2.0)
    create_img(base_dir / "diwali" / "prodA_s1.jpg", 100)
    create_img(lora_dir / "diwali" / "prodA_s1.jpg", 101)
    manifest_rows.append({"product": "prodA", "scene": "diwali", "seed": "1", "variant": "base"})
    manifest_rows.append({"product": "prodA", "scene": "diwali", "seed": "1", "variant": "lora"})

    # 3. diwali: product B, s0, s1, s2 -> test at most 2 per product
    for s in [0, 1, 2]:
        create_img(base_dir / "diwali" / f"prodB_s{s}.jpg", 80)
        create_img(lora_dir / "diwali" / f"prodB_s{s}.jpg", 120)
        manifest_rows.append({"product": "prodB", "scene": "diwali", "seed": str(s), "variant": "base"})
        manifest_rows.append({"product": "prodB", "scene": "diwali", "seed": str(s), "variant": "lora"})

    # 4. holi: product C, s0 -> diff = 50 (should be SKIPPED because holi is excluded)
    create_img(base_dir / "holi" / "prodC_s0.jpg", 50)
    create_img(lora_dir / "holi" / "prodC_s0.jpg", 100)
    manifest_rows.append({"product": "prodC", "scene": "holi", "seed": "0", "variant": "base"})
    manifest_rows.append({"product": "prodC", "scene": "holi", "seed": "0", "variant": "lora"})

    # 5. wedding: product D, s0 -> diff = 25 (kept)
    create_img(base_dir / "wedding" / "prodD_s0.jpg", 70)
    create_img(lora_dir / "wedding" / "prodD_s0.jpg", 95)
    manifest_rows.append({"product": "prodD", "scene": "wedding", "seed": "0", "variant": "base"})
    manifest_rows.append({"product": "prodD", "scene": "wedding", "seed": "0", "variant": "lora"})

    # Write manifest.csv
    manifest_p = compare_dir / "manifest.csv"
    with open(manifest_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["product", "scene", "seed", "variant"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    out_json = tmp_path / "blind_pairs.json"
    pairs = make_blind_pairs(
        compare_dir=compare_dir,
        manifest_path=manifest_p,
        out_json=out_json,
        seed=42,
        min_diff=2.0,
        max_per_scene=6,
        max_per_product=2,
    )

    # Assertions
    assert out_json.exists()
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(pairs) == len(loaded)

    scenes_in_pairs = {p["scene"] for p in pairs}
    assert "holi" not in scenes_in_pairs, "Holi must be excluded from blind pairs!"
    assert "diwali" in scenes_in_pairs
    assert "wedding" in scenes_in_pairs

    # Check that low diff pair (prodA_s1 with diff 1) was dropped
    diwali_pairs = [p for p in pairs if p["scene"] == "diwali"]
    assert not any(p["product"] == "prodA" and p["seed"] == 1 for p in diwali_pairs)

    # Check max 2 per product for prodB (which had 3 candidates)
    prodB_pairs = [p for p in diwali_pairs if p["product"] == "prodB"]
    assert len(prodB_pairs) == 2, "Must enforce at most 2 pairs per product!"

    # Check fields in pair object
    first = pairs[0]
    for required_key in ["pair_id", "scene", "product", "seed", "base_path", "lora_path"]:
        assert required_key in first


# ---------------------------------------------------------------------------
# 2. Winner computation from choice and left_variant
# ---------------------------------------------------------------------------


def test_compute_winner_all_combinations():
    # User picks Left
    assert compute_winner("Left", "base") == "base"
    assert compute_winner("left", "lora") == "lora"

    # User picks Right
    assert compute_winner("Right", "base") == "lora"
    assert compute_winner("right", "lora") == "base"

    # User picks No clear difference / Tie
    assert compute_winner("No clear difference", "base") == "tie"
    assert compute_winner("no clear difference", "lora") == "tie"
    assert compute_winner("tie", "base") == "tie"
    assert compute_winner("neither", "lora") == "tie"


# ---------------------------------------------------------------------------
# 3. Binomial and Wilson computation on fake data
# ---------------------------------------------------------------------------


def test_compute_stats_on_fake_data():
    # Case 1: Decisive win for LoRA (15 vs 5 out of 20)
    w_rate, p_val, ci_low, ci_high = compute_stats(lora_wins=15, base_wins=5)
    assert w_rate == 0.75
    assert p_val < 0.05
    assert 0.50 < ci_low < 0.75
    assert 0.75 < ci_high < 1.0

    # Case 2: Tie in decisive counts (5 vs 5 out of 10)
    w_rate, p_val, ci_low, ci_high = compute_stats(lora_wins=5, base_wins=5)
    assert w_rate == 0.50
    assert p_val == 1.0
    assert ci_low < 0.50 < ci_high

    # Case 3: Zero decisive answers (0 vs 0)
    w_rate, p_val, ci_low, ci_high = compute_stats(lora_wins=0, base_wins=0)
    assert w_rate == 0.0
    assert p_val == 1.0
    assert ci_low == 0.0
    assert ci_high == 1.0


def test_analyze_responses_summary(tmp_path):
    fake_responses = [
        {"rater_id": "r1", "scene": "diwali", "winner": "lora"},
        {"rater_id": "r1", "scene": "diwali", "winner": "lora"},
        {"rater_id": "r2", "scene": "diwali", "winner": "base"},
        {"rater_id": "r2", "scene": "wedding", "winner": "lora"},
        {"rater_id": "r3", "scene": "wedding", "winner": "tie"},
    ]

    out_csv = tmp_path / "summary.csv"
    summary = analyze_responses(fake_responses, out_csv=out_csv)

    assert out_csv.exists()
    assert len(summary) >= 3  # diwali, wedding, OVERALL

    diwali_row = next(r for r in summary if r["scene"] == "diwali")
    assert diwali_row["answers"] == 3
    assert diwali_row["lora_wins"] == 2
    assert diwali_row["base_wins"] == 1
    assert diwali_row["win_rate"] == round(2 / 3, 4)

    overall_row = next(r for r in summary if r["scene"] == "OVERALL")
    assert overall_row["answers"] == 5
    assert overall_row["raters"] == 3
    assert overall_row["ties"] == 1


# ---------------------------------------------------------------------------
# 4. CSV append with threading lock in temp dir
# ---------------------------------------------------------------------------


def test_append_response_thread_safe(tmp_path):
    csv_file = tmp_path / "results" / "responses.csv"
    test_lock = threading.Lock()

    def worker(worker_id):
        for i in range(10):
            append_response(
                csv_path=csv_file,
                rater_id=f"rater_{worker_id}",
                pair_id=f"pair_{worker_id}_{i}",
                scene="diwali",
                left_variant="base",
                choice="Left",
                winner="base",
                timestamp="2026-11-12T12:00:00Z",
                lock=test_lock,
            )

    threads = [threading.Thread(target=worker, args=(wid,)) for wid in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert csv_file.exists()
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 40  # 4 threads * 10 appends
        assert reader[0]["rater_id"].startswith("rater_")
        assert reader[0]["choice"] == "Left"
        assert reader[0]["winner"] == "base"


# ---------------------------------------------------------------------------
# 5. blind_test.py builds Blocks object without launching
# ---------------------------------------------------------------------------


def test_blind_test_app_builds(tmp_path):
    fake_pairs_file = tmp_path / "test_pairs.json"
    fake_pairs_file.write_text("[]", encoding="utf-8")
    fake_csv_file = tmp_path / "test_responses.csv"

    demo = build_app(pairs_file=fake_pairs_file, csv_path=fake_csv_file)
    assert isinstance(demo, gr.Blocks)
