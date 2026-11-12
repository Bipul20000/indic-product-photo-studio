"""Unit tests for LoRA dataset builder logic with CLIP relevance filtering (no network calls, no torch)."""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from scripts.build_lora_dataset import (
    build_caption,
    compute_clip_probs,
    filter_and_rank_by_clip,
    process_and_crop,
    compute_phash,
    is_duplicate,
    clean_author,
    clean_license,
    check_person_yolo,
    check_dark_or_low_res,
    save_metadata_csv,
    load_metadata_csv,
)


def test_caption_builder():
    trigger = "indiandiwali"
    phrase = "rows of lit diyas"
    desc = "festive Diwali decoration with diyas, marigold flowers and warm lights"
    caption = build_caption(trigger, phrase, desc)
    expected = "indiandiwali, rows of lit diyas, festive Diwali decoration with diyas, marigold flowers and warm lights"
    assert caption == expected


def test_caption_builder_from_best_positive():
    """Verify caption built from best positive prompt caption and theme description."""
    trigger = "indianholi"
    best_pos_caption = "bowls of colored gulal powder"
    desc = "Holi festive celebration with vibrant dry gulal powders and festive colors"
    caption = build_caption(trigger, best_pos_caption, desc)
    assert caption == "indianholi, bowls of colored gulal powder, Holi festive celebration with vibrant dry gulal powders and festive colors"


def test_compute_clip_probs_fake_logits():
    """Verify softmax, summed p_pos, and best positive index calculation on fake logits."""
    # 2 positive prompts, 3 negative prompts
    logits = np.array([10.0, 8.0, 5.0, 4.0, 2.0])
    num_pos = 2

    # Analytical softmax calculation
    exp_l = np.exp(logits - np.max(logits))
    expected_probs = exp_l / np.sum(exp_l)
    expected_p_pos = float(np.sum(expected_probs[:num_pos]))
    expected_best_idx = 0

    p_pos, best_idx = compute_clip_probs(logits, num_pos)
    assert pytest.approx(p_pos, rel=1e-5) == expected_p_pos
    assert best_idx == expected_best_idx
    assert p_pos > 0.95  # positive logits are much higher than negative ones


def test_filter_and_rank_by_clip_top_k():
    """Verify filter_and_rank_by_clip sorts candidates by p_pos descending and caps at top_k."""
    # Create 80 candidates with p_pos from 0.01 to 0.80
    candidates = [{"id": i, "clip_p_pos": i * 0.01} for i in range(1, 81)]
    # Candidates with p_pos >= 0.6 are those with i in [60..80] -> 21 candidates (< 40)
    # Threshold will relax to 0.5: i in [50..80] -> 31 candidates (< 40)
    # Threshold will relax to 0.4: i in [40..80] -> 41 candidates (>= 40)
    kept, final_thresh, passed_cnt = filter_and_rank_by_clip(
        candidates,
        min_count=40,
        top_k=25,
        thresholds=(0.6, 0.5, 0.4),
    )

    assert final_thresh == 0.4
    assert passed_cnt == 41
    assert len(kept) == 25
    # Verify descending sort order
    p_values = [c["clip_p_pos"] for c in kept]
    assert p_values == sorted(p_values, reverse=True)
    assert kept[0]["clip_p_pos"] == 0.80
    assert kept[0]["id"] == 80


def test_threshold_relaxation_sequence():
    """Verify that threshold starts at 0.6 and only relaxes to 0.5 or 0.4 when count < 40."""
    # Case 1: 50 candidates with p_pos >= 0.6 -> stays at 0.6
    cands_high = [{"clip_p_pos": 0.75} for _ in range(50)]
    kept, thresh, count = filter_and_rank_by_clip(cands_high, min_count=40, top_k=60)
    assert thresh == 0.6
    assert count == 50
    assert len(kept) == 50

    # Case 2: 25 candidates >= 0.6, 25 candidates in [0.5, 0.59] (total 50 >= 0.5) -> relaxes to 0.5
    cands_mid = [{"clip_p_pos": 0.65} for _ in range(25)] + [{"clip_p_pos": 0.55} for _ in range(25)]
    kept, thresh, count = filter_and_rank_by_clip(cands_mid, min_count=40, top_k=60)
    assert thresh == 0.5
    assert count == 50
    assert len(kept) == 50

    # Case 3: 15 candidates >= 0.6, 15 candidates >= 0.5, 20 candidates >= 0.4 (total 50 >= 0.4) -> relaxes to 0.4
    cands_low = (
        [{"clip_p_pos": 0.65} for _ in range(15)]
        + [{"clip_p_pos": 0.55} for _ in range(15)]
        + [{"clip_p_pos": 0.45} for _ in range(20)]
    )
    kept, thresh, count = filter_and_rank_by_clip(cands_low, min_count=40, top_k=60)
    assert thresh == 0.4
    assert count == 50
    assert len(kept) == 50


def test_process_and_crop_asymmetric_landscape():
    img = np.full((600, 800, 3), 128, dtype=np.uint8)
    crop = process_and_crop(img, target_size=512)
    assert crop.shape == (512, 512, 3)


def test_process_and_crop_asymmetric_portrait():
    img = np.full((800, 600, 3), 128, dtype=np.uint8)
    crop = process_and_crop(img, target_size=512)
    assert crop.shape == (512, 512, 3)


def test_dedupe_logic_synthetic():
    import cv2

    rng = np.random.default_rng(42)
    base = rng.integers(50, 200, (600, 600, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (15, 15), 0)

    h_base = compute_phash(base)
    existing_hashes = [h_base]

    assert is_duplicate(h_base, existing_hashes, max_distance=6) is True

    noise = rng.integers(-2, 3, size=base.shape)
    noisy = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    h_noisy = compute_phash(noisy)
    assert is_duplicate(h_noisy, existing_hashes, max_distance=6) is True

    different = 255 - base
    h_diff = compute_phash(different)
    assert is_duplicate(h_diff, existing_hashes, max_distance=6) is False


def test_author_cap():
    author_counts = {}
    max_author = 8
    accepted = 0
    dropped = 0

    candidates = ["AuthorA"] * 12
    for author in candidates:
        current_count = author_counts.get(author, 0)
        if current_count >= max_author:
            dropped += 1
        else:
            author_counts[author] = current_count + 1
            accepted += 1

    assert accepted == 8
    assert dropped == 4
    assert author_counts["AuthorA"] == 8


def test_metadata_csv_roundtrip_with_clip_columns(tmp_path: Path):
    """Verify that metadata CSV saves and loads clip_p_pos and clip_best_positive."""
    csv_path = tmp_path / "metadata.csv"
    records = [
        {
            "filename": "001.jpg",
            "theme": "diwali",
            "term": "Diwali decoration",
            "caption": "indiandiwali, Diwali decorations with lit diyas, festive Diwali decoration",
            "source_url": "https://example.com/001.jpg",
            "page_url": "https://commons.wikimedia.org/wiki/File:001.jpg",
            "author": "Alice",
            "license": "CC BY-SA 4.0",
            "phash": "8f8f8f8f8f8f8f8f",
            "clip_p_pos": 0.8421,
            "clip_best_positive": "a photo of Diwali decorations with lit diyas",
        },
        {
            "filename": "002.jpg",
            "theme": "holi",
            "term": "Holi colors",
            "caption": "indianholi, bowls of colored gulal powder, vibrant Holi celebration",
            "source_url": "https://example.com/002.jpg",
            "page_url": "https://commons.wikimedia.org/wiki/File:002.jpg",
            "author": "Bob",
            "license": "CC0",
            "phash": "a0a0a0a0a0a0a0a0",
            "clip_p_pos": 0.7654,
            "clip_best_positive": "bowls and heaps of colorful holi gulal powder",
        },
    ]

    save_metadata_csv(records, csv_path)
    loaded = load_metadata_csv(csv_path)

    assert len(loaded) == 2
    assert loaded[0]["filename"] == "001.jpg"
    assert loaded[0]["theme"] == "diwali"
    assert loaded[0]["clip_best_positive"] == "a photo of Diwali decorations with lit diyas"
    assert float(loaded[0]["clip_p_pos"]) == pytest.approx(0.8421, abs=1e-4)

    assert loaded[1]["filename"] == "002.jpg"
    assert loaded[1]["clip_best_positive"] == "bowls and heaps of colorful holi gulal powder"
    assert float(loaded[1]["clip_p_pos"]) == pytest.approx(0.7654, abs=1e-4)

    # Check header explicitly
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    assert "clip_p_pos" in header
    assert "clip_best_positive" in header


def test_mock_person_detector():
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)

    def make_mock_model(cls_id, conf, xyxy):
        mock_box = MagicMock()
        mock_box.cls = [MagicMock(item=lambda: cls_id)]
        mock_box.conf = [MagicMock(item=lambda: conf)]
        mock_box.xyxy = [MagicMock(tolist=lambda: xyxy)]

        mock_res = MagicMock()
        mock_res.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.names = {0: "person", 1: "bicycle"}
        mock_model.return_value = [mock_res]
        return mock_model

    m1 = make_mock_model(cls_id=0, conf=0.50, xyxy=[0, 0, 200, 100])
    assert check_person_yolo(img, m1) is True

    m2 = make_mock_model(cls_id=0, conf=0.90, xyxy=[0, 0, 100, 100])
    assert check_person_yolo(img, m2) is False


def test_dark_or_low_res_check():
    small_img = np.full((500, 800, 3), 128, dtype=np.uint8)
    assert check_dark_or_low_res(small_img) is True

    dark_img = np.full((700, 700, 3), 30, dtype=np.uint8)
    assert check_dark_or_low_res(dark_img) is True

    good_img = np.full((700, 700, 3), 128, dtype=np.uint8)
    assert check_dark_or_low_res(good_img) is False


def test_category_exists_check_mocked():
    from scripts.build_lora_dataset import check_category_exists_and_has_files

    session_with_files = MagicMock()
    resp_with_files = MagicMock()
    resp_with_files.status_code = 200
    resp_with_files.json.return_value = {
        "query": {"pages": {"101": {"pageid": 101, "title": "Category:Diwali", "categoryinfo": {"files": 25}}}}
    }
    session_with_files.get.return_value = resp_with_files
    assert check_category_exists_and_has_files(session_with_files, "Diwali") is True

    session_empty = MagicMock()
    resp_empty = MagicMock()
    resp_empty.status_code = 200
    resp_empty.json.return_value = {
        "query": {"pages": {"102": {"pageid": 102, "title": "Category:EmptyCat", "categoryinfo": {"files": 0}}}}
    }
    session_empty.get.return_value = resp_empty
    assert check_category_exists_and_has_files(session_empty, "EmptyCat") is False


def test_lora_themes_yaml_configuration():
    """Verify themes in configs/lora_themes.yaml have clip_positive, clip_negative, and updated categories."""
    import yaml

    config_path = Path("configs/lora_themes.yaml")
    assert config_path.exists()

    with open(config_path, "r", encoding="utf-8") as f:
        themes = yaml.safe_load(f)

    for theme_name, theme in themes.items():
        assert "clip_positive" in theme
        assert "clip_negative" in theme
        assert len(theme["clip_positive"]) >= 3
        assert len(theme["clip_negative"]) >= 6
        for p in theme["clip_positive"]:
            assert "prompt" in p
            assert "caption" in p

    assert themes["diwali"]["categories"] == ["Diwali", "Diyas"]
    assert themes["raksha_bandhan"]["categories"] == ["Rakhi", "Raksha Bandhan"]
    assert themes["holi"]["categories"] == ["Holi"]
    assert themes["wedding"]["categories"] == ["Weddings in India"]
    assert themes["indian_home"]["categories"] == []
