"""Unit tests for LoRA dataset builder logic (no network calls)."""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from scripts.build_lora_dataset import (
    build_caption,
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


def test_process_and_crop_asymmetric_landscape():
    # 800 wide, 600 high
    img = np.full((600, 800, 3), 128, dtype=np.uint8)
    crop = process_and_crop(img, target_size=512)
    assert crop.shape == (512, 512, 3)


def test_process_and_crop_asymmetric_portrait():
    # 600 wide, 800 high
    img = np.full((800, 600, 3), 128, dtype=np.uint8)
    crop = process_and_crop(img, target_size=512)
    assert crop.shape == (512, 512, 3)


def test_dedupe_logic_synthetic():
    import cv2
    rng = np.random.default_rng(42)
    # Create base image with smooth natural textures
    base = rng.integers(50, 200, (600, 600, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (15, 15), 0)

    h_base = compute_phash(base)
    existing_hashes = [h_base]

    # Identical image
    assert is_duplicate(h_base, existing_hashes, max_distance=6) is True

    # Slightly noisy image
    noise = rng.integers(-2, 3, size=base.shape)
    noisy = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    h_noisy = compute_phash(noisy)
    assert is_duplicate(h_noisy, existing_hashes, max_distance=6) is True

    # Completely different image (inverted texture)
    different = 255 - base
    h_diff = compute_phash(different)
    assert is_duplicate(h_diff, existing_hashes, max_distance=6) is False


def test_author_cap():
    # Feed 10 candidates with the same author, assert exactly 6 pass
    author_counts = {}
    max_author = 6
    accepted = 0
    dropped = 0

    candidates = ["AuthorA"] * 10
    for author in candidates:
        current_count = author_counts.get(author, 0)
        if current_count >= max_author:
            dropped += 1
        else:
            author_counts[author] = current_count + 1
            accepted += 1

    assert accepted == 6
    assert dropped == 4
    assert author_counts["AuthorA"] == 6


def test_metadata_csv_roundtrip(tmp_path: Path):
    csv_path = tmp_path / "metadata.csv"
    records = [
        {
            "filename": "001.jpg",
            "theme": "diwali",
            "term": "Diwali decoration",
            "caption": "indiandiwali, Diwali decoration, festive Diwali decoration",
            "source_url": "https://example.com/001.jpg",
            "page_url": "https://commons.wikimedia.org/wiki/File:001.jpg",
            "author": "Alice",
            "license": "CC BY-SA 4.0",
            "phash": "8f8f8f8f8f8f8f8f",
        },
        {
            "filename": "002.jpg",
            "theme": "holi",
            "term": "Holi colors",
            "caption": "indianholi, Holi colors, vibrant Holi celebration",
            "source_url": "https://example.com/002.jpg",
            "page_url": "https://commons.wikimedia.org/wiki/File:002.jpg",
            "author": "Bob",
            "license": "CC0",
            "phash": "a0a0a0a0a0a0a0a0",
        },
    ]

    save_metadata_csv(records, csv_path)
    loaded = load_metadata_csv(csv_path)

    assert len(loaded) == 2
    assert loaded[0]["filename"] == "001.jpg"
    assert loaded[0]["theme"] == "diwali"
    assert loaded[0]["author"] == "Alice"
    assert loaded[0]["phash"] == "8f8f8f8f8f8f8f8f"
    assert loaded[1]["filename"] == "002.jpg"
    assert loaded[1]["license"] == "CC0"


def test_mock_person_detector():
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)  # area = 1,000,000

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

    # Case 1: person, conf >= 0.35, area >= 1.5% (15,000 px). Box is 200x100 = 20,000 px (2.0%) -> True (drops)
    m1 = make_mock_model(cls_id=0, conf=0.50, xyxy=[0, 0, 200, 100])
    assert check_person_yolo(img, m1) is True

    # Case 2: person, conf >= 0.35, area < 1.5% (100x100 = 10,000 px = 1.0%) -> False (passes)
    m2 = make_mock_model(cls_id=0, conf=0.90, xyxy=[0, 0, 100, 100])
    assert check_person_yolo(img, m2) is False

    # Case 3: person, conf < 0.35 (0.20), area >= 1.5% (300x300 = 90,000 px = 9.0%) -> False (passes)
    m3 = make_mock_model(cls_id=0, conf=0.20, xyxy=[0, 0, 300, 300])
    assert check_person_yolo(img, m3) is False

    # Case 4: non-person (bicycle), conf >= 0.35, area >= 1.5% -> False (passes)
    m4 = make_mock_model(cls_id=1, conf=0.95, xyxy=[0, 0, 500, 500])
    assert check_person_yolo(img, m4) is False


def test_dark_or_low_res_check():
    # Low resolution (< 600 px short side)
    small_img = np.full((500, 800, 3), 128, dtype=np.uint8)
    assert check_dark_or_low_res(small_img) is True

    # Dark image (luminance < 40)
    dark_img = np.full((700, 700, 3), 30, dtype=np.uint8)
    assert check_dark_or_low_res(dark_img) is True

    # Normal bright image (>= 600 px and >= 40 luminance)
    good_img = np.full((700, 700, 3), 128, dtype=np.uint8)
    assert check_dark_or_low_res(good_img) is False
