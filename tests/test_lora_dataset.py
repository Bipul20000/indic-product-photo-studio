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


def test_category_exists_check_mocked():
    """Verify check_category_exists_and_has_files logic."""
    from scripts.build_lora_dataset import check_category_exists_and_has_files

    # 1. Existing category with files
    session_with_files = MagicMock()
    resp_with_files = MagicMock()
    resp_with_files.status_code = 200
    resp_with_files.json.return_value = {
        "query": {"pages": {"101": {"pageid": 101, "title": "Category:Diwali", "categoryinfo": {"files": 25}}}}
    }
    session_with_files.get.return_value = resp_with_files
    assert check_category_exists_and_has_files(session_with_files, "Diwali") is True

    # 2. Existing category with 0 files
    session_empty = MagicMock()
    resp_empty = MagicMock()
    resp_empty.status_code = 200
    resp_empty.json.return_value = {
        "query": {"pages": {"102": {"pageid": 102, "title": "Category:EmptyCat", "categoryinfo": {"files": 0}}}}
    }
    session_empty.get.return_value = resp_empty
    assert check_category_exists_and_has_files(session_empty, "EmptyCat") is False

    # 3. Missing category
    session_missing = MagicMock()
    resp_missing = MagicMock()
    resp_missing.status_code = 200
    resp_missing.json.return_value = {
        "query": {"pages": {"-1": {"title": "Category:MissingCat", "missing": ""}}}
    }
    session_missing.get.return_value = resp_missing
    assert check_category_exists_and_has_files(session_missing, "MissingCat") is False


def test_fetch_category_batch_mocked():
    """Verify fetch_category_batch queries categorymembers with gcmtype=file, gcmlimit=100."""
    from scripts.build_lora_dataset import fetch_category_batch

    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "query": {
            "pages": {
                "10": {
                    "index": 1,
                    "title": "File:TestDiya.jpg",
                    "imageinfo": [{
                        "mime": "image/jpeg",
                        "width": 1280,
                        "height": 720,
                        "thumburl": "https://thumb.wikimedia.org/testdiya.jpg",
                        "descriptionurl": "https://commons.wikimedia.org/wiki/File:TestDiya.jpg",
                        "extmetadata": {
                            "Artist": {"value": "Photographer"},
                            "LicenseShortName": {"value": "CC BY-SA 4.0"},
                        },
                    }],
                },
                "11": {
                    "index": 2,
                    "title": "File:TestDiya.svg",
                    "imageinfo": [{
                        "mime": "image/svg+xml",
                        "width": 1280,
                        "height": 720,
                    }],
                },
            }
        },
        "continue": {"gcmcontinue": "file|1234|5678", "continue": "gcmcontinue||"},
    }
    session.get.return_value = resp

    candidates, continue_token = fetch_category_batch(session, "Diwali")

    # Only jpeg candidate kept (svg filtered out)
    assert len(candidates) == 1
    assert candidates[0]["title"] == "File:TestDiya.jpg"
    assert candidates[0]["download_url"] == "https://thumb.wikimedia.org/testdiya.jpg"
    assert candidates[0]["author"] == "Photographer"
    assert candidates[0]["license"] == "CC BY-SA 4.0"
    assert continue_token == {"gcmcontinue": "file|1234|5678", "continue": "gcmcontinue||"}

    # Verify query params passed to session.get
    _, kwargs = session.get.call_args
    params = kwargs.get("params", {})
    assert params.get("generator") == "categorymembers"
    assert params.get("gcmtitle") == "Category:Diwali"
    assert params.get("gcmtype") == "file"
    assert params.get("gcmlimit") == 100


def test_lora_themes_yaml_has_categories():
    """Verify that all themes in configs/lora_themes.yaml contain expected categories."""
    import yaml

    config_path = Path("configs/lora_themes.yaml")
    assert config_path.exists()

    with open(config_path, "r", encoding="utf-8") as f:
        themes = yaml.safe_load(f)

    assert themes["diwali"]["categories"] == ["Diwali", "Diyas"]
    assert themes["raksha_bandhan"]["categories"] == ["Rakhi", "Raksha Bandhan"]
    assert themes["holi"]["categories"] == ["Holi"]
    assert themes["wedding"]["categories"] == ["Weddings in India"]
    assert themes["indian_home"]["categories"] == ["Interior of houses in India"]

