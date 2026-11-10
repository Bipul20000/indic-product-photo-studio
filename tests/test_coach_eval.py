import csv
import hashlib
from pathlib import Path
import sys
import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.make_coach_eval import build_evaluation_set


def _create_synthetic_image(seed: int, width: int = 800, height: int = 800) -> np.ndarray:
    """Generate a reproducible patterned synthetic image."""
    rng = np.random.RandomState(seed)
    img = rng.randint(50, 200, (height, width, 3), dtype=np.uint8)
    # Add identifiable shapes so edges exist
    cv2.circle(img, (width // 2, height // 2), 150, (30, 220, 220), -1)
    cv2.rectangle(img, (100, 100), (300, 300), (220, 30, 30), 4)
    return img


def test_make_coach_eval_builder(tmp_path: Path):
    """Test the evaluation set builder on 3 synthetic original images."""
    originals_dir = tmp_path / "originals"
    originals_dir.mkdir(parents=True)

    # Create 3 synthetic original images
    for i in range(1, 4):
        img = _create_synthetic_image(seed=i * 10)
        cv2.imwrite(str(originals_dir / f"test_item_{i}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    out_images_1 = tmp_path / "run1" / "images"
    out_labels_1 = tmp_path / "run1" / "labels.csv"

    records_1 = build_evaluation_set(originals_dir, out_images_1, out_labels_1, seed=42)

    # 3 originals -> 3 original good + 3 synthetic good + 6 synthetic bad = 12 total
    assert len(records_1) == 12

    # Verify CSV file and columns
    assert out_labels_1.exists()
    with open(out_labels_1, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["filename", "label", "reason", "source"]
        rows = list(reader)
        assert len(rows) == 12

    # Check distribution
    goods = [r for r in rows if r["label"] == "good"]
    bads = [r for r in rows if r["label"] == "bad"]
    assert len(goods) == 6
    assert len(bads) == 6

    orig_goods = [r for r in goods if r["source"] == "original"]
    synth_goods = [r for r in goods if r["source"] == "synthetic"]
    assert len(orig_goods) == 3
    assert len(synth_goods) == 3

    for r in bads:
        assert r["source"] == "synthetic"
        assert r["reason"] in {"blur", "dark", "bright", "lowres", "clutter"}

    # Verify all generated images exist on disk
    for r in rows:
        img_path = out_images_1 / r["filename"]
        assert img_path.exists()
        assert img_path.stat().st_size > 0

    # Second run to test exact determinism
    out_images_2 = tmp_path / "run2" / "images"
    out_labels_2 = tmp_path / "run2" / "labels.csv"

    records_2 = build_evaluation_set(originals_dir, out_images_2, out_labels_2, seed=42)
    assert records_1 == records_2

    # Compare hashes of files
    for r in rows:
        fname = r["filename"]
        file1 = out_images_1 / fname
        file2 = out_images_2 / fname
        hash1 = hashlib.sha256(file1.read_bytes()).hexdigest()
        hash2 = hashlib.sha256(file2.read_bytes()).hexdigest()
        assert hash1 == hash2, f"Determinism mismatch for {fname}"
