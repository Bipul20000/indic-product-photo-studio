"""Tests for e-commerce catalog pack generation."""

import numpy as np
import pytest

from studio.app_logic import generate_catalog_pack
from studio.generate.backgrounds import FakeBackgroundGenerator


def test_generate_catalog_pack_generates_4_images(tmp_path):
    """Verify generate_catalog_pack outputs a 4-asset suite: Hero, Festive, Lifestyle, Minimal."""
    import cv2

    img_p = tmp_path / "test_prod.jpg"
    img = np.full((300, 300, 3), 200, dtype=np.uint8)
    cv2.circle(img, (150, 150), 70, (40, 100, 220), -1)
    cv2.imwrite(str(img_p), img)

    gen = FakeBackgroundGenerator()
    res = generate_catalog_pack(
        image_path=str(img_p),
        marketplace="square_1000",
        product_name="ceramic tea mug",
        seed=42,
        force_retake=True,
        generator=gen,
    )

    assert res.verdict in ("OK", "RETAKE")
    assert len(res.pack) == 4, "Catalog pack must contain exactly 4 assets"

    labels = [label for _, label in res.pack]
    assert any("Primary Hero" in l for l in labels)
    assert any("Festive Campaign" in l for l in labels)
    assert any("Lifestyle Interior" in l for l in labels)
    assert any("Minimal Studio" in l for l in labels)

    for img_arr, label in res.pack:
        assert isinstance(img_arr, np.ndarray)
        assert img_arr.shape == (1000, 1000, 3)
