"""Tests for product-aware dynamic contextual prompt builder."""

import pytest

from studio.generate.prompt_builder import build_contextual_prompt, detect_product_category


def test_detect_product_category():
    """Verify category detection keywords and placement clauses."""
    cat, clause = detect_product_category("terracotta chai mug")
    assert cat == "drinkware"
    assert "standing stably" in clause

    cat, clause = detect_product_category("handmade leather jutti")
    assert cat == "footwear"
    assert "ground surface" in clause

    cat, clause = detect_product_category("brass ganesha diya")
    assert cat == "decor"
    assert "decorative centerpiece" in clause

    cat, _ = detect_product_category("unbranded mystery item")
    assert cat == "general"


def test_build_contextual_prompt_injection():
    """Verify product name is enriched and injected into raw template."""
    raw = "festive Diwali background with diyas, {product} in the foreground"
    prompt = build_contextual_prompt("diwali", raw, "brass diya", catalog_grade=True)

    assert "brass diya" in prompt
    assert "commercial e-commerce catalog aesthetic" in prompt
    assert "natural contact shadows" in prompt


def test_build_contextual_prompt_white_scene():
    """Verify white scene returns pure studio white prompt."""
    prompt = build_contextual_prompt("white", "anything", "mug")
    assert "pure white" in prompt
