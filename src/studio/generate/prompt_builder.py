"""Product-aware dynamic contextual prompt engine for generative studio backgrounds.

Enriches raw scene prompts with fine-grained product category understanding,
ground plane positioning, contact lighting, and commercial catalog cues.
"""

import re
from typing import Optional, Tuple


PRODUCT_CATEGORIES = {
    "drinkware": (
        ["mug", "cup", "glass", "bottle", "tumbler", "flask", "jar", "teapot", "kettle"],
        "standing stably on the surface with subtle base reflections",
    ),
    "footwear": (
        ["shoe", "sneakers", "jutti", "sandals", "boots", "heels", "slippers", "mojari"],
        "resting naturally on a clean ground surface with realistic floor contact",
    ),
    "cosmetics": (
        ["serum", "cream", "lotion", "perfume", "fragrance", "lipstick", "shampoo", "oil", "soap", "skincare"],
        "standing elegantly on a pristine surface with soft commercial studio reflections",
    ),
    "jewelry": (
        ["necklace", "earring", "ring", "pendant", "bracelet", "bangle", "chain", "brooch"],
        "meticulously displayed on a luxurious textured surface",
    ),
    "decor": (
        ["diya", "idol", "statue", "vase", "candle", "urli", "bowl", "pot", "figurine", "lamp", "lantern"],
        "displayed proudly as a decorative centerpiece on a polished surface",
    ),
    "apparel": (
        ["saree", "sari", "kurta", "dress", "shirt", "dupatta", "scarf", "shawl", "cloth", "fabric"],
        "elegantly presented on a textured background surface",
    ),
    "food": (
        ["tea", "coffee", "sweet", "mithai", "snack", "spice", "chocolate", "dry fruit", "laddu"],
        "presented appetizingly on a clean gourmet serving surface",
    ),
}


def detect_product_category(product_name: str) -> Tuple[str, str]:
    """Detect product category and appropriate surface placement descriptor.

    Args:
        product_name: Raw product name string.

    Returns:
        Tuple of (category_name, placement_clause).
    """
    p_lower = product_name.lower().strip()
    words = set(re.findall(r"\b\w+\b", p_lower))

    for cat_name, (keywords, placement_clause) in PRODUCT_CATEGORIES.items():
        if any(kw in words or kw in p_lower for kw in keywords):
            return cat_name, placement_clause

    return "general", "resting stably on a clean surface with natural contact lighting"


def build_contextual_prompt(
    scene: str,
    raw_prompt_template: str,
    product_name: Optional[str] = None,
    catalog_grade: bool = True,
) -> str:
    """Build an enriched, product-aware diffusion prompt for the requested scene.

    Args:
        scene: Scene name (e.g. 'diwali', 'wedding', 'indian_home', 'minimal', 'white').
        raw_prompt_template: Base prompt template from scenes.yaml.
        product_name: Optional user-supplied product title.
        catalog_grade: Whether to append commercial e-commerce catalog photographic modifiers.

    Returns:
        Harmonized, product-grounded prompt string.
    """
    if scene == "white":
        return "pure white background, clean commercial studio lighting"

    # 1. Clean product title
    prod_clean = product_name.strip() if product_name and product_name.strip() else ""
    if prod_clean:
        _, placement = detect_product_category(prod_clean)
        prod_clause = f"a {prod_clean} {placement}"
    else:
        prod_clause = "the product standing stably on the surface"

    # 2. Insert into template
    if "{product}" in raw_prompt_template:
        prompt = raw_prompt_template.replace("{product}", prod_clause)
    else:
        prompt = f"professional product photo of {prod_clause}, {raw_prompt_template}"

    # 3. Append catalog-grade photographic grounding cues if enabled
    if catalog_grade:
        catalog_cues = (
            ", commercial e-commerce catalog aesthetic, natural contact shadows, "
            "subtle ambient light reflections on surface, sharp focus on product"
        )
        if catalog_cues not in prompt:
            prompt = prompt.rstrip("., ") + catalog_cues

    return prompt
