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
    "appliance": (
        [
            "iron", "steamer", "microwave", "toaster", "blender", "mixer", "grinder",
            "cooker", "dryer", "fan", "heater", "vacuum", "appliance", "hair dryer",
            "kettle", "purifier", "trimmer",
        ],
        "standing cleanly on a flat polished wooden table, with clear empty space around the product and no decorations touching it",
    ),
    "electronics": (
        [
            "phone", "mobile", "laptop", "tablet", "headphones", "earbuds", "speaker",
            "gadget", "watch", "smartwatch", "camera", "charger", "powerbank", "keyboard", "mouse",
        ],
        "resting cleanly on a sleek modern surface, with clean minimal space around the item",
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
        for kw in keywords:
            if " " in kw:
                if re.search(r"\b" + re.escape(kw) + r"\b", p_lower):
                    return cat_name, placement_clause
            elif kw in words:
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

    # 1. Clean product title and category
    prod_clean = product_name.strip() if product_name and product_name.strip() else ""
    cat_name = "general"
    if prod_clean:
        cat_name, placement = detect_product_category(prod_clean)
        prod_clause = f"a {prod_clean} {placement}"
    else:
        prod_clause = "the product standing stably on the surface"

    # 2. Category-Specific Scene Adaptation (e.g. Appliances / Electronics in Festive Scenes)
    if cat_name in ("appliance", "electronics") and scene in ("raksha_bandhan", "diwali", "wedding"):
        # For utility appliances, restrict festival elements strictly to distant room ambiance
        scene_ambiance = {
            "raksha_bandhan": "warm Raksha Bandhan festive lights and sweets placed neatly on a table in the distant blurred room background",
            "diwali": "warm glowing Diwali diyas and festive fairy lights softly blurred in the distant background room",
            "wedding": "luxurious Indian wedding lighting and floral decor softly blurred in the distant background room",
        }.get(scene, "festive warm background lights")

        prompt = (
            f"professional commercial product photo of a modern {prod_clean} standing on a clean flat polished surface, "
            f"with {scene_ambiance}, clean empty tabletop around the product, no decorations touching the item, "
            f"sharp product focus, 8k"
        )
    else:
        # Standard insertion into template
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
