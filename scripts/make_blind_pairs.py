#!/usr/bin/env python3
"""Generate blind evaluation pairs from paired comparison outputs.

Pairs base and LoRA images matching on product, scene, and seed for scenes:
diwali, raksha_bandhan, wedding, and indian_home (skipping holi).
Filters out pairs with mean absolute pixel difference < 2.
Selects up to 6 pairs per scene, with at most 2 per product (seed=42).
Outputs configs/blind_pairs.json.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import cv2
import numpy as np

ALLOWED_SCENES = ["diwali", "raksha_bandhan", "wedding", "indian_home"]


def compute_pixel_diff(img1_path: Union[str, Path], img2_path: Union[str, Path]) -> float:
    """Compute mean absolute pixel difference between two images."""
    im1 = cv2.imread(str(img1_path))
    im2 = cv2.imread(str(img2_path))
    if im1 is None or im2 is None or im1.shape != im2.shape:
        return 0.0
    return float(np.mean(np.abs(im1.astype(np.float32) - im2.astype(np.float32))))


def make_blind_pairs(
    compare_dir: Union[str, Path] = "outputs/compare",
    manifest_path: Optional[Union[str, Path]] = None,
    out_json: Union[str, Path] = "configs/blind_pairs.json",
    seed: int = 42,
    min_diff: float = 2.0,
    max_per_scene: int = 6,
    max_per_product: int = 2,
) -> List[Dict[str, Any]]:
    """Build blind test image pairs according to selection constraints.

    Args:
        compare_dir: Directory containing base/ and lora/ subdirectories.
        manifest_path: Optional path to compare manifest.csv.
        out_json: Destination path for JSON pairs configuration.
        seed: Random seed for deterministic selection.
        min_diff: Minimum mean absolute pixel difference threshold.
        max_per_scene: Maximum number of pairs selected per scene.
        max_per_product: Maximum pairs per product within each scene.

    Returns:
        List of selected pair dictionaries.
    """
    comp_p = Path(compare_dir)
    base_dir = comp_p / "base"
    lora_dir = comp_p / "lora"

    # 1. Discover potential pairs
    candidate_keys = set()

    # Try reading from manifest if available
    m_path = Path(manifest_path) if manifest_path else comp_p / "manifest.csv"
    if m_path.exists():
        with open(m_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sc = row.get("scene", "").strip()
                if sc in ALLOWED_SCENES:
                    prod = row.get("product", "").strip()
                    try:
                        s_val = int(row.get("seed", 0))
                    except ValueError:
                        s_val = 0
                    if prod:
                        candidate_keys.add((prod, sc, s_val))

    # Also scan directory directly in case manifest is absent or partial
    if base_dir.exists() and lora_dir.exists():
        for sc in ALLOWED_SCENES:
            b_sc_dir = base_dir / sc
            l_sc_dir = lora_dir / sc
            if b_sc_dir.exists() and l_sc_dir.exists():
                for b_file in b_sc_dir.glob("*.jpg"):
                    stem = b_file.stem
                    if "_s" in stem:
                        parts = stem.rsplit("_s", 1)
                        prod = parts[0]
                        try:
                            s_val = int(parts[1])
                            l_file = l_sc_dir / b_file.name
                            if l_file.exists():
                                candidate_keys.add((prod, sc, s_val))
                        except ValueError:
                            pass

    # 2. Filter candidate pairs by file existence and pixel difference >= min_diff
    candidates_by_scene: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for prod, sc, s_val in sorted(candidate_keys):
        b_path = base_dir / sc / f"{prod}_s{s_val}.jpg"
        l_path = lora_dir / sc / f"{prod}_s{s_val}.jpg"

        if not (b_path.exists() and l_path.exists()):
            continue

        diff = compute_pixel_diff(b_path, l_path)
        if diff < min_diff:
            continue

        candidates_by_scene[sc].append({
            "product": prod,
            "scene": sc,
            "seed": s_val,
            "base_path": str(b_path),
            "lora_path": str(l_path),
            "pixel_diff": round(diff, 3),
        })

    # 3. Deterministic sampling: up to max_per_scene per scene, <= max_per_product per product
    rng = random.Random(seed)
    selected_pairs: List[Dict[str, Any]] = []
    pair_counter = 1

    for sc in ALLOWED_SCENES:
        cands = list(candidates_by_scene.get(sc, []))
        # Stable sort then shuffle with seeded RNG
        cands.sort(key=lambda x: (x["product"], x["seed"]))
        rng.shuffle(cands)

        scene_selected: List[Dict[str, Any]] = []
        prod_counts: Dict[str, int] = defaultdict(int)

        for c in cands:
            if len(scene_selected) >= max_per_scene:
                break
            prod = c["product"]
            if prod_counts[prod] < max_per_product:
                prod_counts[prod] += 1
                pair_dict = {
                    "pair_id": f"pair_{pair_counter:03d}",
                    "scene": c["scene"],
                    "product": c["product"],
                    "seed": c["seed"],
                    "base_path": c["base_path"],
                    "lora_path": c["lora_path"],
                    "pixel_diff": c["pixel_diff"],
                }
                scene_selected.append(pair_dict)
                pair_counter += 1

        selected_pairs.extend(scene_selected)

    # 4. Save to JSON
    out_p = Path(out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(selected_pairs, f, indent=2)

    print(f"Generated {len(selected_pairs)} blind evaluation pairs across {len(ALLOWED_SCENES)} scenes.")
    for sc in ALLOWED_SCENES:
        n_sc = sum(1 for p in selected_pairs if p["scene"] == sc)
        print(f"  {sc}: {n_sc} pairs")
    print(f"Saved pairs configuration to: {out_p}")

    return selected_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Generate blind evaluation pairs JSON.")
    parser.add_argument("--compare-dir", type=str, default="outputs/compare", help="Directory with base/ and lora/ subdirs")
    parser.add_argument("--manifest", type=str, default=None, help="Optional path to manifest.csv")
    parser.add_argument("--out", type=str, default="configs/blind_pairs.json", help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for selection (default: 42)")
    parser.add_argument("--min-diff", type=float, default=2.0, help="Minimum mean absolute pixel diff (default: 2.0)")
    parser.add_argument("--max-per-scene", type=int, default=6, help="Max pairs per scene (default: 6)")
    parser.add_argument("--max-per-product", type=int, default=2, help="Max pairs per product per scene (default: 2)")
    return parser.parse_args()


def main():
    args = parse_args()
    make_blind_pairs(
        compare_dir=args.compare_dir,
        manifest_path=args.manifest,
        out_json=args.out,
        seed=args.seed,
        min_diff=args.min_diff,
        max_per_scene=args.max_per_scene,
        max_per_product=args.max_per_product,
    )


if __name__ == "__main__":
    main()
