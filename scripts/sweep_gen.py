#!/usr/bin/env python3
"""Parameter sweep for background generation with ControlNet and product fill.

Usage:
    python scripts/sweep_gen.py --fake-generator
    python scripts/sweep_gen.py --scenes diwali,indian_home --seed 0
"""

import argparse
import csv
from pathlib import Path
import sys
import time
from typing import List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import cv2
import numpy as np

from studio.generate.backgrounds import BackgroundGenerator, FakeBackgroundGenerator
from studio.pipeline import run_pipeline

DEFAULT_PRODUCTS = ["Junghans", "Khalili", "2026.06.05_USSR", "Cup_of_coffee"]
DEFAULT_SCENES = ["diwali", "indian_home"]

# Grid of 6 configs: controlnet_scale in {1.0, 0.7, 0.5} x fill in {0.80, 0.65}
SWEEP_CONFIGS = [
    (1.0, 0.80),
    (1.0, 0.65),
    (0.7, 0.80),
    (0.7, 0.65),
    (0.5, 0.80),
    (0.5, 0.65),
]


def get_sweep_configs() -> List[Tuple[float, float]]:
    """Return the 6 parameter combinations for the sweep."""
    return list(SWEEP_CONFIGS)


def config_slug(cn_scale: float, fill: float) -> str:
    """Format directory-safe slug for a parameter configuration."""
    return f"cn{cn_scale:.1f}_fill{fill:.2f}"


def config_label(cn_scale: float, fill: float) -> str:
    """Format human-readable label for a parameter configuration."""
    fill_str = f"{fill:.1f}" if fill == 0.8 else f"{fill:.2f}"
    return f"cn={cn_scale:.1f} fill={fill_str}"


def find_product_files(originals_dir: Path | str, product_prefixes: List[str]) -> List[dict]:
    """Match products in originals directory by filename prefix."""
    orig_path = Path(originals_dir)
    matched = []
    all_files = sorted(list(orig_path.glob("*.jpg")) + list(orig_path.glob("*.png")))

    for prefix in product_prefixes:
        match_file = None
        for f in all_files:
            if f.name.startswith(prefix):
                match_file = f
                break
        if match_file is None:
            # Fallback substring match
            for f in all_files:
                if prefix.lower() in f.name.lower():
                    match_file = f
                    break
        if match_file is not None:
            matched.append({
                "prefix": prefix,
                "stem": match_file.stem,
                "path": match_file,
            })
        else:
            print(f"WARNING: No product photo found matching prefix '{prefix}' in {orig_path}")

    return matched


def create_sweep_contact_sheet(
    products_and_scenes: List[Tuple[dict, str]],
    configs: List[Tuple[float, float]],
    out_dir: Path,
    thumb_size: int = 200,
    header_height: int = 50,
) -> None:
    """Build outputs/sweep/sweep.jpg contact sheet with 8 rows and 7 columns."""
    num_rows = len(products_and_scenes)
    num_cols = len(configs) + 1  # 1 for original, 6 for configs

    total_w = num_cols * thumb_size
    total_h = header_height + num_rows * thumb_size

    sheet = np.full((total_h, total_w, 3), 245, dtype=np.uint8)

    # Draw header row
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
    thick = 1

    # Col 0: "Original"
    cv2.putText(sheet, "Original", (15, 32), font, font_scale, (40, 40, 40), thick, cv2.LINE_AA)

    # Col 1..6: Config labels
    for c_idx, (cn, fill) in enumerate(configs):
        x = (c_idx + 1) * thumb_size
        label = config_label(cn, fill)
        cv2.putText(sheet, label, (x + 10, 32), font, font_scale, (40, 40, 40), thick, cv2.LINE_AA)

    # Draw separator line below header
    cv2.line(sheet, (0, header_height), (total_w, header_height), (200, 200, 200), 1)

    # Fill rows
    for r_idx, (prod, scene) in enumerate(products_and_scenes):
        y_top = header_height + r_idx * thumb_size
        stem = prod["stem"]

        # Col 0: Original image
        orig_img = cv2.imread(str(prod["path"]))
        if orig_img is not None:
            # Letterbox or center crop to thumb_size x thumb_size
            h, w = orig_img.shape[:2]
            scale = min(thumb_size / h, thumb_size / w)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            res = cv2.resize(orig_img, (nw, nh), interpolation=cv2.INTER_AREA)
            tile = np.full((thumb_size, thumb_size, 3), 255, dtype=np.uint8)
            y_off = (thumb_size - nh) // 2
            x_off = (thumb_size - nw) // 2
            tile[y_off:y_off + nh, x_off:x_off + nw] = res
        else:
            tile = np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)

        # Draw product label on original tile
        prod_short = stem[:16]
        cv2.putText(tile, f"{prod_short}", (5, thumb_size - 22), font, 0.40, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(tile, f"({scene})", (5, thumb_size - 6), font, 0.38, (80, 80, 80), 1, cv2.LINE_AA)

        sheet[y_top:y_top + thumb_size, 0:thumb_size] = tile

        # Cols 1..6: Generated images
        for c_idx, (cn, fill) in enumerate(configs):
            cfg_dir = out_dir / config_slug(cn, fill)
            gen_path = cfg_dir / f"{stem}_{scene}.jpg"

            if gen_path.exists():
                g_img = cv2.imread(str(gen_path))
                if g_img is not None:
                    g_thumb = cv2.resize(g_img, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
                else:
                    g_thumb = np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)
            else:
                g_thumb = np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)

            col_x = (c_idx + 1) * thumb_size
            sheet[y_top:y_top + thumb_size, col_x:col_x + thumb_size] = g_thumb

    overview_path = out_dir / "sweep.jpg"
    cv2.imwrite(str(overview_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"\nSweep overview contact sheet saved to {overview_path}")


def main():
    parser = argparse.ArgumentParser(description="Sweep generation parameters across products and scenes")
    parser.add_argument(
        "--products",
        nargs="+",
        default=DEFAULT_PRODUCTS,
        help="Product filename prefixes to match in originals directory",
    )
    parser.add_argument(
        "--originals-dir",
        default="data/coach_eval/originals/good",
        help="Path to originals good directory",
    )
    parser.add_argument(
        "--scenes",
        default="diwali,indian_home",
        help="Comma-separated scenes to generate",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument(
        "--fake-generator",
        action="store_true",
        help="Use deterministic FakeBackgroundGenerator instead of GPU diffusion model",
    )
    parser.add_argument("--out", default="outputs/sweep", help="Output directory for sweep results")
    args = parser.parse_args()

    scene_list = [s.strip() for s in args.scenes.split(",") if s.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Match products
    matched_products = find_product_files(args.originals_dir, args.products)
    if not matched_products:
        print(f"ERROR: No products matched in {args.originals_dir}")
        return

    print(f"Matched {len(matched_products)} products:")
    for p in matched_products:
        print(f"  [{p['prefix']}] -> {p['path'].name}")

    # Build (product, scene) combinations
    products_and_scenes = []
    for prod in matched_products:
        for scene in scene_list:
            products_and_scenes.append((prod, scene))

    print(f"\nTotal combinations to generate: {len(products_and_scenes)} ({len(matched_products)} products x {len(scene_list)} scenes)")

    # 2. Setup generator
    if args.fake_generator:
        print("Using FakeBackgroundGenerator (local test mode)")
        generator = FakeBackgroundGenerator()
    else:
        print("Using GPU BackgroundGenerator")
        generator = BackgroundGenerator()

    configs = get_sweep_configs()
    manifest_rows = []

    print(f"\nRunning sweep over {len(configs)} configurations...")

    # Open manifest file
    manifest_path = out_dir / "manifest.csv"
    manifest_file = open(manifest_path, "w", newline="", encoding="utf-8")
    manifest_writer = csv.DictWriter(
        manifest_file,
        fieldnames=[
            "config",
            "controlnet_scale",
            "fill",
            "product",
            "scene",
            "seed",
            "attempts",
            "person_flag",
            "seconds",
        ],
    )
    manifest_writer.writeheader()

    per_config_stats = []

    for cn_scale, fill in configs:
        slug = config_slug(cn_scale, fill)
        lbl = config_label(cn_scale, fill)
        cfg_dir = out_dir / slug
        cfg_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Config: {lbl} (dir: {slug}) ---")

        images_generated = 0
        attempts_list = []
        person_flag_count = 0
        seconds_list = []

        for prod, scene in products_and_scenes:
            stem = prod["stem"]
            out_file = cfg_dir / f"{stem}_{scene}.jpg"

            t0 = time.time()
            res = run_pipeline(
                image_path=prod["path"],
                scene=scene,
                force=True,
                seed=args.seed,
                fill=fill,
                controlnet_scale=cn_scale,
                generator=generator,
            )
            elapsed = time.time() - t0

            if res.image is not None:
                cv2.imwrite(str(out_file), res.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                images_generated += 1

            attempts_list.append(res.attempts)
            if res.person_flag:
                person_flag_count += 1
            seconds_list.append(elapsed)

            row = {
                "config": slug,
                "controlnet_scale": cn_scale,
                "fill": fill,
                "product": stem,
                "scene": scene,
                "seed": res.seed,
                "attempts": res.attempts,
                "person_flag": res.person_flag,
                "seconds": round(elapsed, 3),
            }
            manifest_writer.writerow(row)
            manifest_file.flush()

        mean_attempts = float(np.mean(attempts_list)) if attempts_list else 0.0
        mean_seconds = float(np.mean(seconds_list)) if seconds_list else 0.0

        per_config_stats.append({
            "config": lbl,
            "slug": slug,
            "images_generated": images_generated,
            "mean_attempts": mean_attempts,
            "person_flags": person_flag_count,
            "mean_seconds": mean_seconds,
        })

    manifest_file.close()
    print(f"\nManifest saved to {manifest_path}")

    # 3. Create sweep contact sheet
    create_sweep_contact_sheet(
        products_and_scenes=products_and_scenes,
        configs=configs,
        out_dir=out_dir,
        thumb_size=200,
    )

    # 4. Print per-config statistics table
    print("\n" + "=" * 80)
    print(f"{'Config':<20} {'Images':<10} {'Mean Attempts':<16} {'Person Flags':<16} {'Mean Seconds':<14}")
    print("-" * 80)
    for s in per_config_stats:
        print(
            f"{s['config']:<20} {s['images_generated']:<10} {s['mean_attempts']:<16.2f} "
            f"{s['person_flags']:<16} {s['mean_seconds']:<14.3f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
