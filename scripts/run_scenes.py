import argparse
import csv
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from studio.coach.coach import assess_with_segmentation
from studio.coach.config import CoachConfig
from studio.coach.io import load_image
from studio.compose.compose import _resolve_preset, composite, place_product
from studio.generate.backgrounds import BackgroundGenerator, FakeBackgroundGenerator
from studio.generate.config import GenConfig, load_scenes
from studio.pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate AI scene backgrounds for product photos.")
    parser.add_argument("--n-products", type=int, default=10, help="Number of passing products to process.")
    parser.add_argument("--scenes", type=str, default="all", help="Comma-separated scene names or 'all'.")
    parser.add_argument("--seeds", type=str, default="0,1", help="Comma-separated random seeds.")
    parser.add_argument("--preset", type=str, default="square_1000", help="Canvas preset name.")
    parser.add_argument("--out", type=str, default="outputs/base_gen", help="Output directory.")
    parser.add_argument("--originals", type=str, default="data/coach_eval/originals/good", help="Directory of original photos.")
    parser.add_argument("--lora", type=str, default=None, help="Optional path to LoRA weights.")
    parser.add_argument("--lora-scale", type=float, default=0.8, help="LoRA cross-attention scale.")
    parser.add_argument("--fake-generator", action="store_true", help="Use FakeBackgroundGenerator for CPU dry runs.")
    return parser.parse_args()


def load_exclusions(exclude_file: Path) -> List[str]:
    """Load exclusion prefixes from configs/coach_eval_exclude.txt."""
    if not exclude_file.exists():
        return []
    excl = []
    with open(exclude_file, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.split("#")[0].strip()
            if clean:
                excl.append(clean)
    return excl


def extract_family(filename: str) -> str:
    """Extract product family by lowercasing, stripping '_orig', and repeatedly stripping trailing '_<digits>'."""
    stem = Path(filename).stem.lower()
    if stem.endswith("_orig"):
        stem = stem[:-5]
    while True:
        m = re.search(r"_\d+$", stem)
        if m:
            stem = stem[:m.start()]
        else:
            break
    return stem


def pick_diverse_products(family_items: List[Tuple[Any, str]], n: int) -> List[Tuple[Any, str]]:
    """Pick n items evenly spaced across unique families using np.linspace with duplicates removed."""
    if len(family_items) <= n:
        return list(family_items)
    indices = np.linspace(0, len(family_items) - 1, n).round().astype(int)
    chosen = []
    seen_idx = set()
    for idx in indices:
        if idx not in seen_idx:
            seen_idx.add(idx)
            chosen.append(family_items[idx])
    return chosen


def make_thumb(image: np.ndarray, target_w: int = 224, target_h: int = 224) -> np.ndarray:
    """Resize image to fit within target_w x target_h box centered on white background."""
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_h, target_w, 3), 255, dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (nw, nh), interpolation=interp)

    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    x_off = (target_w - nw) // 2
    y_off = (target_h - nh) // 2
    canvas[y_off : y_off + nh, x_off : x_off + nw] = resized
    cv2.rectangle(canvas, (0, 0), (target_w - 1, target_h - 1), (220, 220, 220), 1)
    return canvas


def create_overview_sheet(
    overview_products: List[Dict],
    scene_list: List[str],
    primary_seed: int,
    out_dir: Path,
    out_file: Path,
) -> None:
    """Create an overview contact sheet for up to 6 products across scenes."""
    if not overview_products:
        return

    items = overview_products[:6]
    thumb_size = 224
    margin = 14
    header_h = 40
    caption_h = 24
    row_h = thumb_size + caption_h + margin

    columns = ["Original", "White BG"] + [s.replace("_", " ").title() for s in scene_list]
    num_cols = len(columns)

    sheet_w = margin + num_cols * (thumb_size + margin)
    sheet_h = header_h + len(items) * row_h + margin
    sheet = np.full((sheet_h, sheet_w, 3), 248, dtype=np.uint8)

    # Header bar
    cv2.rectangle(sheet, (0, 0), (sheet_w, header_h), (235, 235, 235), -1)
    cv2.line(sheet, (0, header_h), (sheet_w, header_h), (210, 210, 210), 1)

    for col_idx, col_name in enumerate(columns):
        x = margin + col_idx * (thumb_size + margin)
        cv2.putText(sheet, col_name, (x + 8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)

    for row_idx, item in enumerate(items):
        y_top = header_h + margin + row_idx * row_h
        stem = item["stem"]

        # 1. Original
        th_orig = make_thumb(item["img"], thumb_size, thumb_size)

        # 2. White BG
        white_path = out_dir / "white" / f"{stem}.jpg"
        if white_path.exists():
            white_img = cv2.imread(str(white_path))
            th_white = make_thumb(white_img, thumb_size, thumb_size) if white_img is not None else np.full((thumb_size, thumb_size, 3), 255, dtype=np.uint8)
        else:
            th_white = np.full((thumb_size, thumb_size, 3), 255, dtype=np.uint8)

        row_thumbs = [th_orig, th_white]

        # 3. Scene columns
        for s_name in scene_list:
            scene_p = out_dir / s_name / f"{stem}_s{primary_seed}.jpg"
            if scene_p.exists():
                s_img = cv2.imread(str(scene_p))
                th_scene = make_thumb(s_img, thumb_size, thumb_size) if s_img is not None else np.full((thumb_size, thumb_size, 3), 255, dtype=np.uint8)
            else:
                th_scene = np.full((thumb_size, thumb_size, 3), 255, dtype=np.uint8)
            row_thumbs.append(th_scene)

        # Place thumbnails
        for col_idx, th in enumerate(row_thumbs):
            x = margin + col_idx * (thumb_size + margin)
            sheet[y_top : y_top + thumb_size, x : x + thumb_size] = th

        # Caption
        cap_y = y_top + thumb_size + 17
        cv2.putText(sheet, f"Product #{row_idx + 1}: {item['filename']}", (margin + 6, cap_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
        sep_y = y_top + row_h - (margin // 2)
        cv2.line(sheet, (margin, sep_y), (sheet_w - margin, sep_y), (225, 225, 225), 1)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\nSaved scene overview contact sheet to: {out_file}")


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    originals_dir = Path(args.originals)

    if not originals_dir.exists():
        print(f"Error: Originals directory not found: {originals_dir}")
        sys.exit(1)

    # Resolve scenes
    scenes_data = load_scenes()
    all_scene_names = list(scenes_data["scenes"].keys())

    if args.scenes.strip().lower() == "all":
        scene_list = all_scene_names
    else:
        scene_list = [s.strip() for s in args.scenes.split(",") if s.strip()]
        for s in scene_list:
            if s not in all_scene_names:
                raise ValueError(f"Unknown scene '{s}'. Available: {all_scene_names}")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        seeds = [0]

    # Initialize generator
    gen_config = GenConfig(
        lora_path=args.lora,
        lora_scale=args.lora_scale,
    )

    if args.fake_generator:
        generator = FakeBackgroundGenerator(config=gen_config)
    else:
        generator = BackgroundGenerator(config=gen_config)

    # 1. Product selection via Coach gate
    coach_config = CoachConfig()
    exclude_file = Path("configs/coach_eval_exclude.txt")
    exclusions = load_exclusions(exclude_file)

    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp"}
    all_files = sorted([p for p in originals_dir.iterdir() if p.suffix.lower() in valid_exts], key=lambda p: p.name)

    passing_products = []
    gate_records = []
    skipped_count = 0

    print(f"Gating {len(all_files)} originals with Shoot Coach...")

    for img_p in all_files:
        # Check exclusion prefix
        if any(img_p.name.startswith(ex) for ex in exclusions):
            continue

        try:
            img = load_image(img_p)
            report, seg = assess_with_segmentation(img, config=coach_config)
        except Exception as e:
            gate_records.append({"filename": img_p.name, "ok": False, "tips": f"Error: {e}"})
            skipped_count += 1
            continue

        tips_str = " | ".join(report.tips) if report.tips else ""
        gate_records.append({
            "filename": img_p.name,
            "ok": report.ok,
            "tips": tips_str,
        })

        if report.ok:
            passing_products.append({
                "path": img_p,
                "filename": img_p.name,
                "stem": img_p.stem,
                "img": img,
                "report": report,
                "seg": seg,
            })
        else:
            skipped_count += 1

    # Save coach gate decisions
    gate_csv_path = out_dir / "coach_gate.csv"
    with open(gate_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "ok", "tips"])
        writer.writeheader()
        writer.writerows(gate_records)

    print(f"Coach gate complete: {len(passing_products)} passed, {skipped_count} skipped. Decisions saved to: {gate_csv_path}")

    # Deduplicate by family (keeping first of each family in sorted order)
    seen_families = set()
    family_products: List[Tuple[Dict, str]] = []
    for prod in passing_products:
        fam = extract_family(prod["filename"])
        if fam not in seen_families:
            seen_families.add(fam)
            family_products.append((prod, fam))

    print(f"Found {len(family_products)} unique product families among passing originals.")

    # Pick n-products evenly spaced
    chosen_items = pick_diverse_products(family_products, args.n_products)
    selected_products = [item[0] for item in chosen_items]

    # Print chosen filenames and family names before generating
    print(f"\nChosen {len(chosen_items)} products across diverse families:")
    for idx, (p_info, fam) in enumerate(chosen_items, start=1):
        print(f"  {idx}. {p_info['filename']}  (family: '{fam}')")

    # 2. Generation per product, scene, and seed
    white_dir = out_dir / "white"
    white_dir.mkdir(parents=True, exist_ok=True)
    for s_name in scene_list:
        (out_dir / s_name).mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    manifest_exists = manifest_path.exists()
    manifest_file = open(manifest_path, "a", newline="", encoding="utf-8")
    manifest_writer = csv.DictWriter(
        manifest_file,
        fieldnames=["product", "scene", "seed", "prompt", "seconds", "peak_vram_mb"],
    )
    if not manifest_exists:
        manifest_writer.writeheader()

    generated_times = []
    total_images_generated = 0
    canvas_w, canvas_h = _resolve_preset(args.preset)

    for p_idx, prod in enumerate(selected_products, start=1):
        stem = prod["stem"]
        seg = prod["seg"]
        print(f"\n[{p_idx}/{len(selected_products)}] Processing {prod['filename']}...")

        # A. Save white-background version
        white_path = white_dir / f"{stem}.jpg"
        if not white_path.exists():
            canvas_rgba, canvas_mask = place_product(seg.rgba, preset=args.preset)
            bg_white = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
            white_bgr = composite(bg_white, canvas_rgba, canvas_mask, shadow=True, opacity=0.30)
            cv2.imwrite(str(white_path), white_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # B. Generate scenes
        for scene_name in scene_list:
            for seed in seeds:
                scene_out_p = out_dir / scene_name / f"{stem}_s{seed}.jpg"
                if scene_out_p.exists():
                    print(f"  Skipping {scene_name} seed {seed} (already exists: {scene_out_p.name})")
                    continue

                t0 = time.time()
                res = run_pipeline(
                    image_path=prod["path"],
                    scene=scene_name,
                    marketplace=args.preset,
                    force=True,
                    seed=seed,
                    generator=generator,
                    gen_config=gen_config,
                )
                elapsed = time.time() - t0

                if res.image is not None:
                    cv2.imwrite(str(scene_out_p), res.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    total_images_generated += 1
                    generated_times.append(elapsed)

                manifest_row = {
                    "product": stem,
                    "scene": scene_name,
                    "seed": seed,
                    "prompt": res.prompt,
                    "seconds": round(elapsed, 3),
                    "peak_vram_mb": round(generator.peak_vram_mb, 2),
                }
                manifest_writer.writerow(manifest_row)
                manifest_file.flush()
                print(f"  Generated {scene_name} [s={seed}] in {elapsed:.2f}s -> {scene_out_p.name}")

    manifest_file.close()

    # 3. Write overview contact sheet
    create_overview_sheet(
        overview_products=selected_products,
        scene_list=scene_list,
        primary_seed=seeds[0],
        out_dir=out_dir,
        out_file=out_dir / "overview.jpg",
    )

    # 4. Summary Output
    mean_sec = float(np.mean(generated_times)) if generated_times else 0.0
    print("\n" + "=" * 60)
    print("Scene Generation Run Summary")
    print("=" * 60)
    print(f"Products selected: {len(selected_products)}")
    print(f"Images generated: {total_images_generated}")
    print(f"Mean seconds per generated image: {mean_sec:.2f}s")
    print(f"Peak VRAM: {generator.peak_vram_mb:.1f} MB")
    print(f"Loaded base model ID: {getattr(generator, 'loaded_model_id', 'N/A')}")
    print(f"Images skipped by Coach gate: {skipped_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
