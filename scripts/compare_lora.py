#!/usr/bin/env python3
"""Paired Base vs LoRA background generation comparison and metrics evaluation.

Generates outputs with and without LoRA on Coach-gated diverse products,
evaluates CLIP alignment to scene descriptions, person detector retries,
and verifies scale=0.0 equivalence.
"""

import argparse
import csv
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import cv2
import numpy as np
from PIL import Image
import yaml

from scripts.run_scenes import make_thumb, select_diverse_products
from studio.coach.config import CoachConfig
from studio.coach.io import load_image
from studio.generate.backgrounds import BackgroundGenerator, FakeBackgroundGenerator
from studio.generate.config import GenConfig, effective_lora_scale, load_lora_scales
from studio.pipeline import run_pipeline


_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_CLIP_DEVICE = None


def get_clip_scorer():
    """Lazily load CLIP model and processor on CUDA if available, else CPU."""
    global _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_DEVICE
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_DEVICE

    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name, use_safetensors=False).to(device)
        model.eval()
        processor = CLIPProcessor.from_pretrained(model_name)

        _CLIP_MODEL = model
        _CLIP_PROCESSOR = processor
        _CLIP_DEVICE = device
        return _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_DEVICE
    except Exception as e:
        print(f"Warning: Could not initialize CLIP scorer ({e}). CLIP scores will be 0.0.", file=sys.stderr)
        return None, None, "cpu"


def compute_clip_score(image_bgr: np.ndarray, text_description: str) -> float:
    """Compute cosine similarity between image and scene description using CLIP."""
    model, processor, device = get_clip_scorer()
    if model is None or processor is None:
        return 0.0

    try:
        import torch

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        inputs = processor(text=[text_description], images=[pil_img], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Normalize image & text features
            img_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            similarity = torch.sum(img_embeds * txt_embeds, dim=-1).item()
            return float(similarity)
    except Exception as e:
        print(f"Warning: Error calculating CLIP similarity: {e}", file=sys.stderr)
        return 0.0


def load_scene_descriptions(path: str = "configs/lora_themes.yaml") -> Dict[str, str]:
    """Load theme description mapping from configs/lora_themes.yaml."""
    p = Path(path)
    if not p.exists():
        repo_p = _REPO_ROOT / "configs" / "lora_themes.yaml"
        if repo_p.exists():
            p = repo_p
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {k: v.get("description", "") for k, v in data.items() if isinstance(v, dict)}


def format_metrics_table(
    metrics_by_scene: Dict[str, Dict[str, Dict[str, Any]]],
    sanity_checks: Optional[Dict[str, float]] = None,
) -> str:
    """Format evaluation metrics table per scene, base vs LoRA.

    Args:
        metrics_by_scene: Nested dict [scene][variant] -> {clip_score, attempts, person_flag, count, seconds}
        sanity_checks: Mapping of scale 0.0 scenes to mean absolute pixel differences.

    Returns:
        Formatted multi-line text table string.
    """
    lines = []
    lines.append("=" * 96)
    lines.append(f"{'Scene':<18} {'Variant':<8} {'CLIP Score':<12} {'Mean Attempts':<15} {'Person Flagged':<16} {'Mean Seconds':<12}")
    lines.append("-" * 96)

    for scene, variants in sorted(metrics_by_scene.items()):
        for v_idx, variant in enumerate(["base", "lora"]):
            m = variants.get(variant, {})
            clip_val = f"{m.get('clip_score', 0.0):.3f}"
            attempts_val = f"{m.get('attempts', 1.0):.2f}"
            flagged_val = f"{m.get('flagged', 0)}/{m.get('count', 0)}"
            sec_val = f"{m.get('seconds', 0.0):.2f}s"
            scene_label = scene if v_idx == 0 else ""
            lines.append(f"{scene_label:<18} {variant:<8} {clip_val:<12} {attempts_val:<15} {flagged_val:<16} {sec_val:<12}")
        lines.append("-" * 96)

    lines.append("=" * 96)

    if sanity_checks:
        lines.append("Sanity check (scale 0.0 scenes):")
        for sc_name, diff_val in sorted(sanity_checks.items()):
            lines.append(f"  {sc_name}: mean absolute pixel difference = {diff_val:.4f}")

    lines.append("Note: CLIP score measures text alignment only; it does not measure authenticity.")
    return "\n".join(lines)


def create_comparison_sheet(
    row_items: List[Dict[str, Any]],
    seeds: List[int],
    out_file: Path,
    thumb_size: int = 200,
) -> None:
    """Create comparison contact sheet with up to 12 product x scene rows.

    Columns: original | base s0 | LoRA s0 | base s1 | LoRA s1
    """
    n_rows = len(row_items)
    if n_rows == 0:
        return

    s0, s1 = seeds[0], seeds[1]
    col_headers = ["Original", f"Base s={s0}", f"LoRA s={s0}", f"Base s={s1}", f"LoRA s={s1}"]
    n_cols = len(col_headers)

    header_h = 40
    label_w = 150
    margin = 8
    gap = 4

    total_w = label_w + n_cols * thumb_size + (n_cols - 1) * gap + margin * 2
    total_h = header_h + n_rows * thumb_size + (n_rows - 1) * gap + margin * 2

    canvas = np.full((total_h, total_w, 3), 245, dtype=np.uint8)

    # 1. Header labels
    for c_idx, title in enumerate(col_headers):
        x_left = label_w + margin + c_idx * (thumb_size + gap)
        cv2.putText(
            canvas,
            title,
            (x_left + 15, margin + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )

    # 2. Rows
    for r_idx, item in enumerate(row_items):
        y_top = margin + header_h + r_idx * (thumb_size + gap)

        # Label: product and scene
        prod_stem = item["product"][:12]
        scene_name = item["scene"]
        cv2.putText(
            canvas,
            prod_stem,
            (margin + 4, y_top + thumb_size // 2 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"[{scene_name}]",
            (margin + 4, y_top + thumb_size // 2 + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )

        # Column images: orig, base s0, lora s0, base s1, lora s1
        imgs = [
            item.get("orig"),
            item.get(f"base_s{s0}"),
            item.get(f"lora_s{s0}"),
            item.get(f"base_s{s1}"),
            item.get(f"lora_s{s1}"),
        ]

        for c_idx, img in enumerate(imgs):
            x_left = label_w + margin + c_idx * (thumb_size + gap)
            if img is not None:
                thumb = make_thumb(img, target_w=thumb_size, target_h=thumb_size)
                canvas[y_top : y_top + thumb_size, x_left : x_left + thumb_size] = thumb
            cv2.rectangle(canvas, (x_left, y_top), (x_left + thumb_size - 1, y_top + thumb_size - 1), (200, 200, 200), 1)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Comparison sheet saved to: {out_file}")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired Base vs LoRA evaluation script.")
    parser.add_argument("--lora", type=str, default="models/festive_lora", help="Path to LoRA weights")
    parser.add_argument("--n-products", type=int, default=6, help="Number of products to process")
    parser.add_argument(
        "--scenes",
        type=str,
        default="diwali,raksha_bandhan,holi,wedding,indian_home",
        help="Comma-separated scene names",
    )
    parser.add_argument("--seeds", type=str, default="0,1", help="Comma-separated seeds (e.g. 0,1)")
    parser.add_argument("--out", type=str, default="outputs/compare", help="Output directory")
    parser.add_argument("--originals", type=str, default="data/coach_eval/originals/good", help="Directory of good original photos")
    parser.add_argument("--preset", type=str, default="square_1000", help="Marketplace canvas preset")
    parser.add_argument("--fake-generator", action="store_true", help="Use FakeBackgroundGenerator for CPU testing")
    return parser.parse_args(args)


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) < 2:
        seeds = [0, 1]

    # 1. Product Selection
    originals_dir = Path(args.originals)
    selected_products, _ = select_diverse_products(
        originals_dir=originals_dir,
        n_products=args.n_products,
        coach_config=CoachConfig(),
        exclude_file=Path("configs/coach_eval_exclude.txt"),
        gate_csv_path=out_dir / "coach_gate.csv",
    )

    # 2. Build Generators
    # Base generator has no LoRA; LoRA generator has lora_path set
    base_cfg = GenConfig(lora_path=None)
    lora_cfg = GenConfig(lora_path=args.lora)

    if args.fake_generator:
        base_gen = FakeBackgroundGenerator(config=base_cfg)
        lora_gen = FakeBackgroundGenerator(config=lora_cfg)
    else:
        try:
            import torch

            if torch.cuda.is_available():
                base_gen = BackgroundGenerator(config=base_cfg)
                lora_gen = BackgroundGenerator(config=lora_cfg)
            else:
                print("No GPU available, falling back to FakeBackgroundGenerator.")
                base_gen = FakeBackgroundGenerator(config=base_cfg)
                lora_gen = FakeBackgroundGenerator(config=lora_cfg)
        except Exception:
            base_gen = FakeBackgroundGenerator(config=base_cfg)
            lora_gen = FakeBackgroundGenerator(config=lora_cfg)

    # Prepare directories
    base_dir = out_dir / "base"
    lora_dir = out_dir / "lora"
    for sc in scenes:
        (base_dir / sc).mkdir(parents=True, exist_ok=True)
        (lora_dir / sc).mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    manifest_exists = manifest_path.exists()
    manifest_file = open(manifest_path, "a", newline="", encoding="utf-8")
    manifest_writer = csv.DictWriter(
        manifest_file,
        fieldnames=[
            "product",
            "scene",
            "seed",
            "variant",
            "prompt",
            "scale",
            "attempts",
            "person_flag",
            "seconds",
        ],
    )
    if not manifest_exists:
        manifest_writer.writeheader()

    records: List[Dict[str, Any]] = []
    cached_images: Dict[str, np.ndarray] = {}

    scene_descriptions = load_scene_descriptions()

    print(f"\nRunning paired generation for {len(selected_products)} products x {len(scenes)} scenes x {len(seeds)} seeds...")

    for prod_idx, prod in enumerate(selected_products, start=1):
        stem = prod["stem"]
        print(f"\n[{prod_idx}/{len(selected_products)}] Product: {prod['filename']}")

        for scene in scenes:
            for seed in seeds:
                base_img_path = base_dir / scene / f"{stem}_s{seed}.jpg"
                lora_img_path = lora_dir / scene / f"{stem}_s{seed}.jpg"

                # 1. Base generation
                if base_img_path.exists():
                    base_img = cv2.imread(str(base_img_path))
                    base_rec = {
                        "product": stem,
                        "scene": scene,
                        "seed": seed,
                        "variant": "base",
                        "prompt": "",
                        "scale": 0.0,
                        "attempts": 1,
                        "person_flag": False,
                        "seconds": 0.0,
                    }
                else:
                    t0 = time.time()
                    res_base = run_pipeline(
                        image_path=prod["path"],
                        scene=scene,
                        marketplace=args.preset,
                        force=True,
                        seed=seed,
                        generator=base_gen,
                        gen_config=base_cfg,
                        lora_scale=0.0,
                    )
                    sec_base = round(time.time() - t0, 3)
                    base_img = res_base.image
                    if base_img is not None:
                        cv2.imwrite(str(base_img_path), base_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

                    base_rec = {
                        "product": stem,
                        "scene": scene,
                        "seed": seed,
                        "variant": "base",
                        "prompt": res_base.prompt,
                        "scale": 0.0,
                        "attempts": res_base.attempts,
                        "person_flag": res_base.person_flag,
                        "seconds": sec_base,
                    }
                    manifest_writer.writerow(base_rec)
                    manifest_file.flush()

                # 2. LoRA generation
                if lora_img_path.exists():
                    lora_img = cv2.imread(str(lora_img_path))
                    eff_scale = effective_lora_scale(scene)
                    lora_rec = {
                        "product": stem,
                        "scene": scene,
                        "seed": seed,
                        "variant": "lora",
                        "prompt": "",
                        "scale": eff_scale,
                        "attempts": 1,
                        "person_flag": False,
                        "seconds": 0.0,
                    }
                else:
                    t0 = time.time()
                    res_lora = run_pipeline(
                        image_path=prod["path"],
                        scene=scene,
                        marketplace=args.preset,
                        force=True,
                        seed=seed,
                        generator=lora_gen,
                        gen_config=lora_cfg,
                    )
                    sec_lora = round(time.time() - t0, 3)
                    lora_img = res_lora.image
                    if lora_img is not None:
                        cv2.imwrite(str(lora_img_path), lora_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

                    lora_rec = {
                        "product": stem,
                        "scene": scene,
                        "seed": seed,
                        "variant": "lora",
                        "prompt": res_lora.prompt,
                        "scale": res_lora.scale,
                        "attempts": res_lora.attempts,
                        "person_flag": res_lora.person_flag,
                        "seconds": sec_lora,
                    }
                    manifest_writer.writerow(lora_rec)
                    manifest_file.flush()

                cached_images[f"base_{scene}_{stem}_s{seed}"] = base_img
                cached_images[f"lora_{scene}_{stem}_s{seed}"] = lora_img

                records.append(base_rec)
                records.append(lora_rec)

    manifest_file.close()

    # 3. Calculate Metrics (CLIP similarity, attempts, person flags, seconds)
    metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}
    sanity_diffs: Dict[str, List[float]] = {}

    for sc in scenes:
        metrics[sc] = {
            "base": {"clip_scores": [], "attempts": [], "flagged": 0, "count": 0, "seconds": []},
            "lora": {"clip_scores": [], "attempts": [], "flagged": 0, "count": 0, "seconds": []},
        }

    for prod in selected_products:
        stem = prod["stem"]
        for sc in scenes:
            desc = scene_descriptions.get(sc, "")
            sc_scale = effective_lora_scale(sc)

            for seed in seeds:
                b_img = cached_images.get(f"base_{sc}_{stem}_s{seed}")
                l_img = cached_images.get(f"lora_{sc}_{stem}_s{seed}")

                # CLIP scores
                if b_img is not None and desc:
                    b_clip = compute_clip_score(b_img, desc)
                    metrics[sc]["base"]["clip_scores"].append(b_clip)
                if l_img is not None and desc:
                    l_clip = compute_clip_score(l_img, desc)
                    metrics[sc]["lora"]["clip_scores"].append(l_clip)

                # Scale 0.0 sanity check: pixel diff
                if sc_scale == 0.0 and b_img is not None and l_img is not None:
                    if b_img.shape == l_img.shape:
                        diff = float(np.mean(np.abs(b_img.astype(np.float32) - l_img.astype(np.float32))))
                        sanity_diffs.setdefault(sc, []).append(diff)

    # Aggregate record stats
    for r in records:
        sc = r["scene"]
        var = r["variant"]
        if sc in metrics and var in metrics[sc]:
            m = metrics[sc][var]
            m["attempts"].append(r["attempts"])
            if r["person_flag"]:
                m["flagged"] += 1
            m["count"] += 1
            if r["seconds"] > 0:
                m["seconds"].append(r["seconds"])

    summary_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for sc, v_dict in metrics.items():
        summary_metrics[sc] = {}
        for var, d in v_dict.items():
            mean_clip = float(np.mean(d["clip_scores"])) if d["clip_scores"] else 0.0
            mean_att = float(np.mean(d["attempts"])) if d["attempts"] else 1.0
            mean_sec = float(np.mean(d["seconds"])) if d["seconds"] else 0.0
            summary_metrics[sc][var] = {
                "clip_score": mean_clip,
                "attempts": mean_att,
                "flagged": d["flagged"],
                "count": d["count"],
                "seconds": mean_sec,
            }

    sanity_summary = {sc: float(np.mean(v)) for sc, v in sanity_diffs.items()}

    # Print Table
    table_str = format_metrics_table(summary_metrics, sanity_summary)
    print("\n" + table_str)

    # 4. Generate Comparison Sheet (up to 12 rows, covering every scene, >= 2 rows each)
    row_items = []
    # Pick products for each scene: 2 per scene across 5 scenes = 10 rows (or up to 12)
    s0, s1 = seeds[0], seeds[1]
    for sc in scenes:
        # Take first 2 products (or more up to 12 total)
        for prod in selected_products:
            if len(row_items) >= 12:
                break
            count_sc = sum(1 for item in row_items if item["scene"] == sc)
            if count_sc >= 2 and len(row_items) + (len(scenes) - scenes.index(sc) - 1) * 2 >= 12:
                continue

            stem = prod["stem"]
            orig_img = prod["img"]
            b_s0 = cached_images.get(f"base_{sc}_{stem}_s{s0}")
            l_s0 = cached_images.get(f"lora_{sc}_{stem}_s{s0}")
            b_s1 = cached_images.get(f"base_{sc}_{stem}_s{s1}")
            l_s1 = cached_images.get(f"lora_{sc}_{stem}_s{s1}")

            row_items.append({
                "product": stem,
                "scene": sc,
                "orig": orig_img,
                f"base_s{s0}": b_s0,
                f"lora_s{s0}": l_s0,
                f"base_s{s1}": b_s1,
                f"lora_s{s1}": l_s1,
            })
            if count_sc + 1 >= 2 and len(row_items) >= 10:
                break

    create_comparison_sheet(
        row_items=row_items,
        seeds=seeds,
        out_file=out_dir / "compare.jpg",
        thumb_size=200,
    )


if __name__ == "__main__":
    main()
