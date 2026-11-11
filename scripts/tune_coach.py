import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from studio.coach.config import CoachConfig
from studio.coach.io import load_image
from studio.coach.mask_cache import get_mask


def compute_all_candidates(img: np.ndarray, mask: Optional[np.ndarray], config: CoachConfig) -> Dict[str, float]:
    """Compute all feature candidates for Blur, Dark, Bright, and Clutter checks."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 and img.shape[2] == 3 else img
    total_px = h * w

    # --- 1. Blur Candidates (1024 longest side) ---
    longest = max(h, w)
    if longest != config.blur_resize_side and longest > 0:
        scale = config.blur_resize_side / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        gray_1024 = cv2.resize(gray, (new_w, new_h), interpolation=interp)
    else:
        gray_1024 = gray

    row_chunks = np.array_split(gray_1024, 8, axis=0)
    scores_tile_p90 = []
    scores_norm_p90 = []
    scores_scale_ratio = []

    for r_chunk in row_chunks:
        tiles = np.array_split(r_chunk, 8, axis=1)
        for tile in tiles:
            if tile.size == 0:
                continue
            lap_var = float(cv2.Laplacian(tile, cv2.CV_64F).var())
            lum_var = float(np.var(tile))
            th, tw = tile.shape[:2]
            tile_down = cv2.resize(tile, (max(1, tw // 2), max(1, th // 2)), interpolation=cv2.INTER_AREA)
            lap_down_var = float(cv2.Laplacian(tile_down, cv2.CV_64F).var())

            scores_tile_p90.append(lap_var)
            scores_norm_p90.append(lap_var / (lum_var + 1.0))
            scores_scale_ratio.append(lap_var / (lap_down_var + 1e-6))

    blur_tile_p90 = float(np.percentile(scores_tile_p90, 90)) if scores_tile_p90 else 0.0
    blur_norm_p90 = float(np.percentile(scores_norm_p90, 90)) if scores_norm_p90 else 0.0
    blur_scale_ratio = float(np.percentile(scores_scale_ratio, 90)) if scores_scale_ratio else 0.0

    # --- 2. Exposure Candidates (Dark & Bright) ---
    has_valid_mask = (
        mask is not None
        and isinstance(mask, np.ndarray)
        and mask.size > 0
        and len(mask.shape) >= 2
        and (int(np.count_nonzero(mask > 0)) / total_px) >= 0.02
    )

    global_p95 = float(np.percentile(gray, 95))
    global_p5 = float(np.percentile(gray, 5))

    if has_valid_mask:
        prod_pixels = gray[mask > 0]
        product_p95 = float(np.percentile(prod_pixels, 95))
        product_p50 = float(np.percentile(prod_pixels, 50))
        product_p5 = float(np.percentile(prod_pixels, 5))
        product_clip_hi = float(np.count_nonzero(prod_pixels >= 250) / prod_pixels.size)
    else:
        # Fallback to whole-image equivalents
        product_p95 = global_p95
        product_p50 = float(np.percentile(gray, 50))
        product_p5 = global_p5
        product_clip_hi = float(np.count_nonzero(gray >= 250) / gray.size)

    # --- 3. Clutter Candidates (512 longest side + GaussianBlur 5x5 + Canny) ---
    if longest != 512 and longest > 0:
        scale_512 = 512.0 / float(longest)
        w512 = max(1, int(round(w * scale_512)))
        h512 = max(1, int(round(h * scale_512)))
        interp = cv2.INTER_AREA if scale_512 < 1.0 else cv2.INTER_LINEAR
        gray_512 = cv2.resize(gray, (w512, h512), interpolation=interp)
    else:
        gray_512 = gray
        w512, h512 = w, h

    blurred_512 = cv2.GaussianBlur(gray_512, (5, 5), 0)
    edges = cv2.Canny(blurred_512, 50, 150)

    # Global border (outer 15% band)
    band_h = max(1, int(round(h512 * config.border_band_fraction)))
    band_w = max(1, int(round(w512 * config.border_band_fraction)))
    border_mask = np.ones((h512, w512), dtype=bool)
    if h512 > 2 * band_h and w512 > 2 * band_w:
        border_mask[band_h : h512 - band_h, band_w : w512 - band_w] = False

    tot_border = int(np.count_nonzero(border_mask))
    global_border = float(np.count_nonzero((edges > 0) & border_mask) / tot_border) if tot_border > 0 else 0.0

    # Background-based candidates (pixels outside dilated mask)
    if has_valid_mask:
        mask_512 = cv2.resize(mask, (w512, h512), interpolation=cv2.INTER_NEAREST)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        dilated = cv2.dilate((mask_512 > 0).astype(np.uint8), kernel)
        bg_mask = (dilated == 0)
        bg_px = int(np.count_nonzero(bg_mask))

        if bg_px / (h512 * w512) < 0.10:
            # Under 10% background: treat as clean
            bg_edge_density = 0.0
            bg_lum_std = 0.0
        else:
            bg_edge_density = float(np.count_nonzero((edges > 0) & bg_mask) / bg_px)
            bg_lum_std = float(np.std(blurred_512[bg_mask]))
    else:
        bg_edge_density = global_border
        bg_lum_std = float(np.std(blurred_512[border_mask])) if tot_border > 0 else 0.0

    return {
        "blur_tile_p90": blur_tile_p90,
        "blur_norm_p90": blur_norm_p90,
        "blur_scale_ratio": blur_scale_ratio,
        "global_p95": global_p95,
        "product_p95": product_p95,
        "product_p50": product_p50,
        "global_p5": global_p5,
        "product_p5": product_p5,
        "product_clip_hi": product_clip_hi,
        "global_border": global_border,
        "bg_edge_density": bg_edge_density,
        "bg_lum_std": bg_lum_std,
    }


def find_lower_threshold(
    good_vals: List[float],
    bad_vals: List[float],
    max_fpr: float = 0.05,
) -> Tuple[float, float, float]:
    """Find threshold T for a check that fails when value < T, maximizing recall subject to FPR <= max_fpr."""
    n_good = len(good_vals)
    n_bad = len(bad_vals)
    if n_good == 0 or n_bad == 0:
        return 0.0, 0.0, 0.0

    candidates = sorted(list(set(good_vals + bad_vals)))
    best_t = 0.0
    best_recall = -1.0
    best_fpr = 1.0

    for cand in candidates:
        fpr = sum(1 for v in good_vals if v < cand) / n_good
        if fpr <= max_fpr:
            recall = sum(1 for v in bad_vals if v < cand) / n_bad
            if recall > best_recall or (recall == best_recall and cand > best_t):
                best_recall = recall
                best_t = cand
                best_fpr = fpr

    return best_t, best_fpr, best_recall


def find_upper_threshold(
    good_vals: List[float],
    bad_vals: List[float],
    max_fpr: float = 0.05,
) -> Tuple[float, float, float]:
    """Find threshold T for a check that fails when value > T, maximizing recall subject to FPR <= max_fpr."""
    n_good = len(good_vals)
    n_bad = len(bad_vals)
    if n_good == 0 or n_bad == 0:
        return 0.0, 0.0, 0.0

    candidates = sorted(list(set(good_vals + bad_vals)))
    best_t = float("inf")
    best_recall = -1.0
    best_fpr = 1.0

    for cand in candidates:
        fpr = sum(1 for v in good_vals if v > cand) / n_good
        if fpr <= max_fpr:
            recall = sum(1 for v in bad_vals if v > cand) / n_bad
            if recall > best_recall or (recall == best_recall and cand < best_t):
                best_recall = recall
                best_t = cand
                best_fpr = fpr

    return best_t, best_fpr, best_recall


def main() -> None:
    labels_csv = Path("data/coach_eval/labels.csv")
    images_dir = Path("data/coach_eval/images")
    output_cfg_path = Path("configs/coach_thresholds.json")

    if not labels_csv.exists():
        print(f"Error: {labels_csv} not found. Please run make_coach_eval.py first.")
        sys.exit(1)

    with open(labels_csv, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    tune_rows = [r for r in all_rows if r.get("split") == "tune"]
    if not tune_rows:
        print("No rows found with split=tune in labels.csv.")
        sys.exit(1)

    print(f"Extracting features for {len(tune_rows)} images in tune split using mask cache...")
    cfg_base = CoachConfig(load_saved_thresholds=False)

    samples = []
    for row in tune_rows:
        fname = row["filename"]
        img_p = images_dir / fname
        if not img_p.exists():
            continue
        img = load_image(img_p)
        mask = get_mask(img_p, key=fname)
        feats = compute_all_candidates(img, mask, cfg_base)
        sample = {
            "filename": fname,
            "label": row["label"],
            "reason": row["reason"],
            "source": row["source"],
            **feats,
        }
        samples.append(sample)

    good_samples = [s for s in samples if s["label"] == "good"]
    bad_blur = [s for s in samples if s["label"] == "bad" and s["reason"] == "blur"]
    bad_dark = [s for s in samples if s["label"] == "bad" and s["reason"] == "dark"]
    bad_bright = [s for s in samples if s["label"] == "bad" and s["reason"] == "bright"]
    bad_clutter = [s for s in samples if s["label"] == "bad" and s["reason"] == "clutter"]

    print(f"Tune split: {len(good_samples)} good, {len(bad_blur)} bad-blur, {len(bad_dark)} bad-dark, {len(bad_bright)} bad-bright, {len(bad_clutter)} bad-clutter\n")

    # Define all candidate checks: (check_category, candidate_name, feat_key, direction, finder_fn)
    checks = {
        "Blur": [
            ("tile_p90", "blur_tile_p90", "lower", bad_blur),
            ("tile_norm_p90", "blur_norm_p90", "lower", bad_blur),
            ("scale_ratio", "blur_scale_ratio", "lower", bad_blur),
        ],
        "Dark": [
            ("global_p95", "global_p95", "lower", bad_dark),
            ("product_p95", "product_p95", "lower", bad_dark),
            ("product_p50", "product_p50", "lower", bad_dark),
        ],
        "Bright": [
            ("global_p5", "global_p5", "upper", bad_bright),
            ("product_p5", "product_p5", "upper", bad_bright),
            ("product_clip_hi", "product_clip_hi", "upper", bad_bright),
        ],
        "Clutter": [
            ("global_border", "global_border", "upper", bad_clutter),
            ("bg_edge_density", "bg_edge_density", "upper", bad_clutter),
            ("bg_lum_std", "bg_lum_std", "upper", bad_clutter),
        ],
    }

    eval_results = {}
    chosen = {}

    for cat, candidates in checks.items():
        eval_results[cat] = []
        for name, key, direction, bad_group in candidates:
            g_vals = [s[key] for s in good_samples]
            b_vals = [s[key] for s in bad_group]
            if direction == "lower":
                t, fpr, rec = find_lower_threshold(g_vals, b_vals, max_fpr=0.05)
            else:
                t, fpr, rec = find_upper_threshold(g_vals, b_vals, max_fpr=0.05)
            eval_results[cat].append({
                "name": name,
                "key": key,
                "direction": direction,
                "threshold": t,
                "fpr": fpr,
                "recall": rec,
            })

        # Pick candidate with highest recall (tie-break on lowest fpr)
        best = max(eval_results[cat], key=lambda r: (r["recall"], -r["fpr"]))
        chosen[cat] = best

    # Print comprehensive tuning table
    print("=" * 86)
    print("Shoot Coach Parameter Tuning (Phase 2b - Full Candidate Evaluation, Split=Tune)")
    print("=" * 86)
    print(f"{'Category':<10} | {'Candidate Feature':<18} | {'Direction':<9} | {'Threshold':>10} | {'FPR':>7} | {'Recall':>7} | {'Chosen':^6}")
    print("-" * 86)

    for cat, rows in eval_results.items():
        for r in rows:
            is_best = (r["name"] == chosen[cat]["name"])
            chosen_mark = "  *   " if is_best else "      "
            dir_str = "value < T" if r["direction"] == "lower" else "value > T"
            thresh_fmt = f"{r['threshold']:.4f}" if "density" in r["name"] or "ratio" in r["name"] or "clip" in r["name"] else f"{r['threshold']:.2f}"
            print(f"{cat:<10} | {r['name']:<18} | {dir_str:<9} | {thresh_fmt:>10} | {r['fpr']*100:>6.1f}% | {r['recall']*100:>6.1f}% | {chosen_mark}")
        print("-" * 86)

    # Save to configs/coach_thresholds.json
    best_blur = chosen["Blur"]
    best_dark = chosen["Dark"]
    best_bright = chosen["Bright"]
    best_clutter = chosen["Clutter"]

    global_dark = next(r for r in eval_results["Dark"] if r["name"] == "global_p95")
    global_bright = next(r for r in eval_results["Bright"] if r["name"] == "global_p5")
    global_clutter = next(r for r in eval_results["Clutter"] if r["name"] == "global_border")

    thresholds_dict = {
        "blur_method": best_blur["name"],
        "blur_min": round(best_blur["threshold"], 2),
        "dark_feature": best_dark["name"],
        "dark_threshold": round(best_dark["threshold"], 2),
        "dark_max_p95": round(global_dark["threshold"], 2),
        "bright_feature": best_bright["name"],
        "bright_threshold": round(best_bright["threshold"], 4),
        "bright_min_p5": round(global_bright["threshold"], 2),
        "clutter_feature": best_clutter["name"],
        "clutter_threshold": round(best_clutter["threshold"], 4),
        "clutter_max": round(global_clutter["threshold"], 4),
    }

    output_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_cfg_path, "w", encoding="utf-8") as f:
        json.dump(thresholds_dict, f, indent=4)

    print(f"\nChosen candidate parameters saved to: {output_cfg_path}")


if __name__ == "__main__":
    main()
