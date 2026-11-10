import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import cv2
import numpy as np

from studio.coach.config import CoachConfig
from studio.coach.io import load_image


def compute_features(img: np.ndarray, config: CoachConfig) -> Dict[str, float]:
    """Compute raw feature values for Shoot Coach Phase 1d checks."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 and img.shape[2] == 3 else img

    # --- 1. Blur Features at 1024 longest side ---
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

    # --- 2. Exposure Features (Percentiles) ---
    p95 = float(np.percentile(gray, 95))
    p5 = float(np.percentile(gray, 5))

    # --- 3. Clutter Feature at 512 longest side + GaussianBlur 5x5 ---
    if longest != 512 and longest > 0:
        scale_512 = 512.0 / float(longest)
        w512 = max(1, int(round(w * scale_512)))
        h512 = max(1, int(round(h * scale_512)))
        interp = cv2.INTER_AREA if scale_512 < 1.0 else cv2.INTER_LINEAR
        gray_512 = cv2.resize(gray, (w512, h512), interpolation=interp)
    else:
        gray_512 = gray

    h512, w512 = gray_512.shape[:2]
    blurred_512 = cv2.GaussianBlur(gray_512, (5, 5), 0)
    edges = cv2.Canny(blurred_512, 50, 150)

    band_h = max(1, int(round(h512 * config.border_band_fraction)))
    band_w = max(1, int(round(w512 * config.border_band_fraction)))
    border_mask = np.ones((h512, w512), dtype=bool)
    if h512 > 2 * band_h and w512 > 2 * band_w:
        border_mask[band_h : h512 - band_h, band_w : w512 - band_w] = False

    tot_border = int(np.count_nonzero(border_mask))
    clutter_density = float(np.count_nonzero((edges > 0) & border_mask) / tot_border) if tot_border > 0 else 0.0

    return {
        "blur_tile_p90": blur_tile_p90,
        "blur_norm_p90": blur_norm_p90,
        "blur_scale_ratio": blur_scale_ratio,
        "lum_p95": p95,
        "lum_p5": p5,
        "clutter_density": clutter_density,
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


def main():
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

    print(f"Extracting features for {len(tune_rows)} images in tune split...")
    cfg_base = CoachConfig(load_saved_thresholds=False)

    samples = []
    for row in tune_rows:
        fname = row["filename"]
        img_p = images_dir / fname
        if not img_p.exists():
            continue
        img = load_image(img_p)
        feats = compute_features(img, cfg_base)
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

    print(f"Tune set: {len(good_samples)} good, {len(bad_blur)} bad-blur, {len(bad_dark)} bad-dark, {len(bad_bright)} bad-bright, {len(bad_clutter)} bad-clutter\n")

    # 1. Blur Method & Threshold Selection
    blur_candidates = [
        ("tile_p90", "blur_tile_p90"),
        ("tile_norm_p90", "blur_norm_p90"),
        ("scale_ratio", "blur_scale_ratio"),
    ]
    blur_results = {}
    for m_name, feat_key in blur_candidates:
        g_vals = [s[feat_key] for s in good_samples]
        b_vals = [s[feat_key] for s in bad_blur]
        t, fpr, rec = find_lower_threshold(g_vals, b_vals, max_fpr=0.05)
        blur_results[m_name] = (t, fpr, rec)

    best_blur_method = max(blur_results.keys(), key=lambda k: (blur_results[k][2], -blur_results[k][1]))
    best_blur_t, best_blur_fpr, best_blur_rec = blur_results[best_blur_method]

    # 2. Dark threshold (lum_p95 < T)
    g_p95 = [s["lum_p95"] for s in good_samples]
    b_dark = [s["lum_p95"] for s in bad_dark]
    dark_t, dark_fpr, dark_rec = find_lower_threshold(g_p95, b_dark, max_fpr=0.05)

    # 3. Bright threshold (lum_p5 > T)
    g_p5 = [s["lum_p5"] for s in good_samples]
    b_bright = [s["lum_p5"] for s in bad_bright]
    bright_t, bright_fpr, bright_rec = find_upper_threshold(g_p5, b_bright, max_fpr=0.05)

    # 4. Clutter threshold (clutter_density > T)
    g_clutter = [s["clutter_density"] for s in good_samples]
    b_clutter = [s["clutter_density"] for s in bad_clutter]
    clutter_t, clutter_fpr, clutter_rec = find_upper_threshold(g_clutter, b_clutter, max_fpr=0.05)

    # Output table
    table_rows = [
        (f"blur_min ({best_blur_method})", round(best_blur_t, 2), f"{best_blur_fpr * 100:.1f}%", f"{best_blur_rec * 100:.1f}%"),
        ("dark_max_p95", round(dark_t, 2), f"{dark_fpr * 100:.1f}%", f"{dark_rec * 100:.1f}%"),
        ("bright_min_p5", round(bright_t, 2), f"{bright_fpr * 100:.1f}%", f"{bright_rec * 100:.1f}%"),
        ("clutter_max", round(clutter_t, 4), f"{clutter_fpr * 100:.1f}%", f"{clutter_rec * 100:.1f}%"),
    ]

    print("=" * 70)
    print("Shoot Coach Parameter Tuning (Phase 1d, Split = Tune)")
    print("=" * 70)
    print(f"{'Metric':<28} | {'Chosen Value':>12} | {'FPR':>8} | {'Recall':>8}")
    print("-" * 70)
    for name, val, fpr, rec in table_rows:
        print(f"{name:<28} | {val:>12} | {fpr:>8} | {rec:>8}")
    print("-" * 70)

    print("\nBlur Method Evaluation:")
    for m_name, (t, fpr, rec) in blur_results.items():
        print(f" - {m_name:<15}: threshold={t:.2f}, FPR={fpr * 100:.1f}%, recall={rec * 100:.1f}%")

    thresholds_dict = {
        "blur_min": round(best_blur_t, 2),
        "blur_method": best_blur_method,
        "dark_max_p95": round(dark_t, 2),
        "bright_min_p5": round(bright_t, 2),
        "clutter_max": round(clutter_t, 4),
    }

    output_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_cfg_path, "w", encoding="utf-8") as f:
        json.dump(thresholds_dict, f, indent=4)

    print(f"\nTuned thresholds saved to: {output_cfg_path}")


if __name__ == "__main__":
    main()
