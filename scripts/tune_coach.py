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
    """Compute raw feature values for Shoot Coach checks."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 and img.shape[2] == 3 else img

    # Blur resized
    longest = max(h, w)
    if longest != config.blur_resize_side and longest > 0:
        scale = config.blur_resize_side / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        gray_blur = cv2.resize(gray, (new_w, new_h), interpolation=interp)
    else:
        gray_blur = gray

    # 8x8 tiles for blur
    row_chunks = np.array_split(gray_blur, 8, axis=0)
    tile_vars = []
    for r in row_chunks:
        for t in np.array_split(r, 8, axis=1):
            if t.size > 0:
                tile_vars.append(float(cv2.Laplacian(t, cv2.CV_64F).var()))

    p90 = float(np.percentile(tile_vars, 90)) if tile_vars else 0.0
    p95 = float(np.percentile(tile_vars, 95)) if tile_vars else 0.0
    p_max = float(np.max(tile_vars)) if tile_vars else 0.0

    # Brightness
    mean_bright = float(np.mean(gray))
    clipped_frac = float(np.mean((gray <= 5) | (gray >= 250)))

    # Clutter
    edges = cv2.Canny(gray, 50, 150)
    band_h = max(1, int(round(h * config.border_band_fraction)))
    band_w = max(1, int(round(w * config.border_band_fraction)))
    border_mask = np.ones((h, w), dtype=bool)
    if h > 2 * band_h and w > 2 * band_w:
        border_mask[band_h : h - band_h, band_w : w - band_w] = False

    tot_border = int(np.count_nonzero(border_mask))
    clutter_density = float(np.count_nonzero((edges > 0) & border_mask) / tot_border) if tot_border > 0 else 0.0

    return {
        "blur_p90": p90,
        "blur_p95": p95,
        "blur_max": p_max,
        "mean_bright": mean_bright,
        "clipped_frac": clipped_frac,
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

    # Candidates: all sorted unique values from both sets
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

    # 1. Blur Percentile & Threshold
    blur_results = {}
    for p_name, feat_key in [("p90", "blur_p90"), ("p95", "blur_p95"), ("max", "blur_max")]:
        g_vals = [s[feat_key] for s in good_samples]
        b_vals = [s[feat_key] for s in bad_blur]
        t, fpr, rec = find_lower_threshold(g_vals, b_vals, max_fpr=0.05)
        blur_results[p_name] = (t, fpr, rec)

    best_p = max(blur_results.keys(), key=lambda k: (blur_results[k][2], -blur_results[k][1]))
    best_blur_t, best_blur_fpr, best_blur_rec = blur_results[best_p]

    # 2. Dark threshold (mean_brightness < T)
    g_bright = [s["mean_bright"] for s in good_samples]
    b_dark = [s["mean_bright"] for s in bad_dark]
    dark_t, dark_fpr, dark_rec = find_lower_threshold(g_bright, b_dark, max_fpr=0.05)

    # 3. Bright threshold (mean_brightness > T)
    b_bright = [s["mean_bright"] for s in bad_bright]
    bright_t, bright_fpr, bright_rec = find_upper_threshold(g_bright, b_bright, max_fpr=0.05)

    # 4. Clipped fraction threshold (clipped_frac > T)
    g_clipped = [s["clipped_frac"] for s in good_samples]
    # For clipped, evaluate recall against bright and dark defects
    b_clipped = [s["clipped_frac"] for s in (bad_bright + bad_dark)]
    clip_t, clip_fpr, clip_rec = find_upper_threshold(g_clipped, b_clipped, max_fpr=0.05)

    # 5. Clutter threshold (clutter_density > T)
    g_clutter = [s["clutter_density"] for s in good_samples]
    b_clutter = [s["clutter_density"] for s in bad_clutter]
    clutter_t, clutter_fpr, clutter_rec = find_upper_threshold(g_clutter, b_clutter, max_fpr=0.05)

    # Output table
    table_rows = [
        (f"blur_min ({best_p})", round(best_blur_t, 2), f"{best_blur_fpr * 100:.1f}%", f"{best_blur_rec * 100:.1f}%"),
        ("dark_max_mean", round(dark_t, 2), f"{dark_fpr * 100:.1f}%", f"{dark_rec * 100:.1f}%"),
        ("bright_min_mean", round(bright_t, 2), f"{bright_fpr * 100:.1f}%", f"{bright_rec * 100:.1f}%"),
        ("clipped_max_fraction", round(clip_t, 4), f"{clip_fpr * 100:.1f}%", f"{clip_rec * 100:.1f}%"),
        ("clutter_max", round(clutter_t, 4), f"{clutter_fpr * 100:.1f}%", f"{clutter_rec * 100:.1f}%"),
    ]

    print("=" * 68)
    print("Shoot Coach Parameter Tuning (Split = Tune)")
    print("=" * 68)
    print(f"{'Metric':<25} | {'Chosen Value':>12} | {'FPR':>8} | {'Recall':>8}")
    print("-" * 68)
    for name, val, fpr, rec in table_rows:
        print(f"{name:<25} | {val:>12} | {fpr:>8} | {rec:>8}")
    print("-" * 68)

    print("\nBlur Percentile Comparison:")
    for p_name, (t, fpr, rec) in blur_results.items():
        print(f" - {p_name:<4}: threshold={t:.2f}, FPR={fpr * 100:.1f}%, recall={rec * 100:.1f}%")

    thresholds_dict = {
        "blur_min": round(best_blur_t, 2),
        "blur_percentile": best_p,
        "dark_max_mean": round(dark_t, 2),
        "bright_min_mean": round(bright_t, 2),
        "clipped_max_fraction": round(clip_t, 4),
        "clutter_max": round(clutter_t, 4),
    }

    output_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_cfg_path, "w", encoding="utf-8") as f:
        json.dump(thresholds_dict, f, indent=4)

    print(f"\nTuned thresholds saved to: {output_cfg_path}")


if __name__ == "__main__":
    main()
