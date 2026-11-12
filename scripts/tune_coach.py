import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import cv2
import numpy as np

from studio.coach.coach import assess_photo
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


def select_threshold(
    good_vals: List[float],
    bad_vals: List[float],
    direction: str,  # "lower" or "upper"
    fpr_cap: float,
    all_vals: Optional[List[float]] = None,
    default_threshold: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Select candidate threshold maximizing recall with single-check FPR <= fpr_cap.

    Candidate thresholds are the midpoints between adjacent sorted unique feature values.
    Break recall ties by lowest FPR, and if still tied, by the median of tied thresholds.

    Returns:
        (best_threshold, best_fpr, best_recall)
    """
    n_good = len(good_vals)
    n_bad = len(bad_vals)
    if n_good == 0 or n_bad == 0:
        return default_threshold or 0.0, 0.0, 0.0

    raw_vals = all_vals if all_vals is not None else (good_vals + bad_vals)
    u = sorted(list(set(raw_vals)))
    if len(u) <= 1:
        return default_threshold or (u[0] if u else 0.0), 0.0, 0.0

    candidates = [(u[i] + u[i + 1]) / 2.0 for i in range(len(u) - 1)]

    qualifying = []
    for cand in candidates:
        if direction == "lower":
            fpr = sum(1 for v in good_vals if v < cand) / n_good
            rec = sum(1 for v in bad_vals if v < cand) / n_bad
        else:
            fpr = sum(1 for v in good_vals if v > cand) / n_good
            rec = sum(1 for v in bad_vals if v > cand) / n_bad

        if fpr <= fpr_cap:
            qualifying.append({"threshold": cand, "fpr": fpr, "recall": rec})

    if not qualifying:
        fb_t = default_threshold if default_threshold is not None else (u[0] if direction == "lower" else u[-1])
        return fb_t, 0.0, 0.0

    max_rec = max(q["recall"] for q in qualifying)
    tied_rec = [q for q in qualifying if q["recall"] == max_rec]
    min_fpr = min(q["fpr"] for q in tied_rec)
    tied_fpr = [q for q in tied_rec if q["fpr"] == min_fpr]

    tied_cands = [q["threshold"] for q in tied_fpr]
    best_t = float(np.median(tied_cands))
    return best_t, min_fpr, max_rec


def choose_qualifying_cap(
    cap_results: List[Dict[str, Any]],
    max_combined_far: float = 0.10,
) -> float:
    """Select largest cap with combined false alarm rate <= max_combined_far; fallback to 0.0 with warning."""
    qualifying = [r for r in cap_results if r["combined_far"] <= max_combined_far]
    if qualifying:
        best = max(qualifying, key=lambda r: r["cap"])
        return float(best["cap"])
    else:
        print("\nWarning: No cap achieved combined false alarm rate <= 10.0%. Using cap 0.0.")
        return 0.0


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
            "img": img,
            "mask": mask,
            **feats,
        }
        samples.append(sample)

    good_samples = [s for s in samples if s["label"] == "good"]
    bad_blur = [s for s in samples if s["label"] == "bad" and s["reason"] == "blur"]
    bad_dark = [s for s in samples if s["label"] == "bad" and s["reason"] == "dark"]
    bad_bright = [s for s in samples if s["label"] == "bad" and s["reason"] == "bright"]
    bad_clutter = [s for s in samples if s["label"] == "bad" and s["reason"] == "clutter"]

    print(f"Tune split: {len(good_samples)} good, {len(bad_blur)} bad-blur, {len(bad_dark)} bad-dark, {len(bad_bright)} bad-bright, {len(bad_clutter)} bad-clutter\n")

    # Define candidate feature sets per check category: (name, feature_key, direction, bad_group)
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

    caps = [0.05, 0.04, 0.03, 0.02, 0.015, 0.01, 0.005, 0.0]
    cap_evaluations: List[Dict[str, Any]] = []

    for cap in caps:
        chosen_per_cat: Dict[str, Dict[str, Any]] = {}
        all_evals_for_cap: Dict[str, List[Dict[str, Any]]] = {}

        for cat, candidate_list in checks.items():
            cat_results = []
            for name, key, direction, bad_group in candidate_list:
                g_vals = [s[key] for s in good_samples]
                b_vals = [s[key] for s in bad_group]
                all_v = [s[key] for s in samples]
                if cat == "Blur":
                    def_t = cfg_base.blur_min
                elif cat == "Dark":
                    def_t = cfg_base.dark_max_p95
                elif cat == "Bright":
                    def_t = 0.05 if name == "product_clip_hi" else cfg_base.bright_min_p5
                elif cat == "Clutter":
                    def_t = 70.0 if name == "bg_lum_std" else cfg_base.clutter_max
                else:
                    def_t = None
                t, fpr, rec = select_threshold(g_vals, b_vals, direction, cap, all_vals=all_v, default_threshold=def_t)
                cat_results.append({
                    "name": name,
                    "key": key,
                    "direction": direction,
                    "threshold": t,
                    "fpr": fpr,
                    "recall": rec,
                })
            all_evals_for_cap[cat] = cat_results
            # Choose best candidate feature by recall under cap (break tie by lowest FPR)
            best_cand = max(cat_results, key=lambda r: (r["recall"], -r["fpr"]))
            chosen_per_cat[cat] = best_cand

        # Also compute tuned fallback thresholds under this cap for whole-image versions
        fb_dark_t, _, _ = select_threshold(
            [s["global_p95"] for s in good_samples],
            [s["global_p95"] for s in bad_dark],
            "lower",
            cap,
            all_vals=[s["global_p95"] for s in samples],
            default_threshold=cfg_base.dark_max_p95,
        )
        fb_bright_t, _, _ = select_threshold(
            [s["global_p5"] for s in good_samples],
            [s["global_p5"] for s in bad_bright],
            "upper",
            cap,
            all_vals=[s["global_p5"] for s in samples],
            default_threshold=cfg_base.bright_min_p5,
        )
        fb_clutter_t, _, _ = select_threshold(
            [s["global_border"] for s in good_samples],
            [s["global_border"] for s in bad_clutter],
            "upper",
            cap,
            all_vals=[s["global_border"] for s in samples],
            default_threshold=cfg_base.clutter_max,
        )

        # Build candidate CoachConfig for this cap
        cfg_cap = CoachConfig(
            blur_method=chosen_per_cat["Blur"]["name"],
            blur_min=chosen_per_cat["Blur"]["threshold"],
            dark_feature=chosen_per_cat["Dark"]["name"],
            dark_threshold=chosen_per_cat["Dark"]["threshold"],
            dark_max_p95=fb_dark_t,
            bright_feature=chosen_per_cat["Bright"]["name"],
            bright_threshold=chosen_per_cat["Bright"]["threshold"],
            bright_min_p5=fb_bright_t,
            clutter_feature=chosen_per_cat["Clutter"]["name"],
            clutter_threshold=chosen_per_cat["Clutter"]["threshold"],
            clutter_max=fb_clutter_t,
            chosen_cap=cap,
            load_saved_thresholds=False,
        )

        # Evaluate combined Coach on all good tune images (resolution, blur, exposure, clutter, framing)
        good_reps = [assess_photo(s["img"], mask=s["mask"], config=cfg_cap) for s in good_samples]
        comb_far = (sum(1 for r in good_reps if not r.ok) / len(good_samples)) if good_samples else 0.0

        # Compute recall per defect reason on bad tune images
        reason_recalls: Dict[str, float] = {}
        for r_name in ["blur", "bright", "clutter", "dark", "lowres"]:
            b_sub = [s for s in samples if s["label"] == "bad" and s["reason"] == r_name]
            if b_sub:
                b_reps = [assess_photo(s["img"], mask=s["mask"], config=cfg_cap) for s in b_sub]
                rec = sum(1 for r in b_reps if not r.ok) / len(b_sub)
                reason_recalls[r_name] = rec
            else:
                reason_recalls[r_name] = 0.0

        cap_evaluations.append({
            "cap": cap,
            "combined_far": comb_far,
            "reason_recalls": reason_recalls,
            "chosen_per_cat": chosen_per_cat,
            "fb_thresholds": {
                "dark_max_p95": fb_dark_t,
                "bright_min_p5": fb_bright_t,
                "clutter_max": fb_clutter_t,
            },
            "all_evals": all_evals_for_cap,
        })

    # Pick qualifying cap
    chosen_cap_val = choose_qualifying_cap(cap_evaluations, max_combined_far=0.10)
    chosen_record = next(r for r in cap_evaluations if r["cap"] == chosen_cap_val)

    # Print Table of Cap evaluations
    print("=" * 105)
    print("Shoot Coach Joint False-Alarm Budget Tuning (Phase 2c - split=tune)")
    print("=" * 105)
    print(f"{'Cap':<7} | {'Combined FAR':>12} | {'Blur Rec':>8} | {'Bright Rec':>10} | {'Clutter Rec':>11} | {'Dark Rec':>8} | {'Lowres Rec':>10} | {'Status':^12}")
    print("-" * 105)

    for r in cap_evaluations:
        c = r["cap"]
        far = r["combined_far"]
        rec = r["reason_recalls"]
        is_chosen = (c == chosen_cap_val)
        status_str = "CHOSEN (*)" if is_chosen else ""
        print(
            f"{c:<7.3f} | {far * 100:>11.1f}% | {rec['blur'] * 100:>7.1f}% | {rec['bright'] * 100:>9.1f}% | "
            f"{rec['clutter'] * 100:>10.1f}% | {rec['dark'] * 100:>7.1f}% | {rec['lowres'] * 100:>9.1f}% | {status_str:^12}"
        )
    print("=" * 105)

    # Print chosen feature and threshold for every check
    chosen_checks = chosen_record["chosen_per_cat"]
    fb_t = chosen_record["fb_thresholds"]

    print(f"\nChosen Features and Thresholds (cap={chosen_cap_val:.3f}, combined FAR={chosen_record['combined_far']*100:.1f}%):")
    print("-" * 75)
    for cat in ["Blur", "Dark", "Bright", "Clutter"]:
        ch = chosen_checks[cat]
        dir_str = "value < T" if ch["direction"] == "lower" else "value > T"
        t_fmt = f"{ch['threshold']:.4f}" if "density" in ch["name"] or "ratio" in ch["name"] or "clip" in ch["name"] else f"{ch['threshold']:.2f}"
        print(f"  {cat:<8}: feature='{ch['name']}', threshold={t_fmt} ({dir_str}), single-check FPR={ch['fpr']*100:.1f}%, recall={ch['recall']*100:.1f}%")
    print("-" * 75)

    # Save choice to configs/coach_thresholds.json
    best_blur = chosen_checks["Blur"]
    best_dark = chosen_checks["Dark"]
    best_bright = chosen_checks["Bright"]
    best_clutter = chosen_checks["Clutter"]

    thresholds_dict = {
        "chosen_cap": chosen_cap_val,
        "blur_method": best_blur["name"],
        "blur_min": round(best_blur["threshold"], 2),
        "dark_feature": best_dark["name"],
        "dark_threshold": round(best_dark["threshold"], 2),
        "dark_max_p95": round(fb_t["dark_max_p95"], 2),
        "bright_feature": best_bright["name"],
        "bright_threshold": round(best_bright["threshold"], 4),
        "bright_min_p5": round(fb_t["bright_min_p5"], 2),
        "clutter_feature": best_clutter["name"],
        "clutter_threshold": round(best_clutter["threshold"], 4),
        "clutter_max": round(fb_t["clutter_max"], 4),
    }

    output_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_cfg_path, "w", encoding="utf-8") as f:
        json.dump(thresholds_dict, f, indent=4)

    print(f"Saved threshold settings with chosen cap {chosen_cap_val} to: {output_cfg_path}")


if __name__ == "__main__":
    main()
