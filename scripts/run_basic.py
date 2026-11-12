#!/usr/bin/env python3
"""Run basic product cutout and white background compositing pipeline with diagnostics."""

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

# Ensure repo root and src are on Python path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from studio.coach.io import load_image
from studio.compose.compose import PRESETS
from studio.pipeline import PipelineResult, run_pipeline


def load_excluded_stems(exclude_path: Optional[Path]) -> Set[str]:
    """Read exclusion file and return excluded base stems."""
    if not exclude_path or not exclude_path.exists():
        return set()
    stems = set()
    with open(exclude_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entry = line.split("#")[0].strip()
            stem = entry.replace("_orig.jpg", "").replace(".jpg", "").strip()
            if stem:
                stems.add(stem)
    return stems


def is_excluded(filename: str, excluded_stems: Set[str]) -> bool:
    """Check if filename matches any excluded stem."""
    clean_stem = filename.replace("_orig.jpg", "").replace(".jpg", "").strip()
    for ex in excluded_stems:
        if clean_stem == ex or clean_stem.startswith(ex) or ex.startswith(clean_stem):
            return True
    return False


def make_thumbnail_cell(img: np.ndarray, size: int = 320) -> np.ndarray:
    """Fit image into a square size x size thumbnail with centered white padding."""
    if img is None or img.size == 0:
        return np.full((size, size, 3), 255, dtype=np.uint8)

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((size, size, 3), 255, dtype=np.uint8)

    scale = size / max(h, w)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    cell = np.full((size, size, 3), 255, dtype=np.uint8)
    y_off = (size - nh) // 2
    x_off = (size - nw) // 2
    cell[y_off : y_off + nh, x_off : x_off + nw] = resized
    return cell


def create_mask_overlay(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Highlight foreground mask on original image in red at 40% alpha."""
    overlay = img_bgr.copy()
    if mask is not None and mask.shape[:2] == img_bgr.shape[:2]:
        fg = mask > 0
        if np.any(fg):
            red = np.zeros_like(img_bgr)
            red[:, :, 2] = 255  # Red channel in BGR
            overlay[fg] = (0.60 * img_bgr[fg] + 0.40 * red[fg]).astype(np.uint8)
    return overlay


def build_preview_contact_sheet(
    samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_path: Path,
    thumb_size: int = 320,
) -> None:
    """Build a 3-column contact sheet for up to 12 sample results."""
    if not samples:
        return

    rows = []
    for orig, mask, out_img in samples[:12]:
        cell_orig = make_thumbnail_cell(orig, thumb_size)
        overlay = create_mask_overlay(orig, mask)
        cell_overlay = make_thumbnail_cell(overlay, thumb_size)
        cell_out = make_thumbnail_cell(out_img, thumb_size)
        row = np.hstack([cell_orig, cell_overlay, cell_out])
        rows.append(row)

    contact_sheet = np.vstack(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), contact_sheet)
    print(f"Saved preview contact sheet to: {out_path} ({contact_sheet.shape[1]}x{contact_sheet.shape[0]})")


def build_cutoff_diagnostic_sheet(
    records: List[Dict[str, Any]],
    out_path: Path,
    thumb_size: int = 280,
) -> None:
    """Write outputs/cutoff_diagnostic.jpg using images flagged as cut off under the OLD rule.

    Maximum 16 images (evenly spaced if there are more).
    Three columns: original | mask overlay (red, 40% alpha) | final white-background output.
    Under each row print filename, the 4 side_contact values, and old and new verdicts.
    """
    if not records:
        print("No images flagged under OLD cut-off rule for diagnostic sheet.")
        return

    if len(records) > 16:
        indices = np.linspace(0, len(records) - 1, 16, dtype=int)
        selected = [records[i] for i in indices]
    else:
        selected = records

    row_width = thumb_size * 3  # 840 px
    banner_height = 46

    rows = []
    for rec in selected:
        cell_orig = make_thumbnail_cell(rec["orig"], thumb_size)
        overlay = create_mask_overlay(rec["orig"], rec["mask"])
        cell_overlay = make_thumbnail_cell(overlay, thumb_size)
        cell_out = make_thumbnail_cell(rec["out_img"], thumb_size)
        row_cells = np.hstack([cell_orig, cell_overlay, cell_out])

        # Text banner below row
        banner = np.full((banner_height, row_width, 3), 245, dtype=np.uint8)
        banner[0:1, :] = 200  # separator line

        sc = rec["side_contact"]
        old_v = rec["old_verdict"]
        new_v = rec["new_verdict"]
        fn = rec["filename"]

        line1 = f"{fn}   |   OLD: {old_v}   |   NEW: {new_v}"
        line2 = f"Side contact - Top: {sc['top']:.3f}   Bottom: {sc['bottom']:.3f}   Left: {sc['left']:.3f}   Right: {sc['right']:.3f}"

        cv2.putText(banner, line1, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(banner, line2, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)

        combined_row = np.vstack([row_cells, banner])
        rows.append(combined_row)

    diagnostic_sheet = np.vstack(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), diagnostic_sheet)
    print(f"Saved cut-off diagnostic sheet to: {out_path} ({diagnostic_sheet.shape[1]}x{diagnostic_sheet.shape[0]}) with {len(selected)} samples.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run basic product compositing pipeline on images.")
    parser.add_argument("input", type=str, help="Image file path or directory of images.")
    parser.add_argument("--preset", type=str, default="square_1000", choices=list(PRESETS.keys()), help="Canvas preset.")
    parser.add_argument("--out", type=str, default="outputs/basic/", help="Output directory for generated images.")
    parser.add_argument("--force", action="store_true", help="Force composite output even if quality checks fail.")
    parser.add_argument("--exclude", type=str, default="configs/coach_eval_exclude.txt", help="Path to exclusion list.")
    parser.add_argument("--preview", type=str, default="outputs/basic_preview.jpg", help="Path for contact sheet.")
    parser.add_argument("--diagnostic", type=str, default="outputs/cutoff_diagnostic.jpg", help="Path for cutoff diagnostic.")

    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_path = Path(args.preview)
    diagnostic_path = Path(args.diagnostic)

    # Collect files
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted([p for p in input_path.iterdir() if p.suffix.lower() in valid_exts])
    else:
        print(f"Error: Input path '{input_path}' does not exist.")
        sys.exit(1)

    # Exclusions
    exclude_path = Path(args.exclude) if args.exclude else None
    excluded_stems = load_excluded_stems(exclude_path)
    if excluded_stems and input_path.is_dir():
        before_count = len(files)
        files = [p for p in files if not is_excluded(p.name, excluded_stems)]
        print(f"Excluded {before_count - len(files)} files based on {exclude_path}. {len(files)} remaining.")

    if not files:
        print("No files to process.")
        sys.exit(0)

    durations: List[float] = []
    cut_off_old_count = 0
    cut_off_new_count = 0
    framing_too_small = 0
    framing_too_close = 0
    segmentation_failures = 0
    model_usage: Counter = Counter()
    area_outliers: List[Tuple[str, float]] = []
    preview_samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    old_cutoff_records: List[Dict[str, Any]] = []

    print(f"\nProcessing {len(files)} images with preset '{args.preset}', force={args.force}...\n")
    print(f"{'Filename':<52} | {'Model':<9} | {'Verdict':<6} | {'Sec':<5} | Tips")
    print("-" * 120)

    for p in files:
        res: PipelineResult = run_pipeline(
            image_path=p,
            scene="white",
            marketplace=args.preset,
            force=args.force,
        )

        durations.append(res.seconds)
        verdict = "PASS" if res.ok else "FAIL"
        tips_str = "; ".join(res.report.tips) if res.report.tips else "None"
        filename_display = p.name if len(p.name) <= 52 else (p.name[:49] + "...")
        print(f"{filename_display:<52} | {res.segment.model_used:<9} | {verdict:<6} | {res.seconds:>5.2f} | {tips_str}")

        # Model used accounting
        model_usage[res.segment.model_used] += 1

        # Segmentation failures
        if res.segment.failed or res.segment.area_fraction < 0.005:
            segmentation_failures += 1

        area = res.segment.area_fraction
        if area < 0.02 or area > 0.95:
            area_outliers.append((p.name, area))

        # Check cut off under OLD rule vs NEW rule
        mask = res.segment.mask
        m = 4
        h, w = mask.shape[:2]
        if h <= 2 * m or w <= 2 * m:
            old_cutoff = bool(np.any(mask > 0))
        else:
            old_cutoff = bool(
                np.any(mask[:m, :] > 0)
                or np.any(mask[-m:, :] > 0)
                or np.any(mask[:, :m] > 0)
                or np.any(mask[:, -m:] > 0)
            )

        if old_cutoff:
            cut_off_old_count += 1

        new_cutoff = bool(res.segment.touches_border)
        if new_cutoff:
            cut_off_new_count += 1

        # Check framing too small and too close
        for check in res.report.results:
            if check.name == "framing" and not check.passed:
                tip_lower = check.tip.lower()
                if "too small" in tip_lower:
                    framing_too_small += 1
                elif "too close" in tip_lower:
                    framing_too_close += 1

        # Save output image
        if res.image is not None:
            out_img_path = out_dir / f"{p.stem}.jpg"
            cv2.imwrite(str(out_img_path), res.image)

        # Store samples for basic preview grid (up to 12)
        if len(preview_samples) < 12 and res.image is not None:
            orig_img = load_image(p)
            preview_samples.append((orig_img, res.segment.mask, res.image))

        # Store records flagged under OLD cut-off rule for diagnostic sheet
        if old_cutoff:
            orig_img = load_image(p)
            out_for_diag = res.image if res.image is not None else np.full((1000, 1000, 3), 255, dtype=np.uint8)
            new_v = "FAIL" if not res.ok else "PASS"
            old_cutoff_records.append({
                "orig": orig_img,
                "mask": res.segment.mask,
                "out_img": out_for_diag,
                "filename": p.name,
                "side_contact": res.segment.side_contact,
                "old_verdict": "FAIL",
                "new_verdict": new_v,
                "new_cutoff": new_cutoff,
            })

    # Write preview contact sheet
    if preview_samples:
        build_preview_contact_sheet(preview_samples, preview_path)

    # Write cut-off diagnostic sheet
    if old_cutoff_records:
        build_cutoff_diagnostic_sheet(old_cutoff_records, diagnostic_path, thumb_size=280)

    # Print summary report
    print("\n" + "=" * 65)
    print("PHASE 2a EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Number of images processed:     {len(files)}")
    print(f"Mean processing time:           {np.mean(durations):.3f}s")
    print(f"Max processing time:            {np.max(durations):.3f}s")
    print("\nFraming cut-off rule comparison:")
    print(f"  - Cut-off failures under OLD rule: {cut_off_old_count}")
    print(f"  - Cut-off failures under NEW rule: {cut_off_new_count}")
    print("\nOther framing metrics:")
    print(f"  - Product too small:          {framing_too_small}")
    print(f"  - Product too close:          {framing_too_close}")
    print(f"\nTotal segmentation failures:    {segmentation_failures}")
    print("\nModel usage distribution:")
    for model_name, count in model_usage.items():
        print(f"  - {model_name}: {count} image(s)")
    if area_outliers:
        print(f"\nArea fraction outliers (< 0.02 or > 0.95): {len(area_outliers)}")
        for name, area in area_outliers:
            print(f"  - {name}: area_fraction = {area:.4f}")
    else:
        print("\nArea fraction outliers (< 0.02 or > 0.95): 0")
    print("=" * 65)


if __name__ == "__main__":
    main()
