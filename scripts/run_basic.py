#!/usr/bin/env python3
"""Run basic product cutout and white background compositing pipeline."""

import argparse
from pathlib import Path
import sys
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np

# Ensure src is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run basic product compositing pipeline on images.")
    parser.add_argument("input", type=str, help="Image file path or directory of images.")
    parser.add_argument("--preset", type=str, default="square_1000", choices=list(PRESETS.keys()), help="Canvas preset.")
    parser.add_argument("--out", type=str, default="outputs/basic/", help="Output directory for generated images.")
    parser.add_argument("--force", action="store_true", help="Force composite output even if quality checks fail.")
    parser.add_argument("--exclude", type=str, default="configs/coach_eval_exclude.txt", help="Path to exclusion list.")
    parser.add_argument("--preview", type=str, default="outputs/basic_preview.jpg", help="Path for contact sheet.")

    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_path = Path(args.preview)

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
    framing_cut_off = 0
    framing_too_small = 0
    framing_too_close = 0
    segmentation_failures = 0
    area_outliers: List[Tuple[str, float]] = []
    preview_samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    print(f"\nProcessing {len(files)} images with preset '{args.preset}', force={args.force}...\n")
    print(f"{'Filename':<55} | {'Verdict':<6} | {'Sec':<5} | Tips")
    print("-" * 110)

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
        filename_display = p.name if len(p.name) <= 55 else (p.name[:52] + "...")
        print(f"{filename_display:<55} | {verdict:<6} | {res.seconds:>5.2f} | {tips_str}")

        # Metrics accounting
        if res.segment.failed or res.segment.area_fraction < 0.005:
            segmentation_failures += 1

        area = res.segment.area_fraction
        if area < 0.02 or area > 0.95:
            area_outliers.append((p.name, area))

        # Check framing failure types
        for check in res.report.results:
            if check.name == "framing" and not check.passed:
                tip_lower = check.tip.lower()
                if "cut off" in tip_lower:
                    framing_cut_off += 1
                elif "too small" in tip_lower:
                    framing_too_small += 1
                elif "too close" in tip_lower:
                    framing_too_close += 1

        # Save output image if generated
        if res.image is not None:
            out_img_path = out_dir / f"{p.stem}.jpg"
            cv2.imwrite(str(out_img_path), res.image)

            # Store for preview grid (up to 12)
            if len(preview_samples) < 12:
                orig_img = load_image(p)
                preview_samples.append((orig_img, res.segment.mask, res.image))

    # Write preview contact sheet
    if preview_samples:
        build_preview_contact_sheet(preview_samples, preview_path)

    # Print summary report
    print("\n" + "=" * 60)
    print("PHASE 2 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Number of images processed: {len(files)}")
    print(f"Mean processing time:       {np.mean(durations):.3f}s")
    print(f"Max processing time:        {np.max(durations):.3f}s")
    print("\nFraming failures breakdown:")
    print(f"  - Cut off at edge:        {framing_cut_off}")
    print(f"  - Product too small:      {framing_too_small}")
    print(f"  - Product too close:      {framing_too_close}")
    print(f"Total segmentation failures (area < 0.005): {segmentation_failures}")
    print(f"\nArea fraction outliers (< 0.02 or > 0.95): {len(area_outliers)}")
    for name, area in area_outliers:
        print(f"  - {name}: area_fraction = {area:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
