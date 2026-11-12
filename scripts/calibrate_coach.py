from collections import defaultdict
import csv
from pathlib import Path
import sys
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import cv2
import numpy as np

from studio.coach import CoachConfig, assess_photo, load_image
from studio.coach.mask_cache import get_mask
from studio.compose import make_white_background
from studio.env import get_output_dir


def make_thumb(image: np.ndarray, target_w: int = 280, target_h: int = 280) -> np.ndarray:
    """Pad and resize image to fit in target_w x target_h box on white background."""
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


def create_rejected_good_contact_sheet(
    rejected_origs: List[Dict],
    images_dir: Path,
    out_path: Path,
    config: CoachConfig,
    max_items: int = 16,
) -> None:
    """Generate contact sheet for rejected good originals up to max_items."""
    items = rejected_origs[:max_items]
    if not items:
        blank = np.full((100, 400, 3), 255, dtype=np.uint8)
        cv2.putText(blank, "No rejected good originals.", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)
        cv2.imwrite(str(out_path), blank)
        return

    thumb_w = 280
    thumb_h = 280
    margin = 16
    header_h = 44
    caption_h = 46
    row_h = thumb_h + caption_h + margin

    sheet_w = margin + 3 * (thumb_w + margin)
    sheet_h = header_h + len(items) * row_h + margin
    sheet = np.full((sheet_h, sheet_w, 3), 248, dtype=np.uint8)

    # Header bar
    cv2.rectangle(sheet, (0, 0), (sheet_w, header_h), (235, 235, 235), -1)
    cv2.line(sheet, (0, header_h), (sheet_w, header_h), (210, 210, 210), 1)

    headers = ["1. Original Photo", "2. Mask Overlay (Red 40%)", "3. White-Background Output"]
    for col_idx, title in enumerate(headers):
        x = margin + col_idx * (thumb_w + margin)
        cv2.putText(sheet, title, (x + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)

    # Rows
    for row_idx, item in enumerate(items):
        y_top = header_h + margin + row_idx * row_h
        img_p = images_dir / item["filename"]
        img = load_image(img_p)
        mask = get_mask(img_p, key=item["filename"])

        # Col 1: Original
        th_orig = make_thumb(img, thumb_w, thumb_h)

        # Col 2: Mask Overlay (Red, 40% alpha)
        overlay = img.copy().astype(np.float32)
        red_bgr = np.array([0, 0, 255], dtype=np.float32)
        m_bool = (mask > 0)
        overlay[m_bool] = 0.6 * overlay[m_bool] + 0.4 * red_bgr
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        th_overlay = make_thumb(overlay, thumb_w, thumb_h)

        # Col 3: Final white-background output
        try:
            white_out, _, _ = make_white_background(img, config=config)
            th_white = make_thumb(white_out, thumb_w, thumb_h)
        except Exception:
            th_white = np.full((thumb_h, thumb_w, 3), 255, dtype=np.uint8)

        # Place thumbnails
        cols = [th_orig, th_overlay, th_white]
        for col_idx, th in enumerate(cols):
            x = margin + col_idx * (thumb_w + margin)
            sheet[y_top : y_top + thumb_h, x : x + thumb_w] = th

        # Caption under row
        cap_y = y_top + thumb_h + 16
        checks_str = item.get("failing_checks", "")
        failing_vals = []
        for chk in ["resolution", "blur", "brightness", "clutter", "framing"]:
            if chk in checks_str:
                val = item.get(chk, "")
                failing_vals.append(f"{chk}={val}")
        failing_info_str = ", ".join(failing_vals)

        cv2.putText(
            sheet,
            f"#{row_idx + 1}: {item['filename']} [{item['split']}]",
            (margin, cap_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            f"Failed: [{checks_str}]  |  Values: {failing_info_str}",
            (margin, cap_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 0, 180),
            1,
            cv2.LINE_AA,
        )
        sep_y = y_top + row_h - (margin // 2)
        cv2.line(sheet, (margin, sep_y), (sheet_w - margin, sep_y), (225, 225, 225), 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Saved rejected goods contact sheet to: {out_path} ({len(items)} images)")


def format_confusion_matrix(matrix: Dict[tuple, int]) -> str:
    lines = [
        f"{'':>15} | {'Coach: good':>12} | {'Coach: bad':>12}",
        "-" * 45,
        f"{'Human: good':>15} | {matrix[('good', 'good')]:>12} | {matrix[('good', 'bad')]:>12}",
        f"{'Human: bad':>15} | {matrix[('bad', 'good')]:>12} | {matrix[('bad', 'bad')]:>12}",
        "-" * 45,
    ]
    return "\n".join(lines)


def print_split_report(split_title: str, records: List[Dict]) -> None:
    total = len(records)
    if total == 0:
        print(f"\n--- {split_title} --- (0 images)")
        return

    correct = sum(1 for r in records if r["human_label"] == r["coach_verdict"])
    accuracy = (correct / total) * 100.0

    matrix = {
        ("good", "good"): 0,
        ("good", "bad"): 0,
        ("bad", "good"): 0,
        ("bad", "bad"): 0,
    }
    for r in records:
        matrix[(r["human_label"], r["coach_verdict"])] += 1

    good_origs = [r for r in records if r["source"] == "original" and r["human_label"] == "good"]
    orig_fa = sum(1 for r in good_origs if r["coach_verdict"] == "bad")
    orig_far = (orig_fa / len(good_origs) * 100.0) if good_origs else 0.0

    fb_count = sum(1 for r in records if r["used_fallback"])
    fb_pct = (fb_count / total * 100.0) if total > 0 else 0.0

    print(f"\n=================================================================")
    print(f"--- {split_title} ---")
    print(f"=================================================================")
    print(f"Total Evaluated: {total}")
    print(f"Overall Accuracy: {accuracy:.1f}%")
    print(f"False Alarm Rate on Good Originals: {orig_far:.1f}% ({orig_fa}/{len(good_origs)} rejected)")
    print(f"Fraction using Fallback: {fb_pct:.1f}% ({fb_count}/{total} images)")
    print("\nConfusion Matrix:")
    print(format_confusion_matrix(matrix))

    # Per-reason recall
    bad_records = [r for r in records if r["human_label"] == "bad"]
    reason_stats = defaultdict(lambda: {"total": 0, "detected_bad": 0})
    for r in bad_records:
        reason = r["human_reason"] or "unknown"
        reason_stats[reason]["total"] += 1
        if r["coach_verdict"] == "bad":
            reason_stats[reason]["detected_bad"] += 1

    print("\nPer-Reason Recall for Bad Images:")
    print(f"{'Defect Reason':<15} | {'Total':>8} | {'Detected':>10} | {'Recall':>8}")
    print("-" * 48)
    for reason, data in sorted(reason_stats.items()):
        rec = (data["detected_bad"] / data["total"] * 100.0) if data["total"] > 0 else 0.0
        print(f"{reason:<15} | {data['total']:>8} | {data['detected_bad']:>10} | {rec:>7.1f}%")
    print("-" * 48)


def main() -> None:
    eval_dir = Path("data/coach_eval")
    images_dir = eval_dir / "images"
    labels_path = eval_dir / "labels.csv"

    if not labels_path.exists():
        print(f"labels.csv not found at {labels_path}. Please run make_coach_eval.py first.")
        sys.exit(1)

    with open(labels_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        label_rows = list(reader)

    if not label_rows:
        print(f"No entries found in {labels_path}.")
        sys.exit(0)

    out_dir = get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_csv_path = out_dir / "coach_calibration.csv"

    cfg = CoachConfig()  # Auto-loads tuned thresholds from configs/coach_thresholds.json
    print(f"Using CoachConfig: blur_min={cfg.blur_min} ({cfg.blur_method}), dark_feature={cfg.dark_feature} (T={cfg.dark_threshold}), "
          f"bright_feature={cfg.bright_feature} (T={cfg.bright_threshold}), clutter_feature={cfg.clutter_feature} (T={cfg.clutter_threshold})")

    eval_records = []
    for row in label_rows:
        filename = row.get("filename", "").strip()
        human_label = row.get("label", "").strip().lower()
        human_reason = row.get("reason", "").strip()
        source = row.get("source", "original").strip().lower() or "original"
        split = row.get("split", "test").strip().lower() or "test"

        if human_label not in ("good", "bad"):
            continue

        img_path = images_dir / filename
        if not img_path.exists():
            print(f"Warning: Image file not found: {img_path}")
            continue

        try:
            img = load_image(img_path)
            mask = get_mask(img_path, key=filename)
            report = assess_photo(img, mask=mask, config=cfg)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

        coach_verdict = "good" if report.ok else "bad"
        res_dict = {r.name: r for r in report.results}
        tips_str = " | ".join(report.tips) if report.tips else ""
        used_fb = any(r.used_fallback for r in report.results)

        failing_checks = [r.name for r in report.results if not r.passed]

        is_match = (human_label == coach_verdict)
        record = {
            "filename": filename,
            "split": split,
            "source": source,
            "human_label": human_label,
            "coach_verdict": coach_verdict,
            "agreement": "YES" if is_match else "NO",
            "human_reason": human_reason,
            "failing_checks": ", ".join(failing_checks),
            "resolution": res_dict.get("resolution").value if "resolution" in res_dict else "",
            "blur": res_dict.get("blur").value if "blur" in res_dict else "",
            "brightness": res_dict.get("brightness").value if "brightness" in res_dict else "",
            "clutter": res_dict.get("clutter").value if "clutter" in res_dict else "",
            "framing": res_dict.get("framing").value if "framing" in res_dict else "",
            "used_fallback": used_fb,
            "tips": tips_str,
        }
        eval_records.append(record)

    # 1. Headline results: split=test only
    test_records = [r for r in eval_records if r["split"] == "test"]
    print_split_report("Headline Results: split=test", test_records)

    # 2. Split=tune results
    tune_records = [r for r in eval_records if r["split"] == "tune"]
    print_split_report("tune split (thresholds were fit here)", tune_records)

    # 3. List EVERY good image that Coach rejected (both splits)
    rejected_goods = [r for r in eval_records if r["human_label"] == "good" and r["coach_verdict"] == "bad"]
    print(f"\n=================================================================")
    print(f"Good Images Rejected by Coach ({len(rejected_goods)} total across both splits):")
    print(f"=================================================================")
    if rejected_goods:
        for idx, r in enumerate(rejected_goods, start=1):
            fb_str = " [fallback]" if r["used_fallback"] else ""
            print(
                f"{idx}. {r['filename']} [split={r['split']}, source={r['source']}]{fb_str}: failed [{r['failing_checks']}] "
                f"(res={r['resolution']}, blur={r['blur']}, bright={r['brightness']}, clutter={r['clutter']}, framing={r['framing']})"
            )
            if r["tips"]:
                print(f"   Tips: {r['tips']}")
    else:
        print("None! All good images passed.")

    # Write calibration CSV
    fieldnames = [
        "filename",
        "split",
        "source",
        "human_label",
        "coach_verdict",
        "agreement",
        "human_reason",
        "failing_checks",
        "resolution",
        "blur",
        "brightness",
        "clutter",
        "framing",
        "used_fallback",
        "tips",
    ]
    with open(calib_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eval_records)

    # Write contact sheet of rejected good originals (up to 16)
    rejected_good_origs = [
        r for r in eval_records
        if r["source"] == "original" and r["human_label"] == "good" and r["coach_verdict"] == "bad"
    ]
    contact_sheet_path = out_dir / "rejected_good_sheet.jpg"
    create_rejected_good_contact_sheet(
        rejected_good_origs,
        images_dir=images_dir,
        out_path=contact_sheet_path,
        config=cfg,
        max_items=16,
    )

    print(f"\nSaved calibration log to: {calib_csv_path}")


if __name__ == "__main__":
    main()
