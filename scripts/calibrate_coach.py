from collections import defaultdict
import csv
from pathlib import Path
import sys
from typing import Dict, List

from studio.coach import CoachConfig, assess_photo, load_image
from studio.coach.mask_cache import get_mask
from studio.env import get_output_dir


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

    print(f"\nSaved calibration log to: {calib_csv_path}")


if __name__ == "__main__":
    main()
