import csv
from collections import defaultdict
from pathlib import Path
import sys
from typing import Dict, List
from studio.coach import assess_photo, load_image
from studio.env import get_output_dir

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp", ".tiff"}


def format_confusion_matrix(matrix: Dict[tuple, int]) -> str:
    lines = [
        f"{'':>15} | {'Coach: good':>12} | {'Coach: bad':>12}",
        "-" * 45,
        f"{'Human: good':>15} | {matrix[('good', 'good')]:>12} | {matrix[('good', 'bad')]:>12}",
        f"{'Human: bad':>15} | {matrix[('bad', 'good')]:>12} | {matrix[('bad', 'bad')]:>12}",
        "-" * 45,
    ]
    return "\n".join(lines)


def evaluate_subset(name: str, records: List[Dict]) -> Dict:
    total = len(records)
    if total == 0:
        return {
            "name": name,
            "total": 0,
            "accuracy": 0.0,
            "matrix": {("good", "good"): 0, ("good", "bad"): 0, ("bad", "good"): 0, ("bad", "bad"): 0},
            "disagreements": [],
        }

    correct = sum(1 for r in records if r["human_label"] == r["coach_verdict"])
    accuracy = (correct / total) * 100.0

    matrix = {
        ("good", "good"): 0,
        ("good", "bad"): 0,
        ("bad", "good"): 0,
        ("bad", "bad"): 0,
    }
    disagreements = []

    for r in records:
        matrix[(r["human_label"], r["coach_verdict"])] += 1
        if r["human_label"] != r["coach_verdict"]:
            disagreements.append(r)

    return {
        "name": name,
        "total": total,
        "accuracy": accuracy,
        "matrix": matrix,
        "disagreements": disagreements,
    }


def main():
    eval_dir = Path("data/coach_eval")
    images_dir = eval_dir / "images"
    labels_path = eval_dir / "labels.csv"

    if not labels_path.exists():
        eval_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            [p.name for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        )

        with open(labels_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "reason", "source"])
            for fname in image_files:
                writer.writerow([fname, "", "", ""])

        print(
            f"labels.csv not found. Created template with {len(image_files)} image(s) at {labels_path}."
        )
        print("Please annotate 'good' or 'bad' under 'label' (optional: 'reason') and re-run.")
        sys.exit(0)

    with open(labels_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        label_rows = list(reader)

    if not label_rows:
        print(f"No entries found in {labels_path}.")
        sys.exit(0)

    out_dir = get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_csv_path = out_dir / "coach_calibration.csv"

    eval_records = []
    for row in label_rows:
        filename = row.get("filename", "").strip()
        human_label = row.get("label", "").strip().lower()
        human_reason = row.get("reason", "").strip()
        source = row.get("source", "original").strip().lower() or "original"

        if human_label not in ("good", "bad"):
            continue

        img_path = images_dir / filename
        if not img_path.exists():
            print(f"Warning: Image file not found: {img_path}")
            continue

        try:
            img = load_image(img_path)
            report = assess_photo(img)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

        coach_verdict = "good" if report.ok else "bad"
        res_dict = {r.name: r for r in report.results}
        tips_str = " | ".join(report.tips) if report.tips else ""

        is_match = (human_label == coach_verdict)
        record = {
            "filename": filename,
            "source": source,
            "human_label": human_label,
            "coach_verdict": coach_verdict,
            "agreement": "YES" if is_match else "NO",
            "human_reason": human_reason,
            "resolution": res_dict.get("resolution").value if "resolution" in res_dict else "",
            "blur": res_dict.get("blur").value if "blur" in res_dict else "",
            "brightness": res_dict.get("brightness").value if "brightness" in res_dict else "",
            "clutter": res_dict.get("clutter").value if "clutter" in res_dict else "",
            "tips": tips_str,
        }
        eval_records.append(record)

    # Subsets
    orig_records = [r for r in eval_records if r["source"] == "original"]
    synth_records = [r for r in eval_records if r["source"] == "synthetic"]

    stats_orig = evaluate_subset("source=original", orig_records)
    stats_synth = evaluate_subset("source=synthetic", synth_records)
    stats_all = evaluate_subset("Combined Total", eval_records)

    print("=" * 65)
    print("Shoot Coach Calibration Report")
    print("=" * 65)
    print("NOTE: 'good' labels on original images are assumed from Wikimedia")
    print("Commons open licensing, not human-verified.\n")

    for stats in [stats_orig, stats_synth, stats_all]:
        print(f"--- {stats['name']} ---")
        print(f"Total Evaluated: {stats['total']}")
        print(f"Accuracy: {stats['accuracy']:.1f}%")
        print("\nConfusion Matrix:")
        print(format_confusion_matrix(stats["matrix"]))
        print()

    # Per-reason recall for bad images
    bad_records = [r for r in eval_records if r["human_label"] == "bad"]
    reason_stats = defaultdict(lambda: {"total": 0, "detected_bad": 0})
    for r in bad_records:
        reason = r["human_reason"] or "unknown"
        reason_stats[reason]["total"] += 1
        if r["coach_verdict"] == "bad":
            reason_stats[reason]["detected_bad"] += 1

    print("--- Per-Reason Recall for Bad Images ---")
    print(f"{'Defect Reason':<15} | {'Total':>8} | {'Detected':>10} | {'Recall':>8}")
    print("-" * 48)
    for reason, data in sorted(reason_stats.items()):
        rec = (data["detected_bad"] / data["total"] * 100.0) if data["total"] > 0 else 0.0
        print(f"{reason:<15} | {data['total']:>8} | {data['detected_bad']:>10} | {rec:>7.1f}%")
    print("-" * 48)

    # Combined disagreements
    disagreements = stats_all["disagreements"]
    print(f"\n--- Disagreements ({len(disagreements)} total) ---")
    if disagreements:
        for idx, d in enumerate(disagreements[:15], start=1):
            reason_info = f", reason={d['human_reason']}" if d['human_reason'] else ""
            print(
                f"{idx}. {d['filename']} [{d['source']}]: Human={d['human_label']} vs Coach={d['coach_verdict']}{reason_info} "
                f"(res={d['resolution']}, blur={d['blur']}, bright={d['brightness']}, clutter={d['clutter']})"
            )
            if d["tips"]:
                print(f"   Tips: {d['tips']}")
        if len(disagreements) > 15:
            print(f"... and {len(disagreements) - 15} more disagreements.")
    else:
        print("No disagreements found!")

    # Write calibration CSV
    fieldnames = [
        "filename",
        "source",
        "human_label",
        "coach_verdict",
        "agreement",
        "human_reason",
        "resolution",
        "blur",
        "brightness",
        "clutter",
        "tips",
    ]
    with open(calib_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eval_records)

    print(f"\nSaved calibration log to: {calib_csv_path}")


if __name__ == "__main__":
    main()
