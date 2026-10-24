import csv
import sys
from pathlib import Path
from studio.coach import assess_photo, load_image
from studio.env import get_output_dir

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp", ".tiff"}


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
            writer.writerow(["filename", "label", "reason"])
            for fname in image_files:
                writer.writerow([fname, "", ""])

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

    matrix = {
        ("good", "good"): 0,  # Human good, Coach good
        ("good", "bad"): 0,   # Human good, Coach bad
        ("bad", "good"): 0,   # Human bad,  Coach good
        ("bad", "bad"): 0,    # Human bad,  Coach bad
    }

    disagreements = []
    calibration_rows = []
    total_evaluated = 0
    correct_count = 0

    for row in label_rows:
        filename = row.get("filename", "").strip()
        human_label = row.get("label", "").strip().lower()
        human_reason = row.get("reason", "").strip()

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
        total_evaluated += 1
        if is_match:
            correct_count += 1
        else:
            disagreements.append(
                {
                    "filename": filename,
                    "human": human_label,
                    "coach": coach_verdict,
                    "reason": human_reason,
                    "resolution": res_dict.get("resolution").value if "resolution" in res_dict else "",
                    "blur": res_dict.get("blur").value if "blur" in res_dict else "",
                    "brightness": res_dict.get("brightness").value if "brightness" in res_dict else "",
                    "clutter": res_dict.get("clutter").value if "clutter" in res_dict else "",
                    "tips": tips_str,
                }
            )

        matrix[(human_label, coach_verdict)] += 1

        calibration_rows.append(
            {
                "filename": filename,
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
        )

    # Output metrics
    accuracy = (correct_count / total_evaluated * 100.0) if total_evaluated > 0 else 0.0
    print("=" * 50)
    print("Shoot Coach Calibration Report")
    print("=" * 50)
    print(f"Total Evaluated: {total_evaluated}")
    print(f"Overall Accuracy: {accuracy:.1f}%\n")

    print("Confusion Matrix:")
    print(f"{'':>15} | {'Coach: good':>12} | {'Coach: bad':>12}")
    print("-" * 45)
    print(f"{'Human: good':>15} | {matrix[('good', 'good')]:>12} | {matrix[('good', 'bad')]:>12}")
    print(f"{'Human: bad':>15} | {matrix[('bad', 'good')]:>12} | {matrix[('bad', 'bad')]:>12}")
    print("-" * 45)

    if disagreements:
        print(f"\nDisagreements ({len(disagreements)}):")
        for d in disagreements:
            print(
                f"- {d['filename']}: Human={d['human']} vs Coach={d['coach']} "
                f"(res={d['resolution']}, blur={d['blur']}, bright={d['brightness']}, clutter={d['clutter']})"
            )
            if d["tips"]:
                print(f"  Tips: {d['tips']}")
    else:
        print("\nNo disagreements found!")

    fieldnames = [
        "filename",
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
        writer.writerows(calibration_rows)

    print(f"\nSaved calibration log to: {calib_csv_path}")


if __name__ == "__main__":
    main()
