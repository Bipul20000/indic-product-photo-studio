import csv
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
from studio.coach import assess_photo, load_image
from studio.env import get_output_dir

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp", ".tiff"}


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_coach.py <image_path_or_folder>")
        sys.exit(1)

    target_path = Path(sys.argv[1])
    if not target_path.exists():
        print(f"Error: Target path '{target_path}' does not exist.")
        sys.exit(1)

    if target_path.is_file():
        image_files = [target_path]
    elif target_path.is_dir():
        image_files = sorted(
            [p for p in target_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        )
        if not image_files:
            print(f"No image files found in '{target_path}'.")
            sys.exit(0)
    else:
        print(f"Error: '{target_path}' is neither a file nor a directory.")
        sys.exit(1)

    out_dir = get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_csv_path = out_dir / "coach_report.csv"

    rows = []
    for img_path in image_files:
        try:
            img = load_image(img_path)
            report = assess_photo(img)
        except Exception as e:
            print(f"{img_path.name}: RETAKE (Failed to open image: {e})")
            rows.append(
                {
                    "filename": img_path.name,
                    "verdict": "RETAKE",
                    "ok": False,
                    "resolution": 0.0,
                    "blur": 0.0,
                    "brightness": 0.0,
                    "clutter": 0.0,
                    "tips": f"Failed to load image: {e}",
                }
            )
            continue

        verdict = "OK" if report.ok else "RETAKE"
        tips_str = " | ".join(report.tips) if report.tips else ""
        if tips_str:
            print(f"{img_path.name}: {verdict} - {tips_str}")
        else:
            print(f"{img_path.name}: {verdict}")

        res_dict = {r.name: r for r in report.results}
        rows.append(
            {
                "filename": img_path.name,
                "verdict": verdict,
                "ok": report.ok,
                "resolution": res_dict.get("resolution").value if "resolution" in res_dict else "",
                "blur": res_dict.get("blur").value if "blur" in res_dict else "",
                "brightness": res_dict.get("brightness").value if "brightness" in res_dict else "",
                "clutter": res_dict.get("clutter").value if "clutter" in res_dict else "",
                "tips": tips_str,
            }
        )

    fieldnames = [
        "filename",
        "verdict",
        "ok",
        "resolution",
        "blur",
        "brightness",
        "clutter",
        "tips",
    ]
    with open(report_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote report to: {report_csv_path}")


if __name__ == "__main__":
    main()
