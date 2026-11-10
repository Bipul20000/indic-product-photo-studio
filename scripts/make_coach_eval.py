import csv
from collections import Counter
from pathlib import Path
import random
import shutil
import sys
from typing import Dict, List, Tuple
import cv2
import numpy as np

from studio.coach.io import load_image


def apply_blur(img: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur with sigma 6 or motion blur of length 25."""
    pick = random.choice(["gaussian", "motion"])
    if pick == "gaussian":
        return cv2.GaussianBlur(img, (0, 0), sigmaX=6.0, sigmaY=6.0)
    else:
        k = np.zeros((25, 25), dtype=np.float32)
        m_type = random.choice(["h", "v", "d1", "d2"])
        if m_type == "h":
            k[12, :] = 1.0 / 25.0
        elif m_type == "v":
            k[:, 12] = 1.0 / 25.0
        elif m_type == "d1":
            np.fill_diagonal(k, 1.0 / 25.0)
        else:
            np.fill_diagonal(np.fliplr(k), 1.0 / 25.0)
        return cv2.filter2D(img, -1, k)


def apply_dark(img: np.ndarray) -> np.ndarray:
    """Darken image by multiplying by 0.25."""
    return np.clip(img.astype(float) * 0.25, 0, 255).astype(np.uint8)


def apply_bright(img: np.ndarray) -> np.ndarray:
    """Brighten image with alpha=1.8, beta=60."""
    return np.clip(img.astype(float) * 1.8 + 60.0, 0, 255).astype(np.uint8)


def apply_lowres(img: np.ndarray) -> np.ndarray:
    """Downscale image so longest side is 400 px."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > 400:
        scale = 400.0 / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img.copy()


def apply_clutter(img: np.ndarray, all_images: List[np.ndarray], current_idx: int) -> np.ndarray:
    """Paste 6 to 10 random patches from other originals into the outer 15% border band only."""
    h, w = img.shape[:2]
    cluttered = img.copy()
    band_h = max(1, int(round(h * 0.15)))
    band_w = max(1, int(round(w * 0.15)))

    border_mask = np.ones((h, w), dtype=bool)
    if h > 2 * band_h and w > 2 * band_w:
        border_mask[band_h : h - band_h, band_w : w - band_w] = False

    # Pick donor sources
    other_indices = [i for i in range(len(all_images)) if i != current_idx]
    if not other_indices:
        other_indices = [current_idx]

    num_patches = random.randint(6, 10)
    for _ in range(num_patches):
        donor_idx = random.choice(other_indices)
        donor_img = all_images[donor_idx]
        dh, dw = donor_img.shape[:2]

        sz = random.randint(max(10, int(w * 0.10)), max(10, int(w * 0.25)))
        sz = min(sz, dh, dw, h, w)

        src_y = random.randint(0, max(0, dh - sz))
        src_x = random.randint(0, max(0, dw - sz))
        patch = donor_img[src_y : src_y + sz, src_x : src_x + sz]

        side = random.choice(["top", "bottom", "left", "right"])
        if side == "top":
            y1 = 0
            x1 = random.randint(0, max(0, w - sz))
        elif side == "bottom":
            y1 = max(0, h - sz)
            x1 = random.randint(0, max(0, w - sz))
        elif side == "left":
            y1 = random.randint(0, max(0, h - sz))
            x1 = 0
        else:
            y1 = random.randint(0, max(0, h - sz))
            x1 = max(0, w - sz)

        roi = cluttered[y1 : y1 + sz, x1 : x1 + sz]
        roi_mask = border_mask[y1 : y1 + sz, x1 : x1 + sz]
        blended = (0.1 * roi + 0.9 * patch).astype(np.uint8)
        roi[roi_mask] = blended[roi_mask]

    return cluttered


def build_evaluation_set(
    originals_dir: Path,
    output_images_dir: Path,
    output_labels_csv: Path,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Build the Shoot Coach evaluation dataset deterministically."""
    random.seed(seed)
    np.random.seed(seed)

    if output_images_dir.exists():
        shutil.rmtree(output_images_dir)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    if output_labels_csv.exists():
        output_labels_csv.unlink()
    output_labels_csv.parent.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    orig_files = sorted([p for p in originals_dir.iterdir() if p.suffix.lower() in valid_exts])
    if not orig_files:
        print(f"No original images found in {originals_dir}.")
        return []

    print(f"Loading {len(orig_files)} original images from {originals_dir}...")
    loaded_originals = [load_image(p) for p in orig_files]

    records: List[Dict[str, str]] = []

    defect_functions = {
        "blur": apply_blur,
        "dark": apply_dark,
        "bright": apply_bright,
        "lowres": apply_lowres,
    }

    for idx, (orig_path, orig_img) in enumerate(zip(orig_files, loaded_originals)):
        stem = orig_path.stem

        # 1. Original (good, source=original)
        orig_fname = f"{stem}_orig.jpg"
        orig_dest = output_images_dir / orig_fname
        cv2.imwrite(str(orig_dest), orig_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append({
            "filename": orig_fname,
            "label": "good",
            "reason": "",
            "source": "original",
        })

        # 2. Mild good variant (random crop up to 5%, brightness shift +-10%, quality 85)
        h, w = orig_img.shape[:2]
        crop_frac = random.uniform(0.0, 0.05)
        dh = int(round(h * crop_frac))
        dw = int(round(w * crop_frac))
        top = random.randint(0, dh) if dh > 0 else 0
        left = random.randint(0, dw) if dw > 0 else 0
        bottom = h - (dh - top)
        right = w - (dw - left)
        cropped = orig_img[top:bottom, left:right]

        bright_shift = random.uniform(0.9, 1.1)
        mild_img = np.clip(cropped.astype(float) * bright_shift, 0, 255).astype(np.uint8)

        mild_fname = f"{stem}_var_good.jpg"
        mild_dest = output_images_dir / mild_fname
        cv2.imwrite(str(mild_dest), mild_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        records.append({
            "filename": mild_fname,
            "label": "good",
            "reason": "",
            "source": "synthetic",
        })

        # 3. Two bad variants choosing 2 of 5 defects at random without repeating
        defect_choices = sorted(random.sample(["blur", "dark", "bright", "lowres", "clutter"], 2))
        for defect in defect_choices:
            if defect == "clutter":
                bad_img = apply_clutter(orig_img, loaded_originals, idx)
            else:
                bad_img = defect_functions[defect](orig_img)

            bad_fname = f"{stem}_bad_{defect}.jpg"
            bad_dest = output_images_dir / bad_fname
            cv2.imwrite(str(bad_dest), bad_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            records.append({
                "filename": bad_fname,
                "label": "bad",
                "reason": defect,
                "source": "synthetic",
            })

    # Write labels CSV
    fieldnames = ["filename", "label", "reason", "source"]
    with open(output_labels_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Print summary counts
    label_counts = Counter(r["label"] for r in records)
    source_counts = Counter(r["source"] for r in records)
    reason_counts = Counter(r["reason"] for r in records if r["reason"])

    print(f"\nWrote {len(records)} images to {output_images_dir}")
    print(f"Labels saved to: {output_labels_csv}")
    print("\n--- Evaluation Set Summary ---")
    print(f"Total Images: {len(records)}")
    print(f"Labels:  good={label_counts['good']}, bad={label_counts['bad']}")
    print(f"Sources: original={source_counts['original']}, synthetic={source_counts['synthetic']}")
    print(f"Reasons: {dict(sorted(reason_counts.items()))}")

    return records


def main():
    originals_dir = Path("data/coach_eval/originals/good")
    output_images_dir = Path("data/coach_eval/images")
    output_labels_csv = Path("data/coach_eval/labels.csv")

    if not originals_dir.exists():
        print(f"Error: Directory '{originals_dir}' does not exist.")
        print("Please run scripts/fetch_coach_photos.py first.")
        sys.exit(1)

    build_evaluation_set(originals_dir, output_images_dir, output_labels_csv, seed=42)


if __name__ == "__main__":
    main()
