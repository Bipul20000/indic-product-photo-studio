import csv
from collections import Counter
from pathlib import Path
import random
import shutil
import sys
from typing import Dict, List, Optional, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
import cv2
import numpy as np

from studio.coach.io import load_image


def load_excluded_stems(exclude_path: Optional[Path]) -> Set[str]:
    """Read configs/coach_eval_exclude.txt and extract excluded base stems."""
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
    """Check if filename matches any excluded stem prefix or exact name."""
    clean_stem = filename.replace("_orig.jpg", "").replace(".jpg", "").strip()
    for ex in excluded_stems:
        if clean_stem == ex or clean_stem.startswith(ex) or ex.startswith(clean_stem):
            return True
    return False


def apply_blur(img: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur with sigma 6 or motion blur of length 25 at random angle 0-180 deg."""
    pick = random.choice(["gaussian", "motion"])
    if pick == "gaussian":
        return cv2.GaussianBlur(img, (0, 0), sigmaX=6.0, sigmaY=6.0)
    else:
        angle = random.uniform(0.0, 180.0)
        theta = np.radians(angle)
        dx = 12.0 * np.cos(theta)
        dy = 12.0 * np.sin(theta)
        pt1 = (int(round(12 - dx)), int(round(12 - dy)))
        pt2 = (int(round(12 + dx)), int(round(12 + dy)))
        k = np.zeros((25, 25), dtype=np.float32)
        cv2.line(k, pt1, pt2, 1.0, thickness=1)
        s = float(k.sum())
        if s > 0:
            k /= s
        else:
            k[12, 12] = 1.0
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
    """Sample ~40 crops from other originals, keep top 25% by edge density, and paste 8-14 at alpha 1.0."""
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

    # Sample about 40 random crops from OTHER originals (each 12-30% of image width)
    crops_with_density: List[Tuple[float, np.ndarray]] = []
    for _ in range(40):
        donor_idx = random.choice(other_indices)
        donor_img = all_images[donor_idx]
        dh, dw = donor_img.shape[:2]

        sz = random.randint(max(10, int(w * 0.12)), max(10, int(w * 0.30)))
        sz = min(sz, dh, dw, h, w)

        src_y = random.randint(0, max(0, dh - sz))
        src_x = random.randint(0, max(0, dw - sz))
        crop = donor_img[src_y : src_y + sz, src_x : src_x + sz]

        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        edges = cv2.Canny(crop_gray, 50, 150)
        density = float(np.count_nonzero(edges) / (sz * sz))
        crops_with_density.append((density, crop))

    # Keep top 25% by density
    crops_with_density.sort(key=lambda x: x[0], reverse=True)
    top_count = max(1, len(crops_with_density) // 4)
    top_crops = [c[1] for c in crops_with_density[:top_count]]

    # Paste 8 to 14 of those into the outer 15% border band only, at alpha 1.0
    num_to_paste = random.randint(8, 14)
    for _ in range(num_to_paste):
        patch = random.choice(top_crops)
        ph, pw = patch.shape[:2]
        if ph > h or pw > w:
            ph, pw = min(ph, h), min(pw, w)
            patch = patch[:ph, :pw]

        side = random.choice(["top", "bottom", "left", "right"])
        if side == "top":
            y1 = 0
            x1 = random.randint(0, max(0, w - pw))
        elif side == "bottom":
            y1 = max(0, h - ph)
            x1 = random.randint(0, max(0, w - pw))
        elif side == "left":
            y1 = random.randint(0, max(0, h - ph))
            x1 = 0
        else:
            y1 = random.randint(0, max(0, h - ph))
            x1 = max(0, w - pw)

        roi = cluttered[y1 : y1 + ph, x1 : x1 + pw]
        roi_mask = border_mask[y1 : y1 + ph, x1 : x1 + pw]
        roi[roi_mask] = patch[roi_mask]

    return cluttered


def build_evaluation_set(
    originals_dir: Path,
    output_images_dir: Path,
    output_labels_csv: Path,
    exclude_file: Optional[Path] = None,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Build the Shoot Coach evaluation dataset deterministically with honest split and exclusions."""
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
    total_before = len(orig_files)

    # Filter exclusions
    excluded_stems = load_excluded_stems(exclude_file)
    if excluded_stems:
        orig_files = [p for p in orig_files if not is_excluded(p.name, excluded_stems)]
        print(f"Originals: {total_before} before exclusion -> {len(orig_files)} after ({total_before - len(orig_files)} excluded).")

    if not orig_files:
        print(f"No original images found in {originals_dir}.")
        return []

    # Deterministic 50/50 split of original filenames
    split_rng = random.Random(seed)
    orig_names = [p.name for p in orig_files]
    shuffled_names = list(orig_names)
    split_rng.shuffle(shuffled_names)
    half = len(shuffled_names) // 2
    tune_set = set(shuffled_names[:half])

    print(f"Loading {len(orig_files)} original images (tune: {len(tune_set)}, test: {len(orig_files) - len(tune_set)})...")
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
        split = "tune" if orig_path.name in tune_set else "test"

        # 1. Original (good, source=original)
        orig_fname = f"{stem}_orig.jpg"
        orig_dest = output_images_dir / orig_fname
        cv2.imwrite(str(orig_dest), orig_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append({
            "filename": orig_fname,
            "label": "good",
            "reason": "",
            "source": "original",
            "split": split,
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
            "split": split,
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
                "split": split,
            })

    # Write labels CSV
    fieldnames = ["filename", "label", "reason", "source", "split"]
    with open(output_labels_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Print summary counts
    label_counts = Counter(r["label"] for r in records)
    source_counts = Counter(r["source"] for r in records)
    split_counts = Counter(r["split"] for r in records)
    reason_counts = Counter(r["reason"] for r in records if r["reason"])

    print(f"\nWrote {len(records)} images to {output_images_dir}")
    print(f"Labels saved to: {output_labels_csv}")
    print("\n--- Evaluation Set Summary ---")
    print(f"Total Images: {len(records)}")
    print(f"Splits:  tune={split_counts['tune']}, test={split_counts['test']}")
    print(f"Labels:  good={label_counts['good']}, bad={label_counts['bad']}")
    print(f"Sources: original={source_counts['original']}, synthetic={source_counts['synthetic']}")
    print(f"Reasons: {dict(sorted(reason_counts.items()))}")

    return records


def main():
    originals_dir = Path("data/coach_eval/originals/good")
    output_images_dir = Path("data/coach_eval/images")
    output_labels_csv = Path("data/coach_eval/labels.csv")
    exclude_file = Path("configs/coach_eval_exclude.txt")

    if not originals_dir.exists():
        print(f"Error: Directory '{originals_dir}' does not exist.")
        print("Please run scripts/fetch_coach_photos.py first.")
        sys.exit(1)

    build_evaluation_set(originals_dir, output_images_dir, output_labels_csv, exclude_file=exclude_file, seed=42)


if __name__ == "__main__":
    main()
