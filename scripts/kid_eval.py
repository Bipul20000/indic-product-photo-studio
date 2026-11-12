#!/usr/bin/env python3
"""Evaluate Kernel Inception Distance (KID) and Fréchet Inception Distance (FID).

Compares real dataset images from data/lora_train/{diwali,raksha_bandhan,wedding,indian_home}
against generated outputs from outputs/compare/{base,lora}/ for those four scenes.
Uses torchmetrics with lazy imports, resized to 299x299 on CPU.
Outputs summary to outputs/kid_fid_summary.csv.
"""

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
from PIL import Image

EVAL_SCENES = ["diwali", "raksha_bandhan", "wedding", "indian_home"]


def collect_images_for_scenes(base_dir: Union[str, Path], scenes: List[str]) -> List[Path]:
    """Find all jpg/jpeg image paths in subdirectories matching given scenes."""
    p = Path(base_dir)
    images = []
    for sc in scenes:
        sc_dir = p / sc
        if sc_dir.exists() and sc_dir.is_dir():
            for f in sorted(sc_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    images.append(f)
    return images


def load_image_batch(paths: List[Path], target_size: int = 299) -> Any:
    """Load and resize images to target_size x target_size as uint8 torch Tensor (B, 3, H, W)."""
    import torch

    tensors = []
    for path in paths:
        try:
            with Image.open(path) as img:
                img_rgb = img.convert("RGB").resize((target_size, target_size), Image.BILINEAR)
                arr = np.array(img_rgb, dtype=np.uint8)  # (H, W, 3)
                t = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
                tensors.append(t)
        except Exception as e:
            print(f"Warning: Could not read image {path}: {e}", file=sys.stderr)

    if not tensors:
        return torch.empty((0, 3, target_size, target_size), dtype=torch.uint8)
    return torch.stack(tensors, dim=0)


def evaluate_kid_fid(
    real_images: List[Path],
    fake_images: List[Path],
    batch_size: int = 16,
) -> Tuple[float, float, float]:
    """Calculate KID (mean, std) and FID between real and fake image sets.

    Returns:
        (kid_mean, kid_std, fid_score)
    """
    try:
        import torch
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except ImportError as e:
        print(f"Error: torchmetrics[image] or torch-fidelity is not installed: {e}", file=sys.stderr)
        print("Install them with: pip install torchmetrics[image] torch-fidelity", file=sys.stderr)
        raise

    n_fake = len(fake_images)
    if n_fake == 0 or len(real_images) == 0:
        raise ValueError(f"Need non-empty image lists: {len(real_images)} real, {n_fake} fake.")

    subset_size = min(30, n_fake)

    kid = KernelInceptionDistance(subsets=50, subset_size=subset_size)
    fid = FrechetInceptionDistance()

    # Feed real images in batches
    for i in range(0, len(real_images), batch_size):
        batch = load_image_batch(real_images[i : i + batch_size])
        if len(batch) > 0:
            kid.update(batch, real=True)
            fid.update(batch, real=True)

    # Feed fake images in batches
    for i in range(0, len(fake_images), batch_size):
        batch = load_image_batch(fake_images[i : i + batch_size])
        if len(batch) > 0:
            kid.update(batch, real=False)
            fid.update(batch, real=False)

    kid_mean_t, kid_std_t = kid.compute()
    fid_t = fid.compute()

    return float(kid_mean_t.item()), float(kid_std_t.item()), float(fid_t.item())


def run_kid_fid_eval(
    real_dir: Union[str, Path] = "data/lora_train",
    compare_dir: Union[str, Path] = "outputs/compare",
    out_csv: Union[str, Path] = "outputs/kid_fid_summary.csv",
) -> List[Dict[str, Any]]:
    """Run pooled KID and FID evaluation for base and LoRA variants."""
    real_paths = collect_images_for_scenes(real_dir, EVAL_SCENES)
    base_paths = collect_images_for_scenes(Path(compare_dir) / "base", EVAL_SCENES)
    lora_paths = collect_images_for_scenes(Path(compare_dir) / "lora", EVAL_SCENES)

    print(f"Found {len(real_paths)} real images across {EVAL_SCENES} in {real_dir}")
    print(f"Found {len(base_paths)} Base generated images in {compare_dir}/base")
    print(f"Found {len(lora_paths)} LoRA generated images in {compare_dir}/lora")

    if not real_paths:
        print(f"Error: No real images found in {real_dir} for scenes: {EVAL_SCENES}", file=sys.stderr)
        return []
    if not base_paths or not lora_paths:
        print(f"Error: Missing generated images in {compare_dir} (base: {len(base_paths)}, lora: {len(lora_paths)})", file=sys.stderr)
        return []

    results = []

    print("\nComputing KID and FID for Base model...")
    base_kid_mean, base_kid_std, base_fid = evaluate_kid_fid(real_paths, base_paths)
    results.append({
        "variant": "base",
        "n_real": len(real_paths),
        "n_generated": len(base_paths),
        "kid_mean": round(base_kid_mean, 5),
        "kid_std": round(base_kid_std, 5),
        "fid": round(base_fid, 3),
    })

    print("Computing KID and FID for LoRA model...")
    lora_kid_mean, lora_kid_std, lora_fid = evaluate_kid_fid(real_paths, lora_paths)
    results.append({
        "variant": "lora",
        "n_real": len(real_paths),
        "n_generated": len(lora_paths),
        "kid_mean": round(lora_kid_mean, 5),
        "kid_std": round(lora_kid_std, 5),
        "fid": round(lora_fid, 3),
    })

    print("\n" + "=" * 68)
    print("KID / FID Evaluation Summary (Pooled across 4 scenes)")
    print("=" * 68)
    for r in results:
        print(
            f"Variant: {r['variant']:<6} | Real: {r['n_real']} | Gen: {r['n_generated']} | "
            f"KID: {r['kid_mean']:.5f} ± {r['kid_std']:.5f} | FID: {r['fid']:.2f}"
        )
    print("=" * 68)
    print("Warning: FID is unreliable with fewer than ~100 images")

    out_p = Path(out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "n_real", "n_generated", "kid_mean", "kid_std", "fid"],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSummary metrics written to: {out_p}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate KID and FID metrics for Base vs LoRA.")
    parser.add_argument("--real-dir", type=str, default="data/lora_train", help="Directory of real reference images")
    parser.add_argument("--compare-dir", type=str, default="outputs/compare", help="Directory with base/ and lora/ subdirs")
    parser.add_argument("--out", type=str, default="outputs/kid_fid_summary.csv", help="Destination summary CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        run_kid_fid_eval(
            real_dir=args.real_dir,
            compare_dir=args.compare_dir,
            out_csv=args.out,
        )
    except Exception as e:
        print(f"Error during KID/FID evaluation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
