#!/usr/bin/env python3
"""Prepare LoRA training dataset for Hugging Face diffusers imagefolder format.

Reads image-caption pairs from data/lora_train/<theme>/*.jpg and *.txt,
and writes them to data/lora_train_hf/train/ with <theme>_<NNN>.jpg filenames
and a metadata.jsonl index file.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Dict, List, Union


def prepare_dataset(
    src_dir: Union[str, Path] = "data/lora_train",
    dst_dir: Union[str, Path] = "data/lora_train_hf/train",
) -> Dict[str, int]:
    """Prepare HF imagefolder dataset from local lora_train directory.

    Args:
        src_dir: Path to source data directory containing theme subdirectories.
        dst_dir: Destination train directory for copied images and metadata.jsonl.

    Returns:
        Dictionary mapping theme names to image counts.

    Raises:
        FileNotFoundError: If src_dir does not exist or is not a directory.
    """
    src_path = Path(src_dir)
    if not src_path.exists() or not src_path.is_dir():
        raise FileNotFoundError(f"LoRA training source directory not found: {src_path}")

    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    theme_dirs = sorted([p for p in src_path.iterdir() if p.is_dir() and not p.name.startswith(".")])
    if not theme_dirs:
        raise ValueError(f"No theme directories found in: {src_path}")

    counts: Dict[str, int] = {}
    metadata_entries: List[Dict[str, str]] = []

    for theme_dir in theme_dirs:
        theme_name = theme_dir.name
        # Find all jpg / jpeg files
        img_files = sorted(
            [p for p in theme_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}]
        )

        theme_count = 0
        for i, img_p in enumerate(img_files, start=1):
            out_filename = f"{theme_name}_{i:03d}.jpg"
            dest_img_p = dst_path / out_filename

            # Copy image
            shutil.copy2(img_p, dest_img_p)

            # Read matching caption
            cap_p = img_p.with_suffix(".txt")
            if cap_p.exists():
                caption = cap_p.read_text(encoding="utf-8").strip()
            else:
                caption = ""

            metadata_entries.append({
                "file_name": out_filename,
                "text": caption,
            })
            theme_count += 1

        counts[theme_name] = theme_count
        print(f"Theme '{theme_name}': {theme_count} images")

    # Write metadata.jsonl
    metadata_file = dst_path / "metadata.jsonl"
    with open(metadata_file, "w", encoding="utf-8") as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total_count = sum(counts.values())
    print(f"Total: {total_count} images copied to {dst_path}")
    print(f"Metadata index written to: {metadata_file}")

    return counts


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for Hugging Face LoRA training.")
    parser.add_argument("--src", type=str, default="data/lora_train", help="Source directory with theme subfolders.")
    parser.add_argument("--dst", type=str, default="data/lora_train_hf/train", help="Destination train directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        prepare_dataset(src_dir=args.src, dst_dir=args.dst)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error preparing LoRA training data: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
