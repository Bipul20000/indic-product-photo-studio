#!/usr/bin/env python3
"""Wrapper around official diffusers train_text_to_image_lora.py script.

Prepares training dataset, vendors the official example script matching the
installed diffusers version (falling back to main), builds the accelerate launch
command with optimized parameters, and validates output weights.
"""

import argparse
from pathlib import Path
import subprocess
import sys
from typing import List, Optional, Union
import urllib.request

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from studio.generate.config import GenConfig


def get_diffusers_version() -> Optional[str]:
    """Return installed diffusers version, or None if not installed."""
    try:
        import diffusers
        return getattr(diffusers, "__version__", None)
    except Exception:
        return None


def fetch_vendor_script(vendor_dir: Union[str, Path] = "scripts/vendor") -> Path:
    """Fetch official train_text_to_image_lora.py from Hugging Face diffusers repo.

    Attempts to fetch the script matching installed diffusers version.
    If that fails or diffusers is not installed, falls back to 'main' branch.
    Skips downloading if the script already exists for the target version.
    """
    vendor_p = Path(vendor_dir)
    vendor_p.mkdir(parents=True, exist_ok=True)
    target_file = vendor_p / "train_text_to_image_lora.py"

    diffusers_ver = get_diffusers_version()
    target_ver_str = diffusers_ver or "main"

    # Check if existing vendor script matches target version
    if target_file.exists():
        content = target_file.read_text(encoding="utf-8", errors="ignore")
        if f"# diffusers version: {target_ver_str}" in content:
            return target_file

    urls_to_try = []
    if diffusers_ver:
        urls_to_try.append((
            f"https://raw.githubusercontent.com/huggingface/diffusers/v{diffusers_ver}/examples/text_to_image/train_text_to_image_lora.py",
            diffusers_ver,
        ))
    urls_to_try.append((
        "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora.py",
        "main",
    ))

    fetched_code = None
    fetched_ver = None
    fetched_url = None

    for url, ver in urls_to_try:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    fetched_code = resp.read().decode("utf-8")
                    fetched_ver = ver
                    fetched_url = url
                    break
        except Exception as e:
            if ver != "main":
                print(f"Warning: Failed to fetch diffusers version v{ver} from {url} ({e}), trying main branch...", file=sys.stderr)

    if fetched_code is None:
        if target_file.exists():
            print(f"Warning: Could not download fresh script; using existing {target_file}", file=sys.stderr)
            return target_file
        raise RuntimeError("Failed to download train_text_to_image_lora.py from diffusers repo.")

    header = f"# Sourced from: {fetched_url}\n# diffusers version: {fetched_ver}\n\n"
    target_file.write_text(header + fetched_code, encoding="utf-8")
    print(f"Downloaded official training script to {target_file} (version: {fetched_ver})")
    return target_file


def has_checkpoint(output_dir: Union[str, Path]) -> bool:
    """Check if a checkpoint-* folder exists in output_dir."""
    out_p = Path(output_dir)
    if not out_p.exists() or not out_p.is_dir():
        return False
    return any(p.is_dir() and p.name.startswith("checkpoint-") for p in out_p.iterdir())


def build_train_command(
    train_data_dir: str = "data/lora_train_hf",
    output_dir: str = "models/festive_lora",
    max_steps: int = 1500,
    vendor_script: str = "scripts/vendor/train_text_to_image_lora.py",
    pretrained_model_name_or_path: Optional[str] = None,
) -> List[str]:
    """Build accelerate launch command with specified hyperparameters."""
    base_model = pretrained_model_name_or_path or GenConfig().base_models[0]

    cmd = [
        "accelerate",
        "launch",
        "--mixed_precision",
        "fp16",
        vendor_script,
        "--pretrained_model_name_or_path",
        base_model,
        "--train_data_dir",
        train_data_dir,
        "--image_column",
        "image",
        "--caption_column",
        "text",
        "--resolution",
        "512",
        "--random_flip",
        "--train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "4",
        "--gradient_checkpointing",
        "--max_train_steps",
        str(max_steps),
        "--learning_rate",
        "1e-4",
        "--lr_scheduler",
        "cosine",
        "--lr_warmup_steps",
        "100",
        "--rank",
        "16",
        "--mixed_precision",
        "fp16",
        "--seed",
        "42",
        "--checkpointing_steps",
        "250",
        "--checkpoints_total_limit",
        "2",
        "--dataloader_num_workers",
        "2",
        "--report_to",
        "none",
        "--output_dir",
        output_dir,
    ]

    if has_checkpoint(output_dir):
        cmd.extend(["--resume_from_checkpoint", "latest"])

    return cmd


def parse_args():
    parser = argparse.ArgumentParser(description="Train Festive LoRA with official diffusers example.")
    parser.add_argument("--max-steps", type=int, default=1500, help="Maximum training steps (default: 1500)")
    parser.add_argument("--output-dir", type=str, default="models/festive_lora", help="Output directory for weights")
    parser.add_argument("--train-data-dir", type=str, default="data/lora_train_hf", help="HF formatted dataset directory")
    parser.add_argument("--dry-run", action="store_true", help="Print exact training command and exit")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dry_run:
        cmd = build_train_command(
            train_data_dir=args.train_data_dir,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
        )
        print(" ".join(cmd))
        return 0

    # 1. Prepare HF dataset
    print("Step 1: Preparing LoRA training dataset...")
    from scripts.prepare_lora_training import prepare_dataset

    prepare_dataset(
        src_dir="data/lora_train",
        dst_dir=Path(args.train_data_dir) / "train",
    )

    # 2. Fetch official vendor script
    print("\nStep 2: Ensuring official diffusers training script is available...")
    vendor_script_path = fetch_vendor_script()

    # 3. Build training command
    cmd = build_train_command(
        train_data_dir=args.train_data_dir,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        vendor_script=str(vendor_script_path),
    )
    print("\nStep 3: Launching training command:")
    print(" ".join(cmd))
    print("-" * 60)

    # 4. Run training subprocess with streaming output
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if proc.stdout:
        for line in proc.stdout:
            print(line, end="")

    retcode = proc.wait()
    if retcode != 0:
        print(f"\nError: Training process exited with code {retcode}", file=sys.stderr)
        sys.exit(retcode)

    # 5. Check output weights
    weights_path = Path(args.output_dir) / "pytorch_lora_weights.safetensors"
    if weights_path.exists():
        size_mb = weights_path.stat().st_size / (1024 * 1024)
        print(f"\nTraining completed successfully! LoRA weights: {weights_path} ({size_mb:.2f} MB)")
    else:
        print(f"\nError: Expected LoRA weights file not found at: {weights_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
