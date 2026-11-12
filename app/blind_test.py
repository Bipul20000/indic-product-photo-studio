#!/usr/bin/env python3
"""Blind evaluation tool for comparing Base vs LoRA generated backgrounds.

Presents randomized A/B image pairs with randomized sides to anonymous raters,
collects forced-choice authenticity verdicts, and records results thread-safely.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import cv2
import gradio as gr
import numpy as np
from PIL import Image

CSV_LOCK = threading.Lock()

DEFAULT_SCENE_PHRASES = {
    "diwali": "Diwali celebration",
    "raksha_bandhan": "Raksha Bandhan celebration",
    "wedding": "Indian wedding",
    "indian_home": "Indian home",
}


def compute_winner(choice: str, left_variant: str) -> str:
    """Determine winner ('lora', 'base', 'tie') from choice and left_variant."""
    c_lower = choice.strip().lower()
    lv_lower = left_variant.strip().lower()

    if c_lower in ("no clear difference", "tie", "none", "neither"):
        return "tie"
    if c_lower in ("left", "l"):
        return lv_lower
    if c_lower in ("right", "r"):
        return "lora" if lv_lower == "base" else "base"
    return "tie"


def append_response(
    csv_path: Union[str, Path],
    rater_id: str,
    pair_id: str,
    scene: str,
    left_variant: str,
    choice: str,
    winner: str,
    timestamp: Optional[str] = None,
    lock: Optional[threading.Lock] = None,
) -> None:
    """Thread-safe append of a single rater response to CSV."""
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    use_lock = lock if lock is not None else CSV_LOCK

    with use_lock:
        file_exists = p.exists() and p.stat().st_size > 0
        with open(p, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "rater_id",
                    "pair_id",
                    "scene",
                    "left_variant",
                    "choice",
                    "winner",
                    "timestamp",
                ],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "rater_id": rater_id,
                "pair_id": pair_id,
                "scene": scene,
                "left_variant": left_variant,
                "choice": choice,
                "winner": winner,
                "timestamp": ts,
            })


def load_pairs(pairs_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load evaluation pairs from JSON file."""
    p = Path(pairs_path)
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def load_image_rgb(img_path: Union[str, Path]) -> Optional[np.ndarray]:
    """Read image as RGB numpy array to avoid exposing filenames in UI."""
    p = Path(img_path)
    if not p.exists():
        return None
    bgr = cv2.imread(str(p))
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_app(
    pairs_file: Union[str, Path] = "configs/blind_pairs.json",
    csv_path: Union[str, Path] = "results/blind_test_responses.csv",
) -> gr.Blocks:
    """Build the Blind A/B Evaluation Gradio Blocks app."""
    raw_pairs = load_pairs(pairs_file)

    with gr.Blocks(title="Indic Product Photo Studio — Blind Study") as demo:
        # Session state for each user
        session_state = gr.State({
            "rater_id": "",
            "pairs": [],
            "idx": 0,
            "total": 0,
        })

        gr.Markdown("# Indic Product Photo Studio — Background Authenticity Study")

        # 1. Intro Screen
        with gr.Column(visible=True) as intro_view:
            gr.Markdown(
                "### Welcome to the background authenticity evaluation.\n"
                "In this study, you will be shown pairs of product images with different generated backgrounds. "
                "Your task is to judge which background looks more authentic and natural for the indicated Indian festive setting."
            )
            start_btn = gr.Button("Start Evaluation", variant="primary", size="lg")

        # 2. Evaluation Screen
        with gr.Column(visible=False) as eval_view:
            progress_text = gr.Markdown("### Pair 1 of N")
            question_text = gr.Markdown("## Which background looks more like an authentic setting?")

            with gr.Row():
                with gr.Column(scale=1):
                    left_image = gr.Image(
                        type="numpy",
                        label="Option A",
                        interactive=False,
                        width=480,
                    )
                    btn_left = gr.Button("Left (Option A)", variant="secondary", size="lg")

                with gr.Column(scale=1):
                    right_image = gr.Image(
                        type="numpy",
                        label="Option B",
                        interactive=False,
                        width=480,
                    )
                    btn_right = gr.Button("Right (Option B)", variant="secondary", size="lg")

            with gr.Row():
                btn_tie = gr.Button("No clear difference", variant="secondary", size="md")

        # 3. Completion Screen
        with gr.Column(visible=False) as done_view:
            gr.Markdown(
                "## 🎉 Thank You!\n"
                "Your evaluations have been recorded. Thank you for contributing to this research study."
            )

        # Helper to load pair at given index
        def show_current_pair(state):
            idx = state["idx"]
            pairs = state["pairs"]
            if idx >= len(pairs):
                return (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    "",
                    "",
                    None,
                    None,
                    state,
                )

            item = pairs[idx]
            scene = item["scene"]
            phrase = DEFAULT_SCENE_PHRASES.get(scene, scene.replace("_", " ").title())
            q_text = f"## Which background looks more like an authentic {phrase}?"
            prog_text = f"### Pair {idx + 1} of {len(pairs)}"

            left_rgb = load_image_rgb(item["left_path"])
            right_rgb = load_image_rgb(item["right_path"])

            # Fallback if image missing on disk
            if left_rgb is None:
                left_rgb = np.full((480, 480, 3), 230, dtype=np.uint8)
            if right_rgb is None:
                right_rgb = np.full((480, 480, 3), 230, dtype=np.uint8)

            return (
                gr.update(visible=True),
                gr.update(visible=False),
                prog_text,
                q_text,
                left_rgb,
                right_rgb,
                state,
            )

        # Start button click handler
        def on_start():
            # Refresh pairs from disk
            loaded = load_pairs(pairs_file)
            if not loaded:
                loaded = raw_pairs

            # Shuffle pairs for this rater
            assigned = []
            shuffled = list(loaded)
            random.shuffle(shuffled)

            for p in shuffled:
                coin = random.choice([True, False])
                assigned.append({
                    "pair_id": p.get("pair_id", ""),
                    "scene": p.get("scene", ""),
                    "product": p.get("product", ""),
                    "seed": p.get("seed", 0),
                    "left_variant": "lora" if coin else "base",
                    "left_path": p.get("lora_path" if coin else "base_path", ""),
                    "right_path": p.get("base_path" if coin else "lora_path", ""),
                })

            new_state = {
                "rater_id": str(uuid.uuid4()),
                "pairs": assigned,
                "idx": 0,
                "total": len(assigned),
            }

            if not assigned:
                # No pairs available
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    "No evaluation pairs found.",
                    "",
                    None,
                    None,
                    new_state,
                )

            eval_vis, done_vis, prog_text, q_text, l_img, r_img, updated_state = show_current_pair(new_state)
            return (
                gr.update(visible=False),  # Hide intro
                eval_vis,
                done_vis,
                prog_text,
                q_text,
                l_img,
                r_img,
                updated_state,
            )

        start_btn.click(
            fn=on_start,
            inputs=[],
            outputs=[intro_view, eval_view, done_view, progress_text, question_text, left_image, right_image, session_state],
        )

        # Choice button click handler
        def on_choice(choice_str, state):
            idx = state["idx"]
            pairs = state["pairs"]

            if idx < len(pairs):
                current_pair = pairs[idx]
                left_var = current_pair["left_variant"]
                winner = compute_winner(choice_str, left_var)

                # Record response
                append_response(
                    csv_path=csv_path,
                    rater_id=state["rater_id"],
                    pair_id=current_pair["pair_id"],
                    scene=current_pair["scene"],
                    left_variant=left_var,
                    choice=choice_str,
                    winner=winner,
                )

                state["idx"] = idx + 1

            eval_vis, done_vis, prog_text, q_text, l_img, r_img, updated_state = show_current_pair(state)
            return (
                eval_vis,
                done_vis,
                prog_text,
                q_text,
                l_img,
                r_img,
                updated_state,
            )

        btn_left.click(
            fn=lambda st: on_choice("Left", st),
            inputs=[session_state],
            outputs=[eval_view, done_view, progress_text, question_text, left_image, right_image, session_state],
        )

        btn_right.click(
            fn=lambda st: on_choice("Right", st),
            inputs=[session_state],
            outputs=[eval_view, done_view, progress_text, question_text, left_image, right_image, session_state],
        )

        btn_tie.click(
            fn=lambda st: on_choice("No clear difference", st),
            inputs=[session_state],
            outputs=[eval_view, done_view, progress_text, question_text, left_image, right_image, session_state],
        )

    demo.queue()
    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Blind test Gradio evaluation app.")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    parser.add_argument("--port", type=int, default=7861, help="Port to run on (default: 7861)")
    parser.add_argument("--pairs", type=str, default="configs/blind_pairs.json", help="Path to blind pairs JSON")
    parser.add_argument("--responses", type=str, default="results/blind_test_responses.csv", help="Destination responses CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    demo = build_app(pairs_file=args.pairs, csv_path=args.responses)
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
