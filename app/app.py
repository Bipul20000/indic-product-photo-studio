"""Indic Product Photo Studio Gradio App.

Working prototype demonstrating end-to-end product photo evaluation, mask inspection,
and marketplace background generation with Shoot Coach gating.
"""

import argparse
from typing import Any, Optional

import gradio as gr
import torch

from studio.app_logic import check_photo, generate_all, generate_one
from studio.compose.compose import PRESETS


def build_app(
    fake_generator: bool = False,
    default_preset: str = "square_1000",
    generator: Optional[Any] = None,
    lora: Optional[str] = None,
    lora_scale: float = 0.8,
) -> gr.Blocks:
    """Build Gradio Blocks UI and connect event handlers."""
    cuda_available = torch.cuda.is_available()
    is_demo_mode = fake_generator or not cuda_available

    _cached_generator = generator

    def get_generator():
        nonlocal _cached_generator
        if _cached_generator is not None:
            return _cached_generator

        from studio.generate.config import GenConfig

        gen_cfg = GenConfig(lora_path=lora, lora_scale=lora_scale)

        if is_demo_mode:
            from studio.generate.backgrounds import FakeBackgroundGenerator

            _cached_generator = FakeBackgroundGenerator(config=gen_cfg)
        else:
            gr.Info("Loading models, first run takes a minute...")
            from studio.generate.backgrounds import BackgroundGenerator

            _cached_generator = BackgroundGenerator(config=gen_cfg)
        return _cached_generator

    with gr.Blocks(title="Indic Product Photo Studio") as demo:
        gr.Markdown("# Indic Product Photo Studio")
        gr.Markdown("Upload a phone photo of your product, get a marketplace-ready image in a festive Indian scene.")

        if is_demo_mode:
            gr.Markdown("### ⚠️ Demo mode: no GPU, backgrounds are plain gradients")

        with gr.Row():
            # Left column: Inputs
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Upload Product Photo")
                scene_input = gr.Dropdown(
                    choices=["white", "diwali", "raksha_bandhan", "holi", "wedding", "indian_home", "minimal"],
                    value="diwali",
                    label="Scene",
                )
                marketplace_input = gr.Dropdown(
                    choices=list(PRESETS.keys()),
                    value=default_preset if default_preset in PRESETS else "square_1000",
                    label="Marketplace / Preset",
                )
                product_name_input = gr.Textbox(
                    value="",
                    label="Product Name (Optional)",
                    placeholder="e.g. ceramic mug, brass diya, leather bag",
                )
                seed_input = gr.Number(value=0, precision=0, label="Seed")
                force_input = gr.Checkbox(
                    value=False,
                    label="Generate even if Shoot Coach says retake",
                )

                with gr.Row():
                    btn_check = gr.Button("Check my photo", variant="secondary")
                    btn_generate = gr.Button("Generate", variant="primary")
                btn_generate_all = gr.Button("Generate all scenes", variant="secondary")

            # Right column: Outputs
            with gr.Column(scale=1):
                status_output = gr.Textbox(label="Coach Status", interactive=False)
                tips_output = gr.Textbox(label="Coach Tips", lines=3, interactive=False)
                gen_info_output = gr.Textbox(label="Seconds Taken", interactive=False)

                image_output = gr.Image(
                    type="numpy",
                    label="Generated Image",
                    interactive=False,
                    format="png",
                )
                gallery_output = gr.Gallery(
                    label="All Scenes Gallery",
                    show_label=True,
                    columns=3,
                    height="auto",
                )

                with gr.Accordion("Coach Quality Checks & Product Mask", open=True):
                    checks_table_output = gr.Dataframe(
                        headers=["check", "passed", "value"],
                        label="Quality Checks",
                        interactive=False,
                    )
                    mask_output = gr.Image(
                        type="numpy",
                        label="Product Mask Overlay (Red 40%)",
                        interactive=False,
                    )

        # Event Handlers
        def on_check(image_path):
            try:
                res = check_photo(image_path)
                return res.verdict, res.tips, res.table, res.mask_overlay
            except Exception as exc:
                return "ERROR", f"Error checking photo: {exc}", [], None

        def on_generate(image_path, scene, marketplace, product_name, seed, force_retake):
            try:
                gen = get_generator()
                res = generate_one(
                    image_path=image_path,
                    scene=scene,
                    marketplace=marketplace,
                    product_name=product_name,
                    seed=seed,
                    force_retake=force_retake,
                    generator=gen,
                )
                sec_text = f"{res.seconds:.2f}s" if res.seconds > 0 else ""
                return res.image, res.verdict, res.tips, sec_text
            except Exception as exc:
                return None, "ERROR", f"Error during generation: {exc}", ""

        def on_generate_all(image_path, marketplace, product_name, seed, force_retake, progress=gr.Progress()):
            try:
                gen = get_generator()
                res = generate_all(
                    image_path=image_path,
                    marketplace=marketplace,
                    product_name=product_name,
                    seed=seed,
                    force_retake=force_retake,
                    generator=gen,
                    progress=progress,
                )
                sec_text = f"Total: {res.seconds:.2f}s" if res.seconds > 0 else ""
                return res.gallery, res.verdict, res.tips, sec_text
            except Exception as exc:
                return [], "ERROR", f"Error generating all scenes: {exc}", ""

        btn_check.click(
            fn=on_check,
            inputs=[image_input],
            outputs=[status_output, tips_output, checks_table_output, mask_output],
        )

        btn_generate.click(
            fn=on_generate,
            inputs=[image_input, scene_input, marketplace_input, product_name_input, seed_input, force_input],
            outputs=[image_output, status_output, tips_output, gen_info_output],
        )

        btn_generate_all.click(
            fn=on_generate_all,
            inputs=[image_input, marketplace_input, product_name_input, seed_input, force_input],
            outputs=[gallery_output, status_output, tips_output, gen_info_output],
        )

    demo.queue(default_concurrency_limit=1)
    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Indic Product Photo Studio Gradio App")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on (default 7860)")
    parser.add_argument("--fake-generator", action="store_true", help="Use FakeBackgroundGenerator instead of GPU diffusion model")
    parser.add_argument("--preset", type=str, default="square_1000", help="Default canvas preset (default square_1000)")
    parser.add_argument("--lora", type=str, default=None, help="Optional path to LoRA weights")
    parser.add_argument("--lora-scale", type=float, default=0.8, help="LoRA cross-attention scale (default 0.8)")
    return parser.parse_args()


def main():
    args = parse_args()
    demo = build_app(
        fake_generator=args.fake_generator,
        default_preset=args.preset,
        lora=args.lora,
        lora_scale=args.lora_scale,
    )
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
