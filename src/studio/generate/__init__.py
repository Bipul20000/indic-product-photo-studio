"""Background generation module with ControlNet Inpainting and scenes."""

from studio.generate.backgrounds import BackgroundGenerator, FakeBackgroundGenerator
from studio.generate.config import GenConfig, REQUIRED_SCENES, load_scenes
from studio.generate.prep import build_inpaint_inputs, gen_size, make_inpaint_condition

__all__ = [
    "BackgroundGenerator",
    "FakeBackgroundGenerator",
    "GenConfig",
    "REQUIRED_SCENES",
    "load_scenes",
    "build_inpaint_inputs",
    "gen_size",
    "make_inpaint_condition",
]
