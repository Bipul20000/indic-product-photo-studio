import hashlib
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from studio.generate.config import GenConfig
from studio.generate.prep import make_inpaint_condition


class FakeBackgroundGenerator:
    """Mock background generator for offline, local, or CPU testing without torch/GPU.

    Generates smooth deterministic two-color gradients matching the requested scene palette
    and seed, maintaining the exact interface of BackgroundGenerator.
    """

    def __init__(self, config: Optional[GenConfig] = None):
        self.config = config or GenConfig()
        self.loaded_model_id = "fake-gradient-generator"
        self._peak_vram_mb = 0.0

    @property
    def peak_vram_mb(self) -> float:
        """Peak VRAM allocation in MB (0.0 for CPU fake generator)."""
        return self._peak_vram_mb

    def generate(
        self,
        init_image: Image.Image,
        inpaint_mask: Image.Image,
        prompt: str,
        negative: str,
        seed: int = 0,
        controlnet_scale: Optional[float] = None,
        lora_scale: Optional[float] = None,
    ) -> Image.Image:
        """Generate a deterministic smooth gradient background image matching scene and seed."""
        w, h = init_image.size
        p_lower = prompt.lower()

        # Scene-specific palette selection
        if "diwali" in p_lower:
            c1 = (255, 145, 30)   # warm festive golden orange
            c2 = (60, 20, 10)     # dark polished wood
        elif "raksha" in p_lower:
            c1 = (255, 200, 210)  # soft rose petal pink
            c2 = (140, 30, 50)    # rich festive silk red
        elif "holi" in p_lower:
            c1 = (140, 220, 255)  # cheerful sky blue
            c2 = (255, 150, 210)  # gulal magenta
        elif "wedding" in p_lower:
            c1 = (240, 205, 120)  # marigold golden light
            c2 = (150, 20, 30)    # ornate deep red
        elif "indian_home" in p_lower or "home" in p_lower:
            c1 = (235, 215, 190)  # natural warm window light
            c2 = (110, 75, 45)    # warm teak wood
        elif "minimal" in p_lower:
            c1 = (248, 245, 238)  # clean off-white / light beige
            c2 = (215, 210, 200)  # subtle studio shadow
        else:
            # Deterministic hash-based gradient
            seed_hash = int(hashlib.md5(f"{prompt}_{seed}".encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed_hash % (2**31 - 1))
            c1 = tuple(int(x) for x in rng.randint(180, 255, size=3))
            c2 = tuple(int(x) for x in rng.randint(40, 160, size=3))

        # Vary slightly by seed
        s_offset = (seed * 17) % 30
        c1 = tuple(max(0, min(255, val + (s_offset if i % 2 == 0 else -s_offset))) for i, val in enumerate(c1))

        # Build vertical linear gradient
        y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
        top = np.array(c1, dtype=np.float32)
        bottom = np.array(c2, dtype=np.float32)
        grad = (1.0 - y) * top + y * bottom
        grad_uint8 = np.clip(grad, 0, 255).astype(np.uint8)

        return Image.fromarray(grad_uint8, mode="RGB")


class BackgroundGenerator:
    """ControlNet Inpainting Background Generator with SD 1.5.

    Requires an active CUDA GPU (e.g. Kaggle T4/P100). Torch and diffusers are
    lazily imported inside the class methods so that importing studio modules
    never raises an ImportError locally.
    """

    _cached_pipe = None
    _cached_model_id = None

    def __init__(self, config: Optional[GenConfig] = None):
        self.config = config or GenConfig()
        self.pipe = None
        self.loaded_model_id: Optional[str] = None
        self._peak_vram_mb: float = 0.0

    @property
    def peak_vram_mb(self) -> float:
        """Peak GPU VRAM allocation in MB."""
        try:
            import torch
            if torch.cuda.is_available():
                return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        except Exception:
            pass
        return self._peak_vram_mb

    def _ensure_pipeline(self) -> None:
        """Lazily initialize and cache the ControlNet Inpaint pipeline on CUDA."""
        if BackgroundGenerator._cached_pipe is not None:
            self.pipe = BackgroundGenerator._cached_pipe
            self.loaded_model_id = BackgroundGenerator._cached_model_id
            return

        try:
            import torch
        except ImportError as e:
            raise RuntimeError("Scene generation needs PyTorch installed with GPU support. Run it on Kaggle.") from e

        if not torch.cuda.is_available():
            raise RuntimeError("Scene generation needs a GPU. Run it on Kaggle.")

        from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionControlNetInpaintPipeline

        controlnet = ControlNetModel.from_pretrained(
            self.config.controlnet_id,
            torch_dtype=torch.float16,
        )

        pipe = None
        loaded_id = None
        last_error = None

        for base_model in self.config.base_models:
            try:
                print(f"Attempting to load diffusion base model: {base_model}...")
                pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                    base_model,
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
                loaded_id = base_model
                print(f"Successfully loaded diffusion base model: {loaded_id}")
                break
            except Exception as e:
                print(f"Failed to load {base_model}: {e}")
                last_error = e

        if pipe is None:
            raise RuntimeError(f"Could not load any base model from {self.config.base_models}. Last error: {last_error}")

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")

        # Optional LoRA weights
        if self.config.lora_path:
            print(f"Loading LoRA weights from: {self.config.lora_path}")
            pipe.load_lora_weights(self.config.lora_path)

        BackgroundGenerator._cached_pipe = pipe
        BackgroundGenerator._cached_model_id = loaded_id

        self.pipe = pipe
        self.loaded_model_id = loaded_id

    def generate(
        self,
        init_image: Image.Image,
        inpaint_mask: Image.Image,
        prompt: str,
        negative: str,
        seed: int = 0,
        controlnet_scale: Optional[float] = None,
        lora_scale: Optional[float] = None,
    ) -> Image.Image:
        """Run ControlNet inpainting pipeline to generate background.

        Args:
            init_image: Scaled PIL RGB image.
            inpaint_mask: PIL L mask (255 = area to inpaint, 0 = keep).
            prompt: Text prompt with product description.
            negative: Shared negative prompt.
            seed: Integer random seed.

        Returns:
            Generated PIL RGB image at generation resolution.
        """
        self._ensure_pipeline()
        import torch

        # Build inpaint condition tensor
        cond_np = make_inpaint_condition(init_image, inpaint_mask)
        cond_tensor = torch.from_numpy(cond_np).to(dtype=torch.float16, device="cuda")

        generator = torch.Generator(device="cuda").manual_seed(seed)

        cn_scale = self.config.controlnet_scale if controlnet_scale is None else float(controlnet_scale)

        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative,
            "image": init_image,
            "mask_image": inpaint_mask,
            "control_image": cond_tensor,
            "num_inference_steps": self.config.steps,
            "guidance_scale": self.config.guidance,
            "eta": self.config.eta,
            "controlnet_conditioning_scale": cn_scale,
            "generator": generator,
        }

        if self.config.lora_path:
            eff_scale = self.config.lora_scale if lora_scale is None else float(lora_scale)
            if eff_scale is None:
                eff_scale = 0.8

            if eff_scale <= 0.0:
                if hasattr(self.pipe, "disable_lora"):
                    self.pipe.disable_lora()
                elif hasattr(self.pipe, "set_adapters"):
                    try:
                        self.pipe.set_adapters(["default"], [0.0])
                    except Exception:
                        pass
            else:
                if hasattr(self.pipe, "enable_lora"):
                    self.pipe.enable_lora()
                if hasattr(self.pipe, "set_adapters"):
                    try:
                        self.pipe.set_adapters(["default"], [float(eff_scale)])
                    except Exception:
                        try:
                            self.pipe.set_adapters(adapter_weights=[float(eff_scale)])
                        except Exception:
                            pass

        output = self.pipe(**call_kwargs)
        result_pil = output.images[0]

        self._peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        return result_pil
