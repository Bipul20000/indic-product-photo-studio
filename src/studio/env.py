import os
from pathlib import Path


def is_kaggle() -> bool:
    """Return True if running inside a Kaggle notebook environment."""
    return Path("/kaggle").exists()


def get_device() -> str:
    """Return 'cuda' if torch is available and CUDA is working, otherwise 'cpu'."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
    except (ImportError, Exception):
        pass
    return "cpu"


def get_output_dir() -> Path:
    """Return output directory: /kaggle/working/outputs on Kaggle, ./outputs locally."""
    if is_kaggle():
        return Path("/kaggle/working/outputs")
    return Path("./outputs")
