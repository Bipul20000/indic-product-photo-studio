from pathlib import Path
import platform
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
import cv2
from studio.env import get_device, is_kaggle


def main():
    print(f"Python version: {sys.version.split()[0]}")
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Kaggle: {is_kaggle()}")
    device = get_device()
    print(f"Device: {device}")
    print(f"OpenCV version: {cv2.__version__}")

    gpu_name = "Not available"
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except (ImportError, Exception):
        pass
    print(f"GPU: {gpu_name}")


if __name__ == "__main__":
    main()
