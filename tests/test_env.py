from pathlib import Path
import pytest
from studio.env import get_device, get_output_dir, is_kaggle


def test_is_kaggle():
    result = is_kaggle()
    assert isinstance(result, bool)


def test_get_device():
    device = get_device()
    assert device in ("cpu", "cuda")


def test_get_output_dir():
    out_dir = get_output_dir()
    assert isinstance(out_dir, Path)
    if is_kaggle():
        assert str(out_dir) == "/kaggle/working/outputs"
    else:
        assert out_dir == Path("./outputs")
