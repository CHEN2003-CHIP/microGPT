"""Small shared helpers for the microGPT teaching project."""

from __future__ import annotations

import os
import torch


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _detect_compute_dtype():
    env_value = os.environ.get("MICROGPT_DTYPE")
    if env_value in _DTYPE_MAP:
        return _DTYPE_MAP[env_value], f"set via MICROGPT_DTYPE={env_value}"
    if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0):
        return torch.bfloat16, "auto-detected: Ampere-or-newer CUDA GPU"
    return torch.float32, "auto-detected: float32 fallback"


COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()


def get_base_dir():
    """
    Store all generated artifacts in a predictable local directory by default.
    This keeps the teaching project easy to reason about on Windows and laptops.
    """
    if os.environ.get("MICROGPT_BASE_DIR"):
        base_dir = os.environ["MICROGPT_BASE_DIR"]
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_dir = os.path.join(repo_root, ".microchat")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def get_hf_cache_dir():
    """Keep Hugging Face downloads inside the project instead of C:\\Users."""
    cache_dir = os.path.join(get_base_dir(), "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def configure_local_hf_cache():
    """
    Hugging Face defaults to the user profile on Windows. Set these before
    importing datasets/huggingface_hub so teaching runs stay project-local.
    """
    cache_dir = get_hf_cache_dir()
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_dir, "datasets"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(cache_dir, "hub"))
    return cache_dir


def print0(*args, **kwargs):
    """Compatibility helper from the original project. Single-device only now."""
    print(*args, **kwargs)


def autodetect_device_type():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_init(device_type=""):
    """
    Small teaching-friendly device setup.
    Returns the old tuple shape for compatibility with the scripts.
    """
    resolved = autodetect_device_type() if device_type == "" else device_type
    if resolved == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but not available"
    if resolved == "mps":
        assert torch.backends.mps.is_available(), "MPS requested but not available"
    torch.manual_seed(42)
    if resolved == "cuda":
        torch.cuda.manual_seed(42)
        torch.set_float32_matmul_precision("high")
    return False, 0, 0, 1, torch.device(resolved)


def compute_cleanup():
    """No-op kept for a stable training-script interface."""
    return None


def get_dist_info():
    """Single-device teaching build: no distributed execution."""
    return False, 0, 0, 1


def is_ddp_initialized():
    return False
