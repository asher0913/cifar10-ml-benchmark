"""Shared utilities for the ML experiment suite package."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np


def get_torch_device(require_gpu: bool = True) -> Tuple["torch.device", str]:
    """Pick CUDA, MPS, or CPU depending on availability and requirements."""
    import torch

    # Prefer CUDA when available; fall back to Apple MPS; otherwise CPU.
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    if require_gpu:
        raise RuntimeError("No CUDA or MPS device detected; a GPU is required to run these experiments.")
    return torch.device("cpu"), "cpu"


def ensure_dir(path: Path) -> Path:
    """Create directory (including parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Path) -> None:
    """Write a Python object as pretty-printed JSON to disk."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2))


def set_global_seed(seed: int) -> None:
    """Set RNG seeds for Python and NumPy to make runs reproducible."""
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class DeviceContext:
    device: "torch.device"
    backend: str
    pin_memory: bool
    non_blocking: bool
    # pin_memory/non_blocking flags are chosen to speed up host->device transfer.


def get_device_context(require_gpu: bool = True) -> DeviceContext:
    """Return device info plus memory flags for DataLoader friendliness."""
    import torch

    device, backend = get_torch_device(require_gpu)
    pin_memory = backend == "cuda"  # only meaningful for CUDA pinned host memory
    non_blocking = backend in {"cuda", "mps"}  # enables async transfers when possible
    return DeviceContext(device=device, backend=backend, pin_memory=pin_memory, non_blocking=non_blocking)


def seed_torch(seed: int) -> None:
    """Set torch RNG across CPU/CUDA/MPS if present."""
    import torch

    torch.manual_seed(seed)  # CPU side
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # all GPUs
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)  # Apple MPS


def empty_device_cache() -> None:
    """Release cached GPU memory when using CUDA."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
