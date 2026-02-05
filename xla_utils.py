"""
XLA/CUDA environment helpers.

Some TensorFlow builds will try to use XLA on GPU during training/inference.
When XLA compiles for GPU, it needs access to CUDA's `libdevice` bitcode
(typically `nvvm/libdevice/libdevice*.bc`). In some conda/pip setups, XLA
fails to locate this file automatically and aborts with a fatal error.

This module provides a small, TF-free helper to set:
  XLA_FLAGS=--xla_gpu_cuda_data_dir=<cuda_root>

Call `ensure_xla_cuda_data_dir()` *before importing tensorflow* in scripts
that may trigger XLA GPU compilation.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import Optional


def _has_libdevice(cuda_root: str) -> bool:
    libdevice_dir = Path(cuda_root) / "nvvm" / "libdevice"
    if not libdevice_dir.is_dir():
        return False
    return bool(list(libdevice_dir.glob("libdevice*.bc")))


def ensure_xla_cuda_data_dir() -> Optional[str]:
    """
    Ensure `--xla_gpu_cuda_data_dir=...` is present in XLA_FLAGS.

    Returns:
      The chosen CUDA root (string) if one was set/found, else None.
    """
    existing = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_cuda_data_dir=" in existing:
        return None

    # Explicit override.
    override = os.environ.get("XLA_CUDA_DATA_DIR", "").strip()
    if override and _has_libdevice(override):
        os.environ["XLA_FLAGS"] = (existing + f" --xla_gpu_cuda_data_dir={override}").strip()
        return override

    candidates: list[str] = []

    # Common location for the pip wheel providing nvvm/libdevice in TF GPU installs.
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        candidates.append(f"{conda_prefix}/lib/python*/site-packages/nvidia/cuda_nvcc")

    # In case TF is run outside of conda but still has the nvidia pip packages.
    user_site = str(Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia" / "cuda_nvcc")
    candidates.append(user_site)
    candidates.append(str(Path.home() / ".local" / "lib" / "python*" / "site-packages" / "nvidia" / "cuda_nvcc"))

    # System CUDA toolkit fallbacks.
    cuda_home = os.environ.get("CUDA_HOME", "").strip()
    if cuda_home:
        candidates.append(cuda_home)
    candidates.extend([
        "/usr/local/cuda",
        "/usr",
    ])

    # Pick the first candidate (in order) that contains libdevice.
    for pat in candidates:
        for match in sorted(glob.glob(pat)):
            resolved = str(Path(match).resolve())
            if _has_libdevice(resolved):
                os.environ["XLA_FLAGS"] = (existing + f" --xla_gpu_cuda_data_dir={resolved}").strip()
                return resolved

    return None
