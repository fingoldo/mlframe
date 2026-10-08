"""CUDA kernel sources must use explicit 64-bit index types (NVRTC on Windows compiles ``long`` as 32 bits)."""
from __future__ import annotations

from pathlib import Path

from py_ci_shared.cuda_kernel_integer_width import assert_cuda_kernel_integer_width

SRC = Path(__file__).resolve().parents[2] / "src" / "mlframe"


def test_cuda_kernel_sources_use_explicit_64bit_types():
    """No platform-width ``long`` and no int index product in any CUDA kernel source."""
    assert_cuda_kernel_integer_width(SRC, min_files=500, include_advisory=True)
