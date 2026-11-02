"""One place that decides whether a GPU test may run, so no test needs a broad ``except`` around the code under test.

A test that wraps its GPU call in ``try: ... except Exception: pytest.skip(...)`` also turns a real kernel fault, a wrong launch or an assertion failure into
a skip. The check below touches the device once, before the test body, and skips only when there is no usable device.
"""

from __future__ import annotations

import pytest


def require_cuda_device() -> None:
    """Skip the calling test unless cupy imports and a CUDA device answers a one-element allocation."""
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA device")
        cp.zeros(1).sum()
    except Exception as exc:
        pytest.skip(f"needs a usable CUDA device: {exc}")


def require_cublas_matmul() -> None:
    """Skip the calling test unless the cuBLAS matmul probe passes: on a build where it fails the GPU path demotes to the CPU, so comparing the two proves nothing."""
    from mlframe.utils.cublas_probe import cublas_matmul_ok

    if not cublas_matmul_ok():
        pytest.skip("cupy/cuBLAS matmul probe failed on this build; the GPU path demotes to the CPU")


def require_free_vram(nbytes: int) -> None:
    """Skip the calling test unless the device has at least ``nbytes`` of free memory (the call itself skips when there is no usable device)."""
    require_cuda_device()
    import cupy as cp

    free_b, _total = cp.cuda.runtime.memGetInfo()
    if free_b < nbytes:
        pytest.skip(f"needs {nbytes >> 20} MiB of free VRAM, {free_b >> 20} MiB available")
