"""Shared device probe for the GPU test modules."""

from __future__ import annotations


def has_device(cp) -> bool:
    """Whether ``cp`` sees at least one usable CUDA device; the caller decides what to do when it does not."""
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def require_device():
    """The ``cupy`` module when it imports and sees a usable CUDA device, else skip the calling test before it computes anything."""
    import pytest

    cp = pytest.importorskip("cupy")
    if not has_device(cp):
        pytest.skip("no usable CUDA device")
    return cp
