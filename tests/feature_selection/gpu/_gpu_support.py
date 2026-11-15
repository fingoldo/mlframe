"""Shared device probe for the GPU test modules."""

from __future__ import annotations


def has_device(cp) -> bool:
    """Whether ``cp`` sees at least one usable CUDA device; the caller decides what to do when it does not."""
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False
