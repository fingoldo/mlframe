"""Shared cupy import guard.

Collapses the ``try: import cupy except Exception: cp=None; _HAS_CUPY=False``
boilerplate that was copy-pasted across kernel modules into a single helper so
the fallback semantics (any import/runtime error -> no-GPU) live in one place.

Usage at a module top::

    from mlframe.system._gpu_guard import try_import_cupy

    cp, _HAS_CUPY = try_import_cupy()
"""
from __future__ import annotations

from typing import Any, Tuple


def try_import_cupy() -> Tuple[Any, bool]:
    """Return ``(cupy_module_or_None, has_cupy)``.

    Any exception during import (missing package, missing CUDA runtime, driver
    mismatch) is swallowed and reported as ``(None, False)`` -- identical to the
    hand-rolled guards it replaces, so callers keep their CPU fallback path.
    """
    try:
        import cupy as cp

        return cp, True
    except Exception:
        return None, False
