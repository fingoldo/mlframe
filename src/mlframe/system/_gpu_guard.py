"""Shared cupy import guard.

Collapses the ``try: import cupy except Exception: cp=None; _HAS_CUPY=False``
boilerplate that was copy-pasted across kernel modules into a single helper so
the fallback semantics (any import/runtime error -> no-GPU) live in one place.

Usage at a module top::

    from mlframe.system._gpu_guard import try_import_cupy

    cp, _HAS_CUPY = try_import_cupy()
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Optional, Tuple

_GPU_BOUND_TOKENS = ("torch", "cupy", "cuda", "numba.cuda", "tensorflow", "jax")


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


def callable_looks_gpu_bound(fn: Optional[Callable]) -> bool:
    """Heuristically flag a user-supplied callable (custom metric, custom kernel) as GPU-bound.

    User callables passed into a joblib fan-out (ensembling custom metrics, bootstrap ``metric_fns``) are opaque
    to the dispatcher: there is no model_config to inspect, only the function object itself. This inspects the
    callable's own code object AND any closed-over globals/nonlocals for references to a GPU library name (torch,
    cupy, numba.cuda, tensorflow, jax) -- a static heuristic, not proof, but cheap and safe: it never executes the
    callable and degrades to ``False`` (assume CPU-safe) on any introspection failure, so it never blocks legitimate
    CPU-only callables it fails to inspect (e.g. builtins, C-extension callables with no accessible ``__code__``).
    """
    if fn is None:
        return False
    try:
        code = getattr(fn, "__code__", None)
        if code is None:
            return False
        names = set(code.co_names)
        try:
            closure_vars = inspect.getclosurevars(fn)
            names |= set(closure_vars.globals.keys())
            names |= set(closure_vars.nonlocals.keys())
            for value in (*closure_vars.globals.values(), *closure_vars.nonlocals.values()):
                module = getattr(value, "__module__", None)
                if isinstance(module, str) and any(token in module for token in _GPU_BOUND_TOKENS):
                    return True
        except TypeError:
            pass
        return any(any(token in name for token in _GPU_BOUND_TOKENS) for name in names)
    except Exception:
        return False
