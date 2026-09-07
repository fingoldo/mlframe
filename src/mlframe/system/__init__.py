"""System-level helpers (GPU import guards, kernel tuning cache).

Public surface re-exports the shared cupy import guard and GPU-bound-callable predicate so
cross-package callers use ``from mlframe.system import try_import_cupy, callable_looks_gpu_bound``
instead of reaching into the private ``_gpu_guard`` implementation module.
"""
from __future__ import annotations

from ._gpu_guard import callable_looks_gpu_bound, try_import_cupy
# Same reasoning as the cupy guard above: two feature-selection modules size their batches from this
# probe, so it is part of this package's surface rather than something they reach in for.
from ._ram_probe import available_ram_bytes

__all__ = ["available_ram_bytes", "callable_looks_gpu_bound", "try_import_cupy"]
