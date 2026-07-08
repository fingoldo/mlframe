"""System-level helpers (GPU import guards, kernel tuning cache).

Public surface re-exports the shared cupy import guard so cross-package callers use
``from mlframe.system import try_import_cupy`` instead of reaching into the private
``_gpu_guard`` implementation module.
"""
from __future__ import annotations

from ._gpu_guard import try_import_cupy
