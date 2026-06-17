"""Global GPU off-switch for the feature-selection filters.

cupy's own device detection (``cupy.cuda.is_available()`` / ``_CUDA_AVAILABLE``) ignores both
``NUMBA_DISABLE_CUDA`` and ``CUDA_VISIBLE_DEVICES=""``, so GPU dispatch paths gated only on cupy
PRESENCE will still route to the GPU even when the caller asked for none. On a weak GPU (e.g. a
GTX 1050 Ti) that was catastrophic: the batched plug-in MI used by the orth-FE scoring spent ~37%
of a 300k fit in cupy ``argsort`` + GPU-sync ``time.sleep`` with the CPU idle.

``gpu_globally_disabled()`` is the single source of truth every GPU dispatch should consult (in
addition to the per-kernel size/availability gates), so a CPU-only or weak-GPU run can force the CPU
path. Honors:
  * ``MLFRAME_DISABLE_GPU=1`` -- explicit opt-out.
  * ``CUDA_VISIBLE_DEVICES=""`` (empty string) -- the documented mlframe "no GPU on this run"
    convention (numba honors it; cupy does not, hence this shim).
"""
from __future__ import annotations

import os


def gpu_globally_disabled() -> bool:
    """True when this run must avoid the GPU: ``MLFRAME_DISABLE_GPU=1`` or ``CUDA_VISIBLE_DEVICES=""``.

    The single source of truth every GPU dispatch consults (alongside per-kernel size/availability
    gates) so a CPU-only or weak-GPU run can force the CPU path; see the module docstring for why
    cupy's own detection is insufficient."""
    if os.environ.get("MLFRAME_DISABLE_GPU", "").strip() == "1":
        return True
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cvd is not None and cvd.strip() == "":
        return True
    return False


__all__ = ["gpu_globally_disabled"]
