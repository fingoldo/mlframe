"""Global GPU off-switch for the feature-selection filters.

cupy's own device detection (``cupy.cuda.is_available()`` / ``_CUDA_AVAILABLE``) ignores both
``NUMBA_DISABLE_CUDA`` and ``CUDA_VISIBLE_DEVICES=""``, so GPU dispatch paths gated only on cupy
PRESENCE will still route to the GPU even when the caller asked for none. On a weak GPU (e.g. a
GTX 1050 Ti) that was catastrophic: the batched plug-in MI used by the orth-FE scoring spent ~37%
of a 300k fit in cupy ``argsort`` + GPU-sync ``time.sleep`` with the CPU idle.

``gpu_globally_disabled()`` is the single source of truth every GPU dispatch should consult (in
addition to the per-kernel size/availability gates), so a CPU-only or weak-GPU run can force the CPU
path. Honors:
  * ``MLFRAME_DISABLE_GPU=1`` - explicit opt-out.
  * ``CUDA_VISIBLE_DEVICES=""`` (empty string) - the documented mlframe "no GPU on this run"
    convention (numba honors it; cupy does not, hence this shim).
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


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


def cuda_available_for_run() -> bool:
    """True when a CUDA device is present AND this run is allowed to use it.

    The predicate every CPU-vs-GPU dispatch should branch on. A bare ``is_cuda_available()`` answers only
    "does the machine have a GPU", which is not the question: it misses ``MLFRAME_DISABLE_GPU=1`` entirely
    (numba honours ``CUDA_VISIBLE_DEVICES=""``, nothing underneath knows about the mlframe flag), so a run
    that explicitly declared no GPU still routes to the device. Every such site is a silent divergence -
    no exception, just different backends and, where the paths are not bit-identical, a different selection.
    """
    if gpu_globally_disabled():
        return False
    try:
        from pyutilz.core.pythonlib import is_cuda_available

        return bool(is_cuda_available())
    except Exception as e:
        logger.debug("pyutilz is_cuda_available() failed, falling back to numba.cuda.is_available(): %s", e)
        try:
            from numba import cuda as _c

            return bool(getattr(_c, "is_available", lambda: False)())
        except Exception as e2:
            logger.debug("numba.cuda.is_available() probe failed, assuming CUDA unavailable: %s", e2)
            return False


__all__ = ["gpu_globally_disabled", "cuda_available_for_run"]
