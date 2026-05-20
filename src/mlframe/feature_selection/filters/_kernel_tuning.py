"""Shared module-singleton wrapper around ``pyutilz.system.kernel_tuning_cache.KernelTuningCache``.

Building a fresh KernelTuningCache instance per call re-runs ``_load`` ->
``_build_provenance`` -> ``gpu_capability_summary`` -> ``nvidia-smi`` subprocess
on EVERY call site, even though the cache is immutable for the process lifetime
of the loaded payload. Profile of fuzz combo c0143 attributed ~290ms across 6
``discretize_2d_array`` calls (48ms/call) entirely to this per-call subprocess
hit; ``filters/gpu.py`` has two more hot-path sites that pay the same per-call
cost.

This module provides a single lazily-built singleton that all FS hot-path
callers share, collapsing N subprocess spawns into one per process. Returns
None on pyutilz-unavailable systems; callers should fall through to their
hardcoded defaults.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_SINGLETON: Optional[object] = None  # KernelTuningCache | False sentinel
_LOAD_LOCK = threading.Lock()


def get_kernel_tuning_cache() -> Optional[object]:
    """Return the per-process KernelTuningCache singleton, or None if pyutilz is
    unavailable. Sentinel ``False`` caches the import-failure so subsequent
    calls don't re-attempt the lazy import."""
    global _CACHE_SINGLETON
    if _CACHE_SINGLETON is False:
        return None
    if _CACHE_SINGLETON is not None:
        return _CACHE_SINGLETON
    with _LOAD_LOCK:
        if _CACHE_SINGLETON is False:
            return None
        if _CACHE_SINGLETON is None:
            try:
                from pyutilz.system.kernel_tuning_cache import KernelTuningCache
                _CACHE_SINGLETON = KernelTuningCache()
            except ImportError:
                logger.debug(
                    "pyutilz.system.kernel_tuning_cache unavailable; "
                    "filters will use hand-tuned fallbacks"
                )
                _CACHE_SINGLETON = False
                return None
            except Exception as _exc:
                logger.debug(
                    "KernelTuningCache init failed (%s: %s); using fallbacks",
                    type(_exc).__name__, _exc,
                )
                _CACHE_SINGLETON = False
                return None
        return _CACHE_SINGLETON


def _reset_for_tests() -> None:
    """Test-only: clear the singleton so tests with mocked pyutilz can reset state."""
    global _CACHE_SINGLETON
    with _LOAD_LOCK:
        _CACHE_SINGLETON = None


__all__ = ["get_kernel_tuning_cache"]
