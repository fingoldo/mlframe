"""Shared module-singleton wrapper around ``pyutilz.performance.kernel_tuning.cache.KernelTuningCache``.

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
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_SINGLETON: Optional[object] = None  # KernelTuningCache | False sentinel
_LOAD_LOCK = threading.Lock()

# Path to the repo-committed, anonymized DEFAULT tunings JSON (produced by
# ``mlframe.feature_selection._benchmarks.gen_default_tuning``). It ships inside
# the wheel next to THIS loader module (the _benchmarks producer is dev-only and
# not packaged). Resolved relative to this file so it works from a source
# checkout and an installed wheel alike.
_DEFAULT_TUNING_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_kernel_tuning.json")
_DEFAULTS_REGISTERED = False
_DEFAULTS_LOCK = threading.Lock()


def _register_default_tuning_cache() -> None:
    """Register the repo-committed anonymized default-tuning JSON with pyutilz, so
    a fresh host gets measurement-derived dispatch on a local cache MISS (before
    the hand heuristic) while its own background sweep runs.

    Guarded + idempotent + best-effort: a missing file, missing pyutilz, or any
    load error is a silent no-op (the dispatcher just falls through to its
    hand-tuned fallback, exactly as before). Fires ONCE per process."""
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return
    with _DEFAULTS_LOCK:
        if _DEFAULTS_REGISTERED:
            return
        _DEFAULTS_REGISTERED = True  # never re-attempt, even on failure
        if not os.path.isfile(_DEFAULT_TUNING_JSON):
            logger.debug("no default kernel-tuning JSON at %s; using hand fallbacks", _DEFAULT_TUNING_JSON)
            return
        try:
            from pyutilz.performance.kernel_tuning.cache import register_default_cache
        except ImportError:
            logger.debug("pyutilz.performance.kernel_tuning unavailable; skipping default-cache registration")
            return
        try:
            register_default_cache(_DEFAULT_TUNING_JSON)
        except Exception as _exc:  # never let a defaults problem break import
            logger.debug("register_default_cache(%s) failed (%s: %s)", _DEFAULT_TUNING_JSON, type(_exc).__name__, _exc)


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
                from pyutilz.performance.kernel_tuning.cache import KernelTuningCache
                _CACHE_SINGLETON = KernelTuningCache()
            except ImportError:
                logger.debug("pyutilz.performance.kernel_tuning.cache unavailable; " "filters will use hand-tuned fallbacks")
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
    global _CACHE_SINGLETON, _DEFAULTS_REGISTERED
    with _LOAD_LOCK:
        _CACHE_SINGLETON = None
    with _DEFAULTS_LOCK:
        _DEFAULTS_REGISTERED = False


# Register the anonymized default-tuning cache once, at import. This module is
# imported by ``mlframe.feature_selection.filters.__init__`` (the FS package init
# -- the sensible, single import point), so the defaults are live before any
# dispatcher's first lookup. Guarded so a missing file / missing pyutilz is a
# no-op.
_register_default_tuning_cache()


__all__ = ["get_kernel_tuning_cache"]
