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
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CACHE_SINGLETON: Optional[object] = None  # KernelTuningCache | False sentinel
_LOAD_LOCK = threading.Lock()
# Unexpected init failures are retried a bounded number of times rather than latched on the first one: the
# realistic causes (a concurrently-rewritten tuning file, a Windows file lock, a transient nvidia-smi fault) are
# all momentary, and latching costs the whole process its per-host measured thresholds.
_MAX_INIT_ATTEMPTS = 3
_INIT_ATTEMPTS = 0

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
        if not os.path.isfile(_DEFAULT_TUNING_JSON):
            _DEFAULTS_REGISTERED = True
            logger.debug("no default kernel-tuning JSON at %s; using hand fallbacks", _DEFAULT_TUNING_JSON)
            return
        try:
            from pyutilz.performance.kernel_tuning.cache import register_default_cache
        except ImportError:
            _DEFAULTS_REGISTERED = True  # genuinely absent, and it will not appear later in this process
            logger.debug("pyutilz.performance.kernel_tuning unavailable; skipping default-cache registration")
            return
        try:
            register_default_cache(_DEFAULT_TUNING_JSON)
        except Exception as _exc:  # never let a defaults problem break import
            # The flag is NOT set here, so a later caller re-attempts. It used to be set before the try, commented
            # "never re-attempt, even on failure" -- but the failures this catches are transient (the 17KB JSON
            # being rewritten by a concurrent sweep, a Windows PermissionError), and refusing to retry meant every
            # KTC lookup for the rest of the process missed the shipped measurement-derived defaults and fell
            # through to hand heuristics: the exact regression this file exists to prevent, at debug level.
            logger.warning(
                "register_default_cache(%s) failed (%s: %s); the shipped per-hardware kernel-tuning defaults are "
                "NOT registered and dispatch will use hand heuristics until a later call succeeds.",
                _DEFAULT_TUNING_JSON,
                type(_exc).__name__,
                _exc,
            )
        else:
            _DEFAULTS_REGISTERED = True


def get_kernel_tuning_cache() -> Optional[Any]:
    """Return the per-process KernelTuningCache singleton, or None if pyutilz is
    unavailable. Sentinel ``False`` caches the import-failure so subsequent
    calls don't re-attempt the lazy import.

    Typed ``Optional[Any]`` (not the concrete ``KernelTuningCache`` class) since
    pyutilz is an optional dependency and this module avoids importing it at
    module scope; callers rely on the ``.lookup(...)`` duck-typed contract.
    """
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
                # NOT latched off, and NOT quiet. `ImportError` above is the one genuine "unavailable" case, so
                # anything reaching here is unexpected: a corrupt or concurrently-rewritten tuning file, a
                # Windows file lock from another mlframe process, or a transient fault in the nvidia-smi
                # subprocess this constructor spawns. Latching on any of those pinned every kernel-tuning
                # lookup in the package -- 268 dispatch sites -- to hardcoded defaults for the rest of the
                # process, at `debug` level, so a run silently lost its per-host measured thresholds with
                # nothing to explain it. Same failure shape as the documented `_select_mi_backend` regression.
                global _INIT_ATTEMPTS
                _INIT_ATTEMPTS += 1
                if _INIT_ATTEMPTS >= _MAX_INIT_ATTEMPTS:
                    logger.warning(
                        "KernelTuningCache init failed %d times (%s: %s); giving up for this process and using "
                        "hand-tuned fallbacks. Per-host measured kernel thresholds are NOT in effect.",
                        _INIT_ATTEMPTS, type(_exc).__name__, _exc,
                    )
                    _CACHE_SINGLETON = False
                else:
                    logger.warning(
                        "KernelTuningCache init failed (%s: %s); will retry on the next lookup (attempt %d of %d).",
                        type(_exc).__name__, _exc, _INIT_ATTEMPTS, _MAX_INIT_ATTEMPTS,
                    )
                return None
        return _CACHE_SINGLETON


def _reset_for_tests() -> None:
    """Test-only: clear the singleton so tests with mocked pyutilz can reset state."""
    global _CACHE_SINGLETON, _DEFAULTS_REGISTERED, _INIT_ATTEMPTS
    with _LOAD_LOCK:
        _CACHE_SINGLETON = None
        _INIT_ATTEMPTS = 0  # otherwise a test that exercised the retry path leaves the next one pre-exhausted
    with _DEFAULTS_LOCK:
        _DEFAULTS_REGISTERED = False


# Register the anonymized default-tuning cache once, at import. This module is
# imported by ``mlframe.feature_selection.filters.__init__`` (the FS package init
# - the sensible, single import point), so the defaults are live before any
# dispatcher's first lookup. Guarded so a missing file / missing pyutilz is a
# no-op.
_register_default_tuning_cache()


__all__ = ["get_kernel_tuning_cache"]
