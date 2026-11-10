"""What a fit may do about kernel tuning: by default nothing silent.

A kernel's best backend/parameters are measured once per machine and cached. On a machine where the cache is cold (first run, new driver or library, edited kernel) the
dispatchers used to start a multi-minute background sweep INSIDE the first fit, competing with that fit for CPU and GPU and finishing (or not) after it. The fit now uses the
built-in measurement-backed defaults instead and says once that tuning is available; tuning is an explicit act:

    mlframe-tune-kernels ensure        # only what is missing or stale; instant when current

``MLFRAME_AUTOTUNE=on`` restores the old behaviour (sweeps start during a fit); ``off`` (the default) keeps fits free of them. An explicit ``PYUTILZ_KERNEL_DISABLE_SWEEP`` set by
the caller always wins.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from typing import Iterator

__all__ = ["autotune_mode", "cold_kernel_names", "kernel_tuning_fit_policy", "wait_for_cold_check"]

logger = logging.getLogger(__name__)

_ENV_MODE = "MLFRAME_AUTOTUNE"
_ENV_SWEEP_SWITCH = "PYUTILZ_KERNEL_DISABLE_SWEEP"
_ON = frozenset({"1", "on", "true", "yes", "auto"})

_LOCK = threading.Lock()
_DEPTH = 0  # fits currently inside the policy (fits may run in several threads)
_WE_SET_SWITCH = False  # whether the sweep switch was set by this policy and must be removed when the last fit leaves
_WARNED = False  # the cold-cache notice has been given in this process
_CHECKED_KEYS: frozenset = frozenset()  # the registry keys the check has already covered (every one of them fresh)
_CHECK_THREAD: "threading.Thread | None" = None  # the running cold-cache check, for :func:`wait_for_cold_check`


def autotune_mode() -> str:
    """``"on"`` when ``MLFRAME_AUTOTUNE`` allows sweeps during a fit, else ``"off"`` (the default)."""
    return "on" if os.environ.get(_ENV_MODE, "").strip().lower() in _ON else "off"


def cold_kernel_names() -> list:
    """Kernels (already imported in this process) with no valid tuning on this machine; GPU kernels are not counted on a host without CUDA.

    Only the registry as it stands is read - no module walk - so this is cheap enough for the start of a fit.
    """
    from pyutilz.performance.kernel_tuning.cache import KernelTuningCache
    from pyutilz.performance.kernel_tuning.registry import get_registry

    from ._ensure import FRESH, _cuda_present, spec_status

    cuda = _cuda_present()
    cache = KernelTuningCache()
    return sorted(name for name, spec in get_registry().items() if (cuda or not spec.gpu_capable) and spec_status(spec, cache) != FRESH)


def _warn_if_cold() -> None:
    """One log line per process when kernels are untuned here.

    The check reads the per-host cache from disk (1.3 s on the first call of a process, the cache's own load) and nothing in the fit depends on its answer, so it runs on a daemon thread and the
    fit's prelude goes on meanwhile; the thread also leaves the cache loaded for the first kernel dispatch that needs it."""
    global _CHECK_THREAD
    if _WARNED or frozenset(get_registry_keys()) <= _CHECKED_KEYS:
        return
    _CHECK_THREAD = threading.Thread(target=_cold_check, name="kernel-tuning-cold-check", daemon=True)
    _CHECK_THREAD.start()


def wait_for_cold_check(timeout: float = 30.0) -> None:
    """Wait for the cold-cache check started by the last fit entry (a no-op when none is running); for callers and tests that want its log line before they go on."""
    thread = _CHECK_THREAD
    if thread is not None:
        thread.join(timeout)


def get_registry_keys() -> "list[str]":
    """The names of the tuner specs registered so far (cheap: no module walk)."""
    from pyutilz.performance.kernel_tuning.registry import get_registry

    return list(get_registry())


def _cold_check() -> None:
    """Body of :func:`_warn_if_cold`: find the untuned kernels and log the notice once; a failure only means no advice is printed."""
    global _WARNED, _CHECKED_KEYS
    keys = frozenset(get_registry_keys())
    try:
        cold = cold_kernel_names()
    except Exception as exc:  # the notice is advice; it must never break a fit
        # best-effort: the cold-cache notice is advisory; a failed check only means no advice is printed.
        logger.debug("kernel tuning cold-cache check failed: %s", exc)
        return
    # every spec of this registry has been judged: a later fit re-checks only once new tuner modules have registered (the check costs a disk read of the whole per-host cache)
    _CHECKED_KEYS = keys
    if cold:
        _WARNED = True
        logger.warning(
            "kernel tuning cache is cold on this machine for %d kernel(s) (e.g. %s); this fit uses the built-in defaults. Run `mlframe-tune-kernels ensure` once "
            "(about 10-15 minutes; instant when nothing is missing) or set MLFRAME_AUTOTUNE=on to let fits tune in the background.",
            len(cold),
            ", ".join(cold[:3]),
        )


@contextlib.contextmanager
def kernel_tuning_fit_policy() -> Iterator[None]:
    """Run a fit without silent background sweeps (unless ``MLFRAME_AUTOTUNE=on``), restoring the caller's environment afterwards even if the fit raises."""
    global _DEPTH, _WE_SET_SWITCH
    if autotune_mode() == "on":
        yield
        return
    with _LOCK:
        if _DEPTH == 0:
            _warn_if_cold()
            if _ENV_SWEEP_SWITCH not in os.environ:
                os.environ[_ENV_SWEEP_SWITCH] = "1"
                _WE_SET_SWITCH = True
        _DEPTH += 1
    try:
        yield
    finally:
        with _LOCK:
            _DEPTH -= 1
            if _DEPTH == 0 and _WE_SET_SWITCH:
                os.environ.pop(_ENV_SWEEP_SWITCH, None)
                _WE_SET_SWITCH = False
