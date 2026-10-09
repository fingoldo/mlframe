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

__all__ = ["autotune_mode", "cold_kernel_names", "kernel_tuning_fit_policy"]

logger = logging.getLogger(__name__)

_ENV_MODE = "MLFRAME_AUTOTUNE"
_ENV_SWEEP_SWITCH = "PYUTILZ_KERNEL_DISABLE_SWEEP"
_ON = frozenset({"1", "on", "true", "yes", "auto"})

_LOCK = threading.Lock()
_DEPTH = 0  # fits currently inside the policy (fits may run in several threads)
_WE_SET_SWITCH = False  # whether the sweep switch was set by this policy and must be removed when the last fit leaves
_WARNED = False  # the cold-cache notice has been given in this process


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
    """One log line per process when kernels are untuned here."""
    global _WARNED
    if _WARNED:
        return
    try:
        cold = cold_kernel_names()
    except Exception as exc:  # the notice is advice; it must never break a fit
        logger.debug("kernel tuning cold-cache check failed: %s", exc)
        return
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
