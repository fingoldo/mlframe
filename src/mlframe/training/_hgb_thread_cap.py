"""Load-aware thread cap for sklearn HistGradientBoosting fits.

When the host is already busy, HGB's OpenMP pool (one thread per logical core by default) oversubscribes the CPU and
the fit can run orders of magnitude slower than with a few threads. The decision is taken once per ``fit`` call from a
short CPU-load probe, never inside the boosting loop.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Iterator, Optional

from mlframe.config import HGBOOST_MODEL_TYPES

logger = logging.getLogger(__name__)

HGB_THREAD_CAP_ENABLED: bool = True
# psutil.cpu_percent is averaged over all logical CPUs: one fully busy core on a 22-thread host reads about 4.5%.
# 50% therefore means roughly half the machine is already busy, which is where OpenMP barrier waits start to dominate.
HGB_THREAD_CAP_LOAD_THRESHOLD_PCT: float = 50.0
HGB_THREAD_CAP_LIMIT: int = 4
HGB_THREAD_CAP_PROBE_SECONDS: float = 0.2


def _probe_cpu_percent(interval: float) -> float:
    """Average CPU utilisation over all logical cores, sampled for ``interval`` seconds."""
    import psutil

    return float(psutil.cpu_percent(interval=interval))


def is_hgb_model(model: object) -> bool:
    """True for sklearn HistGradientBoosting classifiers and regressors, matched by class name."""
    return type(model).__name__ in HGBOOST_MODEL_TYPES


def should_cap_threads(cpu_percent: float, threshold_pct: Optional[float] = None) -> bool:
    """True when the measured load is strictly above the threshold (the module default when not given)."""
    if threshold_pct is None:
        threshold_pct = HGB_THREAD_CAP_LOAD_THRESHOLD_PCT
    return cpu_percent > threshold_pct


@contextlib.contextmanager
def hgb_fit_thread_cap(model: object, *, enabled: Optional[bool] = None) -> Iterator[bool]:
    """Limit OpenMP/BLAS threads for the duration of a HistGradientBoosting ``fit`` when the CPU is loaded.

    ``enabled=None`` reads ``HGB_THREAD_CAP_ENABLED`` at call time, so the module switch can be flipped at runtime.
    Yields True when the cap is active. Non-HGB models, disabled runs and failed load probes keep the library default.
    """
    if enabled is None:
        enabled = HGB_THREAD_CAP_ENABLED
    if not enabled or not is_hgb_model(model):
        yield False
        return
    try:
        load_pct = _probe_cpu_percent(HGB_THREAD_CAP_PROBE_SECONDS)
    except Exception as exc:  # psutil missing or probe failure: keep the default thread count
        logger.debug("HGB thread cap: CPU load probe failed (%s); keeping default threads", exc)
        yield False
        return
    if not should_cap_threads(load_pct):
        yield False
        return
    from threadpoolctl import threadpool_limits

    logger.info("HGB fit: CPU load %.0f%% > %.0f%%, limiting threads to %d", load_pct, HGB_THREAD_CAP_LOAD_THRESHOLD_PCT, HGB_THREAD_CAP_LIMIT)
    with threadpool_limits(limits=HGB_THREAD_CAP_LIMIT):
        yield True
