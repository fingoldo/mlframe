"""Per-family FE wall-time accounting for MRMR.

The MRMR FE step fans out into several independent families -- orthogonal pair / triplet / quadruplet / adaptive-arity
cross-basis, and the smart-polynom pair search. cProfile cannot cleanly attribute their cost (compiled-njit / cupy time is
mis-charged to the Python caller, and the orchestrator note in ``_mrmr_fe_step/_step_core.py`` shows the body itself is at
floor), so which family dominates a given fit was previously unknowable. This module records true perf_counter wall per
family across a whole fit and logs a summary, so a default-disable / gating decision is measured, not guessed.

Usage: wrap each family entry call in ``with fe_family_timer("triplet"):`` and call ``log_fe_family_summary()`` once the fit
finishes. Zero overhead beyond one perf_counter pair per family invocation; safe to leave on permanently.
"""
from __future__ import annotations

import functools
import logging
import threading
from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

# family -> [total_wall_seconds, n_invocations]; process-global so nested-fit / composite-discovery passes accumulate and a
# single end-of-run summary reflects the whole suite. Guarded by a lock because FE families can run under joblib threads.
_FE_FAMILY_WALL: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
_LOCK = threading.Lock()


@contextmanager
def fe_family_timer(name: str):
    _t0 = perf_counter()
    try:
        yield
    finally:
        _dt = perf_counter() - _t0
        with _LOCK:
            slot = _FE_FAMILY_WALL[name]
            slot[0] += _dt
            slot[1] += 1


def fe_timed(name: str):
    """Decorator form of ``fe_family_timer`` for wrapping a whole family entry function with a one-line edit."""
    def _deco(func):
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            with fe_family_timer(name):
                return func(*args, **kwargs)
        return _wrapped
    return _deco


def reset_fe_family_wall() -> None:
    with _LOCK:
        _FE_FAMILY_WALL.clear()


def get_fe_family_wall() -> dict[str, tuple[float, int]]:
    with _LOCK:
        return {k: (v[0], int(v[1])) for k, v in _FE_FAMILY_WALL.items()}


def log_fe_family_summary(*, reset: bool = True) -> None:
    """Emit one INFO line ranking FE families by total wall, then optionally reset. No-op when nothing was recorded."""
    with _LOCK:
        if not _FE_FAMILY_WALL:
            return
        rows = sorted(_FE_FAMILY_WALL.items(), key=lambda kv: kv[1][0], reverse=True)
        total = sum(v[0] for _, v in rows) or 1e-9
        parts = [f"{name}={wall:.1f}s({100.0 * wall / total:.0f}%, x{n})" for name, (wall, n) in rows]
        if reset:
            _FE_FAMILY_WALL.clear()
    logger.info("MRMR FE per-family wall: %s  [total %.1fs]", "  ".join(parts), total)
