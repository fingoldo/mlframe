"""Structured phase timing for train_mlframe_models_suite.

The ``phase(name, **context)`` context manager logs a START/DONE pair with
duration and records the timing in a process-local registry. At the end of a
training suite, :func:`format_phase_summary` produces a ranked table of where
the wall-clock time actually went.

Usage
-----
>>> from mlframe.training.phases import phase, reset_phase_registry, format_phase_summary
>>> reset_phase_registry()
>>> with phase("predict_proba", model="cb", split="test", n_rows=900_000):
...     probs = model.predict_proba(df)
>>> print(format_phase_summary())

Design notes
------------
- Logging goes through a module-level logger at INFO. A nested phase keeps its
  own START/DONE — useful because nested overlap is additive in the registry.
- Exceptions are re-raised; the DONE line still prints so a crashing phase
  still gets a duration (marked ``DONE (raised ...)``).
- Thread-safe via a lock on the registry; cheap enough for the suite's call
  rate (~dozens to hundreds of phases per run).
"""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from time import perf_counter as _timer
from typing import Any, Dict, Iterator, List, Tuple

logger = logging.getLogger(__name__)


class _PhaseRegistry:
    __slots__ = ("_lock", "_totals", "_counts")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def record(self, name: str, seconds: float) -> None:
        with self._lock:
            self._totals[name] = self._totals.get(name, 0.0) + seconds
            self._counts[name] = self._counts.get(name, 0) + 1

    def snapshot(self) -> List[Tuple[str, float, int]]:
        with self._lock:
            return sorted(
                ((n, self._totals[n], self._counts[n]) for n in self._totals),
                key=lambda kv: -kv[1],
            )

    def reset(self) -> None:
        with self._lock:
            self._totals.clear()
            self._counts.clear()


_registry = _PhaseRegistry()


def reset_phase_registry() -> None:
    """Clear accumulated timings. Call once at the start of a training suite."""
    _registry.reset()


def record_phase(name: str, seconds: float) -> None:
    """Manually add a timing without using the context manager (e.g. to backfill
    an already-measured duration)."""
    _registry.record(name, seconds)


def phase_snapshot() -> List[Tuple[str, float, int]]:
    """Return list of ``(name, total_seconds, call_count)`` sorted by total desc."""
    return _registry.snapshot()


def format_phase_summary(top: int = 30) -> str:
    """Format the top-N phases by accumulated wall-clock time into a table."""
    rows = _registry.snapshot()[:top]
    if not rows:
        return "[phases] no timings recorded"
    name_w = max(len("phase"), max(len(n) for n, _, _ in rows))
    header = f"{'phase'.ljust(name_w)}   total       calls    avg"
    sep = "-" * len(header)
    lines = [header, sep]
    for name, total, count in rows:
        avg = total / count if count else 0.0
        lines.append(f"{name.ljust(name_w)}  {total:8.2f}s  {count:6d}  {avg:7.3f}s")
    return "\n".join(lines)


def _format_ctx(context: Dict[str, Any]) -> str:
    if not context:
        return ""
    parts = []
    for k, v in context.items():
        if v is None:
            continue
        parts.append(f"{k}={v}")
    return " ".join(parts)


@contextmanager
def phase(name: str, level: int = logging.DEBUG, **context: Any) -> Iterator[None]:
    """Time a block, log START/DONE, and accumulate into the global registry.

    Parameters
    ----------
    name:
        Phase label used as the registry key. Keep it stable across calls so
        repeated invocations aggregate.
    level:
        Logging level for the START/DONE lines. Default is now ``DEBUG`` so
        the START/DONE pair doesn't duplicate higher-signal INFO lines
        emitted from the callers themselves (e.g. core.py's
        "X done — shape, elapsed"). Set to ``INFO`` explicitly when the
        phase has no caller-side log counterpart. Exception paths
        (``RAISED ...``) are always logged at WARNING so failures stay
        visible even with the default verbosity.
    **context:
        Optional key=value metadata appended to the log lines (model name,
        split name, row count, etc.). Not included in the registry key.
    """
    ctx_str = _format_ctx(context)
    logger.log(level, f"[phase] {name} START {ctx_str}".rstrip())
    t0 = _timer()
    raised: BaseException | None = None
    try:
        yield
    except BaseException as e:
        raised = e
        raise
    finally:
        dt = _timer() - t0
        _registry.record(name, dt)
        if raised is not None:
            logger.warning(f"[phase] {name} RAISED {type(raised).__name__} after {dt:.2f}s {ctx_str}".rstrip())
        else:
            logger.log(level, f"[phase] {name} DONE in {dt:.2f}s {ctx_str}".rstrip())
