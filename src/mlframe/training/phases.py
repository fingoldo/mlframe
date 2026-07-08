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
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class _PhaseRegistry:
    """Process-local, thread-safe accumulator of per-phase-name total duration, call count, and
    cumulative RAM delta. A single module-level instance backs all `phase(...)` context managers."""

    __slots__ = ("_counts", "_lock", "_ram_deltas_gb", "_totals")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        # Per-phase cumulative RAM delta (GB).
        # Phase-level RAM growth surfaces leaks that the per-call ``Done.
        # RAM usage: NGB`` line can't show -- e.g. "compute_split_metrics
        # +1.2GB across 32 calls" tells the operator which phase to
        # investigate when total RSS climbs unexpectedly.
        self._ram_deltas_gb: dict[str, float] = {}

    def record(self, name: str, seconds: float, ram_delta_gb: float = 0.0) -> None:
        """Accumulate one call's duration (and optional RAM delta) into the running totals for name."""
        with self._lock:
            self._totals[name] = self._totals.get(name, 0.0) + seconds
            self._counts[name] = self._counts.get(name, 0) + 1
            if ram_delta_gb:
                self._ram_deltas_gb[name] = self._ram_deltas_gb.get(name, 0.0) + ram_delta_gb

    def snapshot(self) -> list[tuple[str, float, int]]:
        """Return ``(name, total_seconds, call_count)`` tuples for every recorded phase, sorted
        by total duration descending so the biggest time sinks come first."""
        with self._lock:
            return sorted(
                ((n, self._totals[n], self._counts[n]) for n in self._totals),
                key=lambda kv: -kv[1],
            )

    def ram_delta_snapshot(self) -> dict[str, float]:
        """Per-phase cumulative RAM delta in GB. Sorted by absolute
        magnitude to surface biggest leaks/releases first."""
        with self._lock:
            return dict(sorted(
                self._ram_deltas_gb.items(),
                key=lambda kv: -abs(kv[1]),
            ))

    def reset(self) -> None:
        """Wipe all accumulated totals/counts/RAM deltas; called once at the start of a fresh suite run."""
        with self._lock:
            self._totals.clear()
            self._counts.clear()
            self._ram_deltas_gb.clear()


_registry = _PhaseRegistry()


def reset_phase_registry() -> None:
    """Clear accumulated timings. Call once at the start of a training suite."""
    _registry.reset()


def record_phase(name: str, seconds: float, ram_delta_gb: float = 0.0) -> None:
    """Manually add a timing without using the context manager (e.g. to backfill
    an already-measured duration). ``ram_delta_gb`` optional; not all callers
    have a RAM measurement available."""
    _registry.record(name, seconds, ram_delta_gb=ram_delta_gb)


def phase_snapshot() -> list[tuple[str, float, int]]:
    """Return list of ``(name, total_seconds, call_count)`` sorted by total desc."""
    return _registry.snapshot()


def phase_ram_snapshot() -> dict[str, float]:
    """Return ``{phase_name: cumulative_ram_delta_gb}`` sorted by abs magnitude."""
    return _registry.ram_delta_snapshot()


def _try_get_rss_gb() -> float:
    """Best-effort current process RSS in GB; 0.0 if psutil missing."""
    try:
        import psutil
        return float(psutil.Process().memory_info().rss / (1024 ** 3))
    except Exception:
        return 0.0


def format_phase_summary(top: int = 30) -> str:
    """Format the top-N phases by accumulated wall-clock time into a table.

    Adds a ``+/-RAM_GB`` column when the registry has captured per-phase
    RAM deltas (auto-populated by the ``phase()`` ctx
    manager). Phases where the delta is exactly 0.0 (no measurement, or
    truly zero net change) render as ``     `` (blank) so the column
    only highlights phases that moved RSS.
    """
    rows = _registry.snapshot()[:top]
    if not rows:
        return "[phases] no timings recorded"
    ram_deltas = _registry.ram_delta_snapshot()
    name_w = max(len("phase"), max(len(n) for n, _, _ in rows))
    header = f"{'phase'.ljust(name_w)}   total       calls    avg     +/-RAM"
    sep = "-" * len(header)
    lines = [header, sep]
    for name, total, count in rows:
        avg = total / count if count else 0.0
        delta = ram_deltas.get(name, 0.0)
        if abs(delta) >= 0.05:  # only show >=50MB to suppress noise
            ram_str = f" {delta:+6.2f}GB"
        else:
            ram_str = "        "
        lines.append(f"{name.ljust(name_w)}  {total:8.2f}s  {count:6d}  {avg:7.3f}s{ram_str}")
    return "\n".join(lines)


def _format_ctx(context: dict[str, Any], max_val_len: int = 120) -> str:
    """Format phase-context kwargs for log output with value truncation.

    Truncation rationale: callers may pass large objects as context
    kwargs, e.g. ``phase("fit", eval_set=huge_list)``
    or a debugger accidentally passing a 10k-row DataFrame. Without
    truncation the log line grows to MB+ per phase START/DONE pair,
    which blows past log rotation, breaks structured log aggregation
    (newline injection), and wastes disk on no useful signal. Only the
    *value* side is truncated; the key stays intact so the line is
    still greppable by field name. Truncation uses ``repr`` so quotes/
    commas don't get interpreted as separators downstream.
    """
    if not context:
        return ""
    parts = []
    for k, v in context.items():
        if v is None:
            continue
        s = str(v)
        if len(s) > max_val_len:
            s = s[: max_val_len - 3] + "..."
        parts.append(f"{k}={s}")
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
    # Bracket each phase with RSS sample so the registry accumulates a
    # RAM-delta column. psutil-optional; if
    # missing or transiently fails, we simply skip the delta.
    rss_pre_gb = _try_get_rss_gb()
    raised: BaseException | None = None
    try:
        yield
    except BaseException as e:
        raised = e
        raise
    finally:
        dt = _timer() - t0
        rss_post_gb = _try_get_rss_gb()
        ram_delta = (rss_post_gb - rss_pre_gb) if (rss_pre_gb > 0 and rss_post_gb > 0) else 0.0
        _registry.record(name, dt, ram_delta_gb=ram_delta)
        # Render +/-XXX MB on the DONE line only when the change is
        # >=50MB; otherwise the lines spam without information.
        ram_str = f" delta_RAM={ram_delta*1024:+.0f}MB" if abs(ram_delta) >= 0.05 else ""
        if raised is not None:
            logger.warning(f"[phase] {name} RAISED {type(raised).__name__} after {dt:.2f}s{ram_str} {ctx_str}".rstrip())
        else:
            logger.log(level, f"[phase] {name} DONE in {dt:.2f}s{ram_str} {ctx_str}".rstrip())
