"""Per-family FE wall-time accounting for MRMR.

The MRMR FE step fans out into several independent families - orthogonal pair / triplet / quadruplet / adaptive-arity
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
import time
from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Callable, Iterator, Optional, TypeVar

from . import _stage_ledger as _ledger

_F = TypeVar("_F", bound=Callable[..., Any])

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

# family -> [total_wall_seconds, n_invocations]; process-global so nested-fit / composite-discovery passes accumulate and a
# single end-of-run summary reflects the whole suite. Guarded by a lock because FE families can run under joblib threads.
_FE_FAMILY_WALL: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
_LOCK = threading.Lock()
# In-flight timer count and the maximum ever observed. When families run concurrently (joblib threads, or a
# nested fit), their intervals OVERLAP, so the per-family walls no longer sum to real elapsed time and the
# percentages are shares-of-CPU-time, not shares-of-wall. Track the peak concurrency so the summary can say so
# instead of silently reporting a total that exceeds the fit's actual duration.
_INFLIGHT = 0
_MAX_INFLIGHT = 0


# Opt-in resident-memory accounting (``MLFRAME_FE_FAMILY_RSS=1``): per family the highest process RSS seen while one of its calls ran above the RSS at that call's start ("peak over entry") and the
# RSS still held when the call returned ("retained"); the figures are the maxima over the calls. ``tracemalloc`` does not see numba / cupy-host / driver memory (42 percent of the growth of a 100000 x 1000
# fit), the process RSS does. Concurrent families overlap, so a peak over entry may include a sibling's allocations (the summary flags it like the walls).
_FE_FAMILY_RSS: dict[str, list[float]] = {}
_RSS_ACTIVE: dict[int, list[float]] = {}  # entry id -> [rss at entry, highest rss seen]
_RSS_POLLER: "threading.Thread | None" = None
_RSS_NEXT_ID = 0


def _rss_tracking_enabled() -> bool:
    """Whether the per-family RSS accounting is on (environment ``MLFRAME_FE_FAMILY_RSS=1``)."""
    import os

    return os.environ.get("MLFRAME_FE_FAMILY_RSS", "0").strip() == "1"


def _process_rss() -> float:
    """Resident set size of this process in bytes (0.0 when psutil cannot read it)."""
    try:
        import psutil

        return float(psutil.Process().memory_info().rss)
    except Exception:
        return 0.0


def _rss_poll_loop() -> None:
    """Raise the highest-seen RSS of every active entry every 50 ms while at least one entry is active."""
    import time

    while True:
        time.sleep(0.05)
        rss = _process_rss()
        with _LOCK:
            for rec in _RSS_ACTIVE.values():
                if rss > rec[1]:
                    rec[1] = rss


def _rss_enter() -> int:
    """Register an entry with its starting RSS; start the poller on first use. Returns the entry id."""
    global _RSS_POLLER, _RSS_NEXT_ID
    rss = _process_rss()
    with _LOCK:
        _RSS_NEXT_ID += 1
        eid = _RSS_NEXT_ID
        _RSS_ACTIVE[eid] = [rss, rss]
        if _RSS_POLLER is None:
            _RSS_POLLER = threading.Thread(target=_rss_poll_loop, name="fe-family-rss", daemon=True)
            _RSS_POLLER.start()
    return eid


def _rss_exit(eid: int, name: str) -> None:
    """Close an entry and fold its peak over entry and retained growth into the per-family maxima."""
    rss = _process_rss()
    with _LOCK:
        start, high = _RSS_ACTIVE.pop(eid, (rss, rss))
        high = max(high, rss)
        slot = _FE_FAMILY_RSS.setdefault(name, [0.0, 0.0])
        slot[0] = max(slot[0], high - start)
        slot[1] = max(slot[1], rss - start)


@contextmanager
def fe_family_timer(name: str, device: Optional[str] = None, in_gb: Optional[float] = None) -> Iterator[None]:
    """Context manager that accumulates wall time and invocation count for ``name`` into the process-global ``_FE_FAMILY_WALL`` map under the lock.

    With the stage ledger on (``MLFRAME_STAGE_LEDGER``) the interval is also an event with its epoch start and end; ``device`` and ``in_gb`` label it where the call site knows them."""
    global _INFLIGHT, _MAX_INFLIGHT
    with _LOCK:
        _INFLIGHT += 1
        if _INFLIGHT > _MAX_INFLIGHT:
            _MAX_INFLIGHT = _INFLIGHT
    _eid = _rss_enter() if _rss_tracking_enabled() else 0
    _ledger_on = _ledger.enabled()
    _epoch0 = time.time() if _ledger_on else 0.0
    _t0 = perf_counter()
    try:
        yield
    finally:
        _dt = perf_counter() - _t0
        if _ledger_on:
            _ledger.record(name, _epoch0, time.time(), device=device, in_gb=in_gb)
        if _eid:
            _rss_exit(_eid, name)
        with _LOCK:
            _INFLIGHT -= 1
            slot = _FE_FAMILY_WALL[name]
            slot[0] += _dt
            slot[1] += 1


def record_fe_family_wall(name: str, dt: float) -> None:
    """Directly record an already-measured duration into the ``name`` bucket.

    For call sites where wrapping the timed region in ``with fe_family_timer(name):`` would force
    reindenting a large pre-existing block; callers measure ``perf_counter()`` before/after
    themselves and pass the delta here. Equivalent to one ``fe_family_timer`` context exit
    (it does not participate in the concurrency tracking, since the caller owns the timed region)."""
    if _ledger.enabled():  # the region just ended: its start is ``dt`` ago
        _now = time.time()
        _ledger.record(name, _now - float(dt), _now)
    with _LOCK:
        slot = _FE_FAMILY_WALL[name]
        slot[0] += float(dt)
        slot[1] += 1


def fe_timed(name: str) -> Callable[[_F], _F]:
    """Decorator form of ``fe_family_timer`` for wrapping a whole family entry function with a one-line edit."""
    def _deco(func: _F) -> _F:
        """Wrap ``func`` so every call is timed under ``fe_family_timer(name)``."""
        @functools.wraps(func)
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            """Time this call into the ``name`` family bucket, then delegate to the wrapped function."""
            with fe_family_timer(name):
                return func(*args, **kwargs)
        return _wrapped  # type: ignore[return-value]
    return _deco


#: The lap timers the current thread has opened a lap on (a stage worker closes them when its stage ends).
_THREAD_TIMERS = threading.local()


class FamilyLapTimer:
    """Attribute stage wall to families from their on/off gate alone, so no family needs its own timer.

    Every cascade family opens with ``_fe_family_on(flag)``. A lap starts when a gate returns True and ends at the next gate call or at ``close()`` (stage boundary / end of fit),
    and its wall is recorded under ``gate:<flag>``. Gates that return False accrue nothing. The stage body between two gates is charged to the family whose gate opened last,
    which is exact for a sequential cascade; ``gate:`` names stay apart from ``fe_timed`` names so the two kinds of buckets are never double counted in one ranking.

    The open lap is kept per thread: cascades that run concurrently each walk their own gates on their own thread, so one stage never closes or restarts another stage's lap. A worker thread
    ends its laps with :func:`close_thread_laps` when its stage finishes; laps of different threads overlap in time, so their sum can exceed the wall of the section (see the concurrency caveat of
    :func:`log_fe_family_summary`)."""

    def __init__(self) -> None:
        self._tls = threading.local()

    def _lap(self) -> Any:
        """The current thread's lap state (``flag``, ``t0``), created on first use and registered for :func:`close_thread_laps`."""
        tls = self._tls
        if not hasattr(tls, "flag"):
            tls.flag = None
            tls.t0 = 0.0
            timers = getattr(_THREAD_TIMERS, "timers", None)
            if timers is None:
                timers = _THREAD_TIMERS.timers = []
            timers.append(self)
        return tls

    def close(self) -> None:
        """End this thread's open lap, if any, and record its wall under ``gate:<flag>``."""
        lap = self._lap()
        if lap.flag is not None:
            record_fe_family_wall("gate:" + lap.flag, perf_counter() - lap.t0)
            if getattr(lap, "epoch0", 0.0):
                _ledger.record("gate:" + lap.flag, lap.epoch0, time.time())
                lap.epoch0 = 0.0
            if getattr(lap, "eid", 0):
                _rss_exit(lap.eid, "gate:" + lap.flag)
                lap.eid = 0
            lap.flag = None

    def gate(self, flag: str, is_on: bool) -> bool:
        """Close this thread's previous lap, open one for ``flag`` when ``is_on``, and return ``is_on`` unchanged."""
        self.close()
        if is_on:
            lap = self._lap()
            lap.flag = flag
            lap.eid = _rss_enter() if _rss_tracking_enabled() else 0
            lap.epoch0 = time.time() if _ledger.enabled() else 0.0
            lap.t0 = perf_counter()
        return is_on


def close_thread_laps() -> None:
    """End the open lap of every :class:`FamilyLapTimer` the calling thread has used (called when a stage worker finishes its stage)."""
    for timer in list(getattr(_THREAD_TIMERS, "timers", ())):
        timer.close()


def fe_family_wall_since(before: "dict[str, tuple[float, int]]") -> "dict[str, dict[str, float]]":
    """Per-family wall and call count accumulated since the ``before`` snapshot (``get_fe_family_wall()``), longest first.

    ``wall_s`` is the sum of the family's timed intervals, so the families of concurrent cascades overlap in time and their walls add up to more than the fit's elapsed time. ``calls`` is the
    number of timed intervals (laps for the ``gate:`` families)."""
    now = get_fe_family_wall()
    rows = {}
    for name, (wall, calls) in now.items():
        w0, c0 = before.get(name, (0.0, 0))
        if calls - c0 > 0 or wall - w0 > 0.0:
            rows[name] = {"wall_s": round(float(wall - w0), 4), "calls": int(calls - c0)}
    return dict(sorted(rows.items(), key=lambda kv: -kv[1]["wall_s"]))


def reset_fe_family_wall() -> None:
    """Clear all accumulated per-family wall-time/invocation counters (e.g. between independent fits in the same process)."""
    global _MAX_INFLIGHT
    with _LOCK:
        _FE_FAMILY_WALL.clear()
        _FE_FAMILY_RSS.clear()
        _MAX_INFLIGHT = _INFLIGHT


def get_fe_family_wall() -> dict[str, tuple[float, int]]:
    """Snapshot the current per-family ``(total_wall_seconds, n_invocations)`` map without resetting it."""
    with _LOCK:
        return {k: (v[0], int(v[1])) for k, v in _FE_FAMILY_WALL.items()}


def get_fe_family_max_concurrency() -> int:
    """Peak number of ``fe_family_timer`` regions that were ever in flight simultaneously.

    ``> 1`` means some family intervals OVERLAPPED, so the recorded walls sum to more than the fit's real
    elapsed time and the summary percentages are CPU-time shares, not wall shares.
    """
    with _LOCK:
        return int(_MAX_INFLIGHT)


def log_fe_family_summary(*, reset: bool = True) -> None:
    """Emit one INFO line ranking FE families by total wall, then optionally reset. No-op when nothing was recorded."""
    global _MAX_INFLIGHT
    with _LOCK:
        if not _FE_FAMILY_WALL:
            return
        rows = sorted(_FE_FAMILY_WALL.items(), key=lambda kv: kv[1][0], reverse=True)
        total = sum(v[0] for _, v in rows)
        if total == 0:
            total = 1e-9
        parts = [f"{name}={wall:.1f}s({100.0 * wall / total:.0f}%, x{n})" for name, (wall, n) in rows]
        overlapped = _MAX_INFLIGHT > 1
        peak = int(_MAX_INFLIGHT)
        if reset:
            _FE_FAMILY_WALL.clear()
            _MAX_INFLIGHT = _INFLIGHT
    # Without this caveat a reader would compare the total against the fit's wall clock and conclude the
    # accounting is broken; concurrent families genuinely double-count overlapping intervals.
    caveat = f"  [CONCURRENT: peak {peak} families in flight; walls overlap, shares are CPU-time not wall]" if overlapped else ""
    logger.info("MRMR FE per-family wall: %s  [total %.1fs]%s", "  ".join(parts), total, caveat)
    _ledger.dump_if_requested()
    with _LOCK:
        mem_rows = sorted(_FE_FAMILY_RSS.items(), key=lambda kv: kv[1][0], reverse=True)  # kept after the summary (the fit's own end-of-fit summary resets the walls): ``reset_fe_family_wall`` clears it
    if mem_rows:
        logger.info(
            "MRMR FE per-family RSS (MB, max over calls: peak above the entry / still held at exit): %s",
            "  ".join(f"{name}=+{peak / 2**20:.0f}/+{kept / 2**20:.0f}" for name, (peak, kept) in mem_rows),
        )
