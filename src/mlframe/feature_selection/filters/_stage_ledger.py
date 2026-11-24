"""Per-stage ledger of an MRMR fit: when every stage started and ended, on which thread and device it ran and how many GB of data it read.

The question the ledger answers (owner, 2026-11-24): for each stage, was there a device with low load and enough free memory in the window BEFORE it started, and if so why was the stage not started earlier?
The ledger holds the stages; ``mlframe.feature_selection._benchmarks.profiling.hardware_timeline`` samples the machine (CPU per core, free RAM, GPU utilisation, free VRAM) on the same epoch clock, and
``stage_idle_report`` joins the two.

Off by default and free when off (one environment read per call). ``MLFRAME_STAGE_LEDGER=1`` keeps the events in memory (``snapshot()``); a path as the value also appends them as JSON lines at the end of
the fit (``dump_if_requested()``, called by ``log_fe_family_summary``). Every ``fe_family_timer`` / ``FamilyLapTimer`` interval and every FE stage of the concurrency scheduler is an event, so the ledger covers
the whole fit without a second set of call sites; ``device`` and ``in_gb`` are filled where the call site knows them (the FE stages: the slot's device and the step-input frame), ``None`` elsewhere (the report
then classifies the stage by the load measured while it ran)."""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

ENV = "MLFRAME_STAGE_LEDGER"
_FALSE = {"", "0", "false", "no", "off"}
_EVENTS: "list[dict[str, Any]]" = []
_LOCK = threading.Lock()


def _setting() -> str:
    """The raw value of ``MLFRAME_STAGE_LEDGER`` (empty when unset)."""
    return os.environ.get(ENV, "").strip()


def enabled() -> bool:
    """Whether stage events are being recorded."""
    return _setting().lower() not in _FALSE


def frame_gb(n_rows: int, n_cols: int, itemsize: int = 4) -> float:
    """Size in GB of a dense ``n_rows x n_cols`` frame of ``itemsize``-byte cells (the input size of a stage that reads the step-input frame)."""
    return float(n_rows) * float(n_cols) * float(itemsize) / 2**30


def record(name: str, t0: float, t1: float, *, device: Optional[str] = None, in_gb: Optional[float] = None) -> None:
    """Append one stage event; ``t0`` and ``t1`` are epoch seconds (``time.time()``)."""
    if not enabled():
        return
    event = {"name": str(name), "t0": float(t0), "t1": float(t1), "thread": threading.get_ident(), "thread_name": threading.current_thread().name, "device": device, "in_gb": in_gb}
    with _LOCK:
        _EVENTS.append(event)


@contextmanager
def stage(name: str, *, device: Optional[str] = None, in_gb: Optional[float] = None) -> Iterator[None]:
    """Record the wall of the enclosed block as one stage event (a no-op when the ledger is off)."""
    if not enabled():
        yield
        return
    t0 = time.time()
    try:
        yield
    finally:
        record(name, t0, time.time(), device=device, in_gb=in_gb)


def snapshot() -> "list[dict[str, Any]]":
    """A copy of the events recorded so far, in the order they ended."""
    with _LOCK:
        return [dict(e) for e in _EVENTS]


def reset() -> None:
    """Forget the recorded events."""
    with _LOCK:
        _EVENTS.clear()


def dump(path: str) -> int:
    """Append the recorded events to ``path`` as JSON lines and return their number."""
    events = snapshot()
    if events:
        with open(path, "a", encoding="utf-8") as fh:
            for e in events:
                fh.write(json.dumps(e) + "\n")
    return len(events)


def dump_if_requested() -> int:
    """At the end of a fit: when the environment value is a path, append the events to it and clear them; a plain ``1`` keeps them in memory. Returns the number written."""
    setting = _setting()
    if setting.lower() in _FALSE or setting == "1" or setting.lower() in {"true", "yes", "on"}:
        return 0
    written = dump(setting)
    reset()
    return written
