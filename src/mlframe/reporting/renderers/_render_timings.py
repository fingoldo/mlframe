"""Per-chart-type wall time for everything the suite renders.

At the default settings a run draws hundreds of figures across dozens of chart types, and until now the only
visible number was the total time of the phase that contained them. That is enough to know the report is slow
and not enough to know WHICH chart to cap, cache or drop -- a production run spent longer on post-fit diagnostics
than on the model fit itself, with no way to attribute it.

The key is derived from the output path -- the file stem with the backend tag and any numeric/hex disambiguator
stripped -- rather than passed by each of the ~100 call sites. That means a chart drawn once per target appears
as one row per target (``calibration_y_dummy`` and ``calibration_y_price`` stay separate). Collapsing those would
mean guessing which part of a stem is the target name, and a wrong guess merges two genuinely different charts;
the per-target rows sort next to each other anyway, so the cost of a family is still readable off the table.

Timings accumulate process-wide behind a lock (renders run on a thread pool), and are stamped into
``metadata["chart_timings"]`` plus logged as a table at the end of a suite.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_TIMINGS: Dict[str, Dict[str, float]] = {}

# A trailing ``.matplotlib``/``.plotly`` backend tag, and a numeric or hex disambiguator some call sites append.
_SUFFIX_RE = re.compile(r"\.(matplotlib|plotly)$|_[0-9a-f]{6,}$|_\d+$")


def chart_type_of(base_path: str) -> str:
    """Collapse one output path to the chart TYPE it is an instance of."""
    import os

    # Explicit emptiness checks rather than ``or``: a caller passing "" means "no name", which is exactly the
    # case being handled, so collapsing it through a truthiness default hides the intent.
    stem = os.path.basename(str(base_path)) if base_path else ""
    if not stem:
        return "unnamed"
    prev = None
    while prev != stem:
        prev = stem
        stem = _SUFFIX_RE.sub("", stem)
    return stem if stem else "unnamed"


def record_chart_render(chart_type: str, seconds: float, *, backend: str = "") -> None:
    """Add one render's wall time to its type's running total."""
    key = f"{chart_type} [{backend}]" if backend else chart_type
    with _LOCK:
        slot = _TIMINGS.setdefault(key, {"seconds": 0.0, "count": 0.0, "max_seconds": 0.0})
        slot["seconds"] += float(seconds)
        slot["count"] += 1.0
        slot["max_seconds"] = max(slot["max_seconds"], float(seconds))


def reset_chart_timings() -> None:
    """Drop everything recorded so far; called at the start of a suite so one run's table is one run's charts."""
    with _LOCK:
        _TIMINGS.clear()


def chart_timings_snapshot() -> List[Dict[str, Any]]:
    """Recorded types, most expensive first. Each row is ``{chart, seconds, count, avg_seconds, max_seconds}``."""
    with _LOCK:
        rows: List[Dict[str, Any]] = [
            {
                "chart": name,
                "seconds": round(v["seconds"], 4),
                "count": int(v["count"]),
                "avg_seconds": round(v["seconds"] / v["count"], 4) if v["count"] else 0.0,
                "max_seconds": round(v["max_seconds"], 4),
            }
            for name, v in _TIMINGS.items()
        ]
    rows.sort(key=lambda r: -float(r["seconds"]))
    return rows


def format_chart_timings(rows: List[Dict[str, Any]], *, top: int = 25) -> str:
    """The table as it appears in the log: total time is what you can act on, so it leads and it sorts."""
    if not rows:
        return "[charts] no figures rendered"
    total = sum(float(r["seconds"]) for r in rows)
    width = max(len(str(r["chart"])) for r in rows[:top])
    _figs = sum(int(r["count"]) for r in rows)
    lines = [f"[charts] {len(rows)} chart type(s), {_figs} figure(s), {total:.2f}s total"]
    lines.append(f"{'chart'.ljust(width)}  {'total':>8}  {'calls':>6}  {'avg':>8}  {'slowest':>8}  {'share':>6}")
    lines.append("-" * (width + 44))
    for r in rows[:top]:
        share = 100.0 * float(r["seconds"]) / total if total else 0.0
        lines.append(
            f"{str(r['chart']).ljust(width)}  {r['seconds']:7.2f}s  {r['count']:6d}  {r['avg_seconds']:7.3f}s  " f"{r['max_seconds']:7.3f}s  {share:5.1f}%"
        )
    if len(rows) > top:
        lines.append(f"... and {len(rows) - top} more chart type(s), {sum(float(r['seconds']) for r in rows[top:]):.2f}s")
    return "\n".join(lines)


__all__ = ["chart_timings_snapshot", "chart_type_of", "format_chart_timings", "record_chart_render", "reset_chart_timings"]
