"""Per-owner rollup of every model-dataset build in a run.

Individual ``[dataset-build]`` lines answer "what was built"; five of them scattered across three minutes of log
do not answer "who spent that time". A production run built five 1.96M-row LightGBM datasets on a fit whose model
list was ``['cb']`` -- visible only as five separate lines attributed to ``sklearn.model_selection._validation``,
which names the machinery and not the caller. Aggregating by owner turns that into one line naming the mlframe
module, its build count and its total cost, which is what a reader can act on.

Cheap by construction: one dict update per dataset build, and builds are already expensive enough that the
bookkeeping is invisible next to them.
"""

from __future__ import annotations

import logging
import sys as _sys
import threading
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_BUILDS: Dict[str, Dict[str, float]] = {}


def infer_build_callsite(skip_frames: int = 2) -> str:
    """Where this dataset build came from: the nearest non-library frame, and the mlframe frame that owns it.

    Reporting only the nearest non-library frame is not enough to act on. A production log attributed five
    1.96M-row LightGBM builds to ``sklearn.model_selection._validation:1319`` -- true, and useless: sklearn
    is a mechanism, not an owner, and nothing in mlframe was named. Anything that dispatches through a
    third-party driver (sklearn CV, joblib, functools wrappers) hides its caller the same way.

    So the walk skips the model libraries AND the common dispatch layers, and separately keeps looking for
    the first ``mlframe.*`` frame. The result reads ``owner <- mechanism`` when the two differ, so the
    reader gets both the module to change and the machinery that fired it.
    """
    _MODEL_LIBS = ("catboost.", "xgboost.", "lightgbm.")
    # Dispatch layers that stand between mlframe and the build: naming one of these answers "how", never "who".
    _DISPATCH = ("sklearn.", "joblib.", "threadpoolctl", "concurrent.", "functools", "asyncio.", "numpy.", "pandas.")
    try:
        frame: Any = _sys._getframe(skip_frames)
        nearest = None
        owner = None
        for _ in range(60):
            if frame is None:
                break
            mod = frame.f_globals.get("__name__", "?") or "?"
            if nearest is None and not mod.startswith(_MODEL_LIBS):
                nearest = f"{mod}:{frame.f_lineno}"
            if owner is None and mod.startswith("mlframe."):
                owner = f"{mod}:{frame.f_lineno}"
            # An mlframe frame that is itself only a thin build shim still counts as the owner: it is the
            # first place a reader can put a breakpoint. Keep walking only until BOTH are known.
            if owner is not None and nearest is not None and not nearest.startswith(_DISPATCH):
                break
            frame = frame.f_back
        if owner and nearest and owner != nearest:
            return f"{owner} <- {nearest}"
        return owner or nearest or "?"
    except Exception as exc:
        logger.debug("_infer_callsite: stack walk failed, call site unknown: %s", exc)
        return "?"


# Module-name fragments of the internal fit loops whose per-iteration builds are demoted to DEBUG: composite discovery,
# screening, baseline-diagnostics ablation, and every feature-selection wrapper (RFECV folds x elimination steps, the
# permutation-importance scorer calls). Those builds are the per-fold noise; the per-owner rollup still counts them.
_INTERNAL_LOOP_MODULE_MARKERS = ("composite", "screening", "baseline_diagnostics", "mlframe.feature_selection.")

# A build this slow stays at INFO even inside an internal loop: its cost is worth a line of its own.
SLOW_INTERNAL_BUILD_SECONDS = 5.0


def originates_in_internal_loop(skip_frames: int = 2) -> bool:
    """True if any ancestor frame lives in a known per-iteration internal fit loop.

    The reported call site is the nearest non-library frame, which is often mlframe's own thin dataset-build shim
    (lgb_shim / xgb_shim) and so masks the originating loop; the whole stack is scanned instead. 60 frames, not 25:
    an internal loop that dispatches through sklearn CV plus joblib puts more than 25 frames between itself and the
    constructor.
    """
    try:
        frame: Any = _sys._getframe(skip_frames)
        for _ in range(60):
            if frame is None:
                break
            mod = (frame.f_globals.get("__name__", "") or "").lower()
            if any(m in mod for m in _INTERNAL_LOOP_MODULE_MARKERS):
                return True
            frame = frame.f_back
    except Exception as exc:
        logger.debug("originates_in_internal_loop: stack walk failed: %s", exc)
    return False


def dataset_build_log_level(elapsed: float, skip_frames: int = 3) -> int:
    """INFO for a main-path build or a slow one; DEBUG for a routine build fired from inside an internal fit loop."""
    if elapsed >= SLOW_INTERNAL_BUILD_SECONDS:
        return logging.INFO
    return logging.DEBUG if originates_in_internal_loop(skip_frames=skip_frames) else logging.INFO


def record_dataset_build(label: str, owner: str, rows: int, seconds: float) -> None:
    """Add one dataset build to its owner's running totals."""
    key = f"{owner} [{label}]"
    with _LOCK:
        slot = _BUILDS.setdefault(key, {"count": 0.0, "seconds": 0.0, "rows": 0.0, "max_rows": 0.0})
        slot["count"] += 1.0
        slot["seconds"] += float(seconds)
        slot["rows"] += float(max(rows, 0))
        slot["max_rows"] = max(slot["max_rows"], float(max(rows, 0)))


def reset_dataset_build_stats() -> None:
    """Drop everything recorded so far, so one suite's table describes one suite's builds."""
    with _LOCK:
        _BUILDS.clear()


def dataset_build_snapshot() -> List[Dict[str, Any]]:
    """Recorded owners, most rows-built first. Rows built is the driver of the cost, so it decides the order."""
    with _LOCK:
        rows: List[Dict[str, Any]] = [
            {
                "owner": name,
                "count": int(v["count"]),
                "seconds": round(v["seconds"], 3),
                "rows_total": int(v["rows"]),
                "rows_max": int(v["max_rows"]),
            }
            for name, v in _BUILDS.items()
        ]
    rows.sort(key=lambda r: -float(r["rows_total"]))
    return rows


def format_dataset_build_stats(rows: List[Dict[str, Any]], *, top: int = 15) -> str:
    """The table as it appears in the log."""
    if not rows:
        return "[dataset-builds] none recorded"
    total_rows = sum(int(r["rows_total"]) for r in rows)
    total_builds = sum(int(r["count"]) for r in rows)
    width = max(len(str(r["owner"])) for r in rows[:top])
    lines = [f"[dataset-builds] {total_builds} build(s) across {len(rows)} owner(s), {total_rows:,} row(s) materialised"]
    lines.append(f"{'owner'.ljust(width)}  {'builds':>6}  {'rows total':>14}  {'largest':>12}  {'ctor s':>7}")
    lines.append("-" * (width + 48))
    lines.extend(f"{str(r['owner']).ljust(width)}  {r['count']:6d}  {r['rows_total']:14,}  {r['rows_max']:12,}  {r['seconds']:7.2f}" for r in rows[:top])
    if len(rows) > top:
        lines.append(f"... and {len(rows) - top} more owner(s)")
    return "\n".join(lines)


__all__ = [
    "SLOW_INTERNAL_BUILD_SECONDS",
    "dataset_build_log_level",
    "dataset_build_snapshot",
    "infer_build_callsite",
    "originates_in_internal_loop",
    "format_dataset_build_stats",
    "record_dataset_build",
    "reset_dataset_build_stats",
]
