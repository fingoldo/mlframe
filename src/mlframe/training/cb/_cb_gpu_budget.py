"""Enforce a wall-clock limit on CatBoost GPU fits without losing the model.

CatBoost has no wall-clock limit and rejects Python callbacks on GPU, so the only way to stop a runaway GPU fit is to
interrupt it -- which leaves the estimator unfitted. The way around that: the fit runs with CatBoost snapshots enabled
(``save_snapshot``, into the guard's per-fit temp dir). When the monitor sees the configured time budget exceeded, or a
runaway fit (elapsed beyond ``MLFRAME_CB_GPU_RUNAWAY_FACTOR`` x the time the whole iteration budget would take at the
fit's own early rate), it interrupts the fit, and the fit is resumed from the snapshot with ``iterations`` capped at the
iterations already done. CatBoost then loads the snapshot, trains at most the few iterations after the last snapshot
save, and returns a normally fitted model -- the model trained so far instead of nothing.

Why ``max_ctr_complexity`` is pinned on the resume: CatBoost ignores ``iterations`` / ``learning_rate`` when it checks
that a snapshot matches the current params, but it auto-derives ``max_ctr_complexity`` from the planned iteration
count (measured: 4 for a 100000-iteration plan, 1 for a 150-iteration one at depth 8), and that derived value IS
compared -- so a plain capped resume fails with "Current training params differ from the params saved in snapshot".
The value the original fit used is read back from the snapshot's own serialised params and pinned.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_SNAPSHOT_INTERVAL_S = 120.0
DEFAULT_RUNAWAY_FACTOR = 3.0
_SNAPSHOT_NAME = "mlframe_fit.snap"


def _float_env(name: str, default: float) -> float:
    """Non-negative float from env var ``name``; ``default`` when unset or unparsable."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def snapshot_interval_from_env() -> float:
    """CatBoost snapshot interval in seconds from ``MLFRAME_CB_GPU_SNAPSHOT_S``; 0 or unset falls back to the default."""
    value = _float_env("MLFRAME_CB_GPU_SNAPSHOT_S", DEFAULT_SNAPSHOT_INTERVAL_S)
    return value if value > 0 else DEFAULT_SNAPSHOT_INTERVAL_S  # 0 would make CatBoost snapshot on every iteration


def runaway_factor_from_env() -> float:
    """0 disables enforcement of the runaway rule (the monitor still warns)."""
    return _float_env("MLFRAME_CB_GPU_RUNAWAY_FACTOR", DEFAULT_RUNAWAY_FACTOR)


def snapshot_params(snapshot_file: str) -> Optional[dict]:
    """The training params CatBoost serialised into a snapshot file (JSON embedded in the binary), or None."""
    try:
        with open(snapshot_file, "rb") as f:
            blob = f.read()
        start = blob.find(b'{"')
        if start < 0:
            return None
        obj, _ = json.JSONDecoder().raw_decode(blob[start:].decode("utf-8", errors="replace"))
        return obj if isinstance(obj, dict) else None
    except (OSError, ValueError):
        return None


def enable_snapshots(guard: Any) -> bool:
    """Turn on CatBoost snapshots for the guarded fit, into its temp dir. False when enforcement is impossible here."""
    est = guard.est
    if threading.current_thread() is not threading.main_thread():
        # The interrupt can only be delivered to the main thread; a fit on another thread can only be watched.
        logger.debug("cb-gpu budget: fit not on the main thread; limits are warned about, not enforced.")
        return False
    params = est.get_params()
    if params.get("save_snapshot") or params.get("snapshot_file"):
        logger.debug("cb-gpu budget: caller manages CatBoost snapshots; limits are warned about, not enforced.")
        return False
    guard.ensure_tmp_dir()
    guard.remember_original(("save_snapshot", "snapshot_file", "snapshot_interval"))
    guard.snapshot_file = os.path.join(guard.tmp_dir, _SNAPSHOT_NAME)
    est.set_params(save_snapshot=True, snapshot_file=guard.snapshot_file, snapshot_interval=int(snapshot_interval_from_env()))
    return True


def resume_capped(guard: Any, refit: Callable[[], Any]) -> Any:
    """After a limit interrupt: refit from the snapshot with ``iterations`` capped at the iterations already done."""
    from ._cb_gpu_monitor import read_time_left_tail

    est = guard.est
    row = read_time_left_tail(est.get_params().get("train_dir") or "")
    if row is None or not guard.snapshot_file or not os.path.exists(guard.snapshot_file):
        raise RuntimeError(f"CatBoost GPU fit interrupted ({guard.limit_reason}) but no progress/snapshot is available to resume from; the fit is lost.")
    done = int(row[0]) + 1
    saved = snapshot_params(guard.snapshot_file) or {}
    mcc = (saved.get("cat_feature_params") or {}).get("max_ctr_complexity")
    overrides: dict = {"iterations": done}
    if mcc is not None and est.get_params().get("max_ctr_complexity") is None:
        overrides["max_ctr_complexity"] = mcc
    guard.remember_original(tuple(overrides))
    est.set_params(**overrides)
    logger.warning(
        "[cb-gpu-budget] %s interrupted at iter=%d (%s); resuming from its snapshot with iterations capped at %d so the model "
        "trained so far is kept (params restored afterwards).",
        guard.model_type_name, done - 1, guard.limit_reason, done,
    )
    return refit()


def fit_with_cb_gpu_guard(
    unguarded_fit: Callable[..., Any],
    model: Any,
    model_obj: Any,
    model_type_name: str,
    train_df: Any,
    train_target: Any,
    fit_params: dict,
    verbose: bool = False,
) -> Any:
    """Run one fit under ``CatBoostGpuFitGuard``; resume from the snapshot if the guard's monitor stopped it at a limit."""
    from ._cb_gpu_monitor import CatBoostGpuFitGuard

    def _fit():
        """The unguarded fit with this call's arguments."""
        return unguarded_fit(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose)

    with CatBoostGpuFitGuard(model, model_obj, model_type_name, fit_params) as guard:
        result = None
        try:
            guard.set_fit_running(True)
            try:
                result = _fit()
            finally:
                guard.set_fit_running(False)
        except KeyboardInterrupt:
            if not guard.limit_reason:
                raise  # a genuine Ctrl+C: never swallowed
            if result is None:
                guard.stop_monitor()
                result = resume_capped(guard, _fit)
        return result
