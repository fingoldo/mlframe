"""CatBoost-on-GPU fit guard: no Python callbacks, native early stopping only, and a side thread watching progress.

CatBoost rejects ANY ``callbacks=`` list when ``task_type="GPU"`` ("User defined callbacks are not supported for GPU"), so
mlframe's UniversalCallback (time budget, patience, progress, RAM monitoring) and the monotonic-decline stop cannot run
there. Wiring them anyway costs a guaranteed-to-fail first fit plus a Pool rebuild, and then the retry runs with no
supervision at all: a production fit ran 3.2 h at ~1/48 of the throughput the same model had the day before and
nothing flagged it.

What this module does instead, decided from the model's resolved params right before ``fit``:

* strips ``callbacks`` from the fit kwargs (the reactive CatBoostError fallback in the training loop stays as a backstop);
* logs ONCE per process which mlframe features that disables and whether native early stopping is configured;
* points CatBoost's ``train_dir`` at a unique temp dir (``allow_writing_files`` is off by default in mlframe) so the
  booster's own per-iteration ``time_left.tsv`` (``iter / Passed ms / Remaining ms``) can be read, restoring the
  caller's params after the fit so the pickled model does not carry a temp path;
* runs :class:`CatBoostGpuFitMonitor`, a daemon thread that every ``MLFRAME_CB_GPU_MONITOR_S`` seconds (default 60, 0
  disables) logs iteration, it/s, elapsed and CatBoost's ETA plus GPU utilisation / memory / other GPU processes, and
  WARNs on a throughput collapse or a stall, naming likely causes (GPU contention, VRAM pressure / spill).

Time budget / runaway fits: CatBoost has no native wall-clock limit, and an interrupted fit is left UNFITTED. The guard
therefore runs the fit with CatBoost snapshots on, and when the monitor sees the configured ``time_budget_mins`` exceeded
or a runaway fit, it interrupts and the fit is resumed from the snapshot with ``iterations`` capped: the model trained so
far is kept. See ``_cb_gpu_budget`` (and ``MLFRAME_CB_GPU_RUNAWAY_FACTOR`` / ``MLFRAME_CB_GPU_SNAPSHOT_S``).
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MONITOR_INTERVAL_S = 60.0
# Current throughput below this fraction of the fit's own early throughput counts as a collapse.
COLLAPSE_RATIO = 0.25
# Elapsed beyond this multiple of the early-rate projection for the full iteration budget counts as overrun.
OVERRUN_FACTOR = 3.0

_ONCE_LOCK = threading.Lock()
_NOTICE_LOGGED = False


def cb_model_is_gpu(model: Any) -> bool:
    """True when ``model`` (a CatBoost estimator, possibly wrapped in a Pipeline / TransformedTargetRegressor) resolves to GPU."""
    try:
        est = model
        if type(est).__name__ == "Pipeline":
            est = est.steps[-1][1]
        est = getattr(est, "regressor", None) or est
        _cb_mod = sys.modules.get("catboost")
        _cb_base = getattr(_cb_mod, "CatBoost", None) if _cb_mod is not None else None
        if not (type(est).__module__.startswith("catboost") or (_cb_base is not None and isinstance(est, _cb_base))):
            return False
        params = est.get_params()
        return str(params.get("task_type") or "").upper() == "GPU"
    except Exception:
        return False


def _native_es_summary(est: Any) -> str:
    try:
        p = est.get_params()
    except Exception:
        return "unknown"
    parts = [f"{k}={p[k]}" for k in ("early_stopping_rounds", "od_type", "od_wait", "od_pval") if p.get(k) is not None]
    return ", ".join(parts) if parts else "NONE (no early_stopping_rounds / od_wait set: the fit runs the full iteration budget)"


def _time_budget_from_callbacks(callbacks: Any) -> Optional[float]:
    for cb in callbacks or []:
        tb = getattr(cb, "time_budget_mins", None)
        if tb:
            try:
                return float(tb) * 60.0
            except (TypeError, ValueError):
                continue
    return None


def _log_notice_once(model_type_name: str, stripped: List[Any], es_summary: str, budget_s: Optional[float]) -> None:
    global _NOTICE_LOGGED
    with _ONCE_LOCK:
        if _NOTICE_LOGGED:
            return
        _NOTICE_LOGGED = True
    names = sorted({type(c).__name__ for c in stripped}) or ["none wired"]
    logger.warning(
        "CatBoost GPU fits run WITHOUT mlframe Python callbacks (CatBoost rejects callbacks on GPU). Disabled for every "
        "CatBoost GPU fit in this process: UniversalCallback time budget / patience / progress + RAM logging, monotonic-decline "
        "stop, per-iteration metric capture (dropped here: %s). Native early stopping in effect: %s.%s A side-thread monitor "
        "(MLFRAME_CB_GPU_MONITOR_S, default %ds) reports progress / throughput collapse instead. [%s]",
        ", ".join(names), es_summary,
        f" Configured time budget {budget_s / 60:.0f} min is only WARNED about (no safe native limit)." if budget_s else "",
        int(DEFAULT_MONITOR_INTERVAL_S), model_type_name,
    )


def _reset_notice_for_tests() -> None:
    global _NOTICE_LOGGED
    _NOTICE_LOGGED = False


def monitor_interval_from_env(default: float = DEFAULT_MONITOR_INTERVAL_S) -> float:
    raw = os.environ.get("MLFRAME_CB_GPU_MONITOR_S", "").strip()
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


# ----------------------------------------------------------------------------------------------------------------------
# Progress file reading
# ----------------------------------------------------------------------------------------------------------------------


def read_time_left_tail(train_dir: str) -> Optional[Tuple[int, float, float]]:
    """Last complete ``(iter, passed_s, remaining_s)`` row of ``<train_dir>/time_left.tsv``; reads only the file tail."""
    path = os.path.join(train_dir, "time_left.tsv")
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4096))
            chunk = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    lines = chunk.split("\n")
    # The final element is either "" (file ends with newline) or a partially written row: skip it.
    for line in reversed(lines[:-1]):
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            try:
                return int(parts[0]), float(parts[1]) / 1000.0, float(parts[2]) / 1000.0
            except ValueError:
                continue
    return None


def _fmt_s(s: Optional[float]) -> str:
    if s is None:
        return "?"
    s = max(0.0, float(s))
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.2f}h"


class CatBoostGpuFitMonitor:
    """Side thread that watches a CatBoost fit through its ``train_dir`` progress file. Never raises into the fit."""

    def __init__(
        self,
        train_dir: str,
        *,
        interval_s: float = DEFAULT_MONITOR_INTERVAL_S,
        label: str = "CatBoost",
        total_iterations: Optional[int] = None,
        time_budget_s: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
        gpu_probe: Optional[Callable[[], Any]] = None,
        log: Optional[logging.Logger] = None,
        on_limit: Optional[Callable[[str], bool]] = None,
        runaway_factor: float = 0.0,
    ):
        self.train_dir = train_dir
        self.interval_s = float(interval_s)
        self.label = label
        self.total_iterations = total_iterations
        self.time_budget_s = time_budget_s
        self._clock = clock
        self._gpu_probe = gpu_probe
        self._log = log or logger
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = clock()
        self._last: Optional[Tuple[float, int]] = None  # (clock, iter) at the previous poll
        self._early_rates: List[float] = []
        self._stall_polls = 0
        self._collapse_warned_at = -10**9
        self._stall_warned_at = -10**9
        self._budget_warned = False
        self._overrun_warned = False
        # Called on a time-budget / runaway breach until it returns True (it declines while no snapshot exists yet).
        self.on_limit = on_limit
        self.runaway_factor = float(runaway_factor or 0.0)
        self._limit_done = False
        self.polls = 0
        self.warnings: List[str] = []

    # -- lifecycle ------------------------------------------------------------------------------------------------------
    def start(self) -> "CatBoostGpuFitMonitor":
        self._t0 = self._clock()
        if self.interval_s > 0:
            self._thread = threading.Thread(target=self._run, name="mlframe-cb-gpu-monitor", daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and threading.current_thread() is not t:
            t.join(timeout)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.poll_once()
            except Exception as e:  # best-effort: a monitoring bug must never reach the fit
                try:
                    self._log.debug("cb-gpu-monitor poll failed: %s", e)
                except Exception:
                    pass

    # -- one observation --------------------------------------------------------------------------------------------------
    def _gpu(self) -> Tuple[Any, str]:
        probe = self._gpu_probe
        if probe is None:
            from .._gpu_state_probe import gpu_snapshot as probe  # lazy: keeps import cheap when no GPU fit happens
        try:
            snap = probe()
        except Exception:
            snap = None
        from .._gpu_state_probe import format_gpu_snapshot

        return snap, format_gpu_snapshot(snap, exclude_pid=os.getpid())

    def _warn(self, msg: str, *args: Any) -> None:
        text = msg % args if args else msg
        self.warnings.append(text)
        self._log.warning(text)

    def poll_once(self) -> Optional[Dict[str, Any]]:
        """Read progress, log one status line, run the collapse / stall / budget checks. Returns the computed stats."""
        self.polls += 1
        now = self._clock()
        elapsed = now - self._t0
        row = read_time_left_tail(self.train_dir)
        snap, gpu_s = self._gpu()
        if row is None:
            self._log.info("[cb-gpu-monitor] %s: no progress file yet in %s after %s | %s", self.label, self.train_dir, _fmt_s(elapsed), gpu_s)
            self._check_budget(elapsed, None, gpu_s)
            return None
        it, _passed_s, remaining_s = row
        cur_rate: Optional[float] = None
        if self._last is not None:
            dt = now - self._last[0]
            d_it = it - self._last[1]
            if dt > 0:
                cur_rate = d_it / dt
            if d_it <= 0:
                self._stall_polls += 1
            else:
                self._stall_polls = 0
                if len(self._early_rates) < 3:
                    self._early_rates.append(cur_rate or 0.0)
        elif it > 0 and elapsed > 0:
            # First observation: iterations since fit start. Includes CatBoost's startup (quantisation, H2D copy), so it
            # UNDER-states the early rate -- which only makes the collapse check more conservative.
            cur_rate = it / elapsed
            self._early_rates.append(cur_rate)
        self._last = (now, it)
        early = max(self._early_rates) if self._early_rates else None
        total = f"/{self.total_iterations}" if self.total_iterations else ""
        self._log.info(
            "[cb-gpu-monitor] %s: iter=%d%s it/s=%s (early %s) elapsed=%s cb-ETA=%s | %s",
            self.label, it, total,
            f"{cur_rate:.2f}" if cur_rate is not None else "?", f"{early:.2f}" if early else "?",
            _fmt_s(elapsed), _fmt_s(remaining_s), gpu_s,
        )
        stats = {"iter": it, "rate": cur_rate, "early_rate": early, "elapsed": elapsed, "remaining": remaining_s}
        self._check_collapse(stats, snap, gpu_s)
        self._check_budget(elapsed, it, gpu_s)
        return stats

    # -- checks ---------------------------------------------------------------------------------------------------------
    def _causes(self, snap: Any) -> str:
        causes = []
        try:
            from .._gpu_state_probe import other_gpu_processes

            others = other_gpu_processes(snap, exclude_pid=os.getpid())
            if others:
                causes.append("GPU contention - other processes on the GPU: " + ", ".join(f"{p.get('name')}[{p.get('pid')}]" for p in others[:10]))
            for g in (snap or {}).get("gpus", []):
                used, tot = g.get("mem_used_mb"), g.get("mem_total_mb")
                if used and tot and used / tot > 0.9:
                    causes.append(f"VRAM pressure on gpu{g.get('index')} ({used:.0f}/{tot:.0f}MB): CatBoost may be spilling / re-transferring")
                util = g.get("util_pct")
                if util is not None and util < 30:
                    causes.append(f"gpu{g.get('index')} utilisation only {util:.0f}%: the fit is starved (host-side bottleneck, paging, or CPU contention)")
        except Exception:
            pass
        if not causes:
            causes.append("GPU contention by another process, VRAM spill to host memory, host RAM paging, or thermal / power throttling")
        return "; ".join(causes)

    def _check_collapse(self, stats: Dict[str, Any], snap: Any, gpu_s: str) -> None:
        early, rate = stats["early_rate"], stats["rate"]
        if self._stall_polls >= 1:
            # Zero progress is a stall, not a collapse; one empty interval can be a long metric evaluation, two is a signal.
            if self._stall_polls >= 2 and self.polls - self._stall_warned_at >= 10:
                self._stall_warned_at = self.polls
                self._warn(
                    "[cb-gpu-monitor] %s STALLED: no new iteration for %d polls (%s) at iter=%d. Likely causes: %s. | %s",
                    self.label, self._stall_polls, _fmt_s(self._stall_polls * self.interval_s), stats["iter"], self._causes(snap), gpu_s,
                )
            return
        if early and rate is not None and len(self._early_rates) >= 1 and rate < COLLAPSE_RATIO * early:
            # Re-warn at most every 10 polls while the collapse persists so a multi-hour slow fit stays visible without flooding.
            if self.polls - self._collapse_warned_at >= 10:
                self._collapse_warned_at = self.polls
                self._warn(
                    "[cb-gpu-monitor] %s THROUGHPUT COLLAPSE: %.2f it/s now vs %.2f it/s early in this fit (%.0fx slower) at iter=%d, "
                    "elapsed %s. Likely causes: %s. | %s",
                    self.label, rate, early, early / max(rate, 1e-9), stats["iter"], _fmt_s(stats["elapsed"]), self._causes(snap), gpu_s,
                )
        if early and self.total_iterations and self.runaway_factor > 0:
            projected = self.total_iterations / early
            if stats["elapsed"] > self.runaway_factor * projected and stats["elapsed"] > 300:
                self._request_limit(
                    f"runaway: elapsed {_fmt_s(stats['elapsed'])} > {self.runaway_factor:g}x the {_fmt_s(projected)} the full "
                    f"{self.total_iterations}-iteration budget would take at this fit's early rate"
                )
        if early and self.total_iterations and not self._overrun_warned:
            projected = self.total_iterations / early
            if stats["elapsed"] > OVERRUN_FACTOR * projected and stats["elapsed"] > 300:
                self._overrun_warned = True
                self._warn(
                    "[cb-gpu-monitor] %s OVERRUN: elapsed %s is %.1fx the %s the full %d-iteration budget would take at this fit's early rate. "
                    "Likely causes: %s.",
                    self.label, _fmt_s(stats["elapsed"]), stats["elapsed"] / projected, _fmt_s(projected), self.total_iterations, self._causes(snap),
                )

    def _request_limit(self, reason: str) -> None:
        if self._limit_done or self.on_limit is None:
            return
        try:
            self._limit_done = bool(self.on_limit(reason))
        except Exception as e:  # never raise into the fit
            self._log.debug("cb-gpu-monitor on_limit failed: %s", e)

    def _check_budget(self, elapsed: float, it: Optional[int], gpu_s: str) -> None:
        if self.time_budget_s and elapsed > self.time_budget_s:
            if not self._budget_warned:
                self._budget_warned = True
                self._warn(
                    "[cb-gpu-monitor] %s EXCEEDED the configured time budget (%s > %s) at iter=%s. %s | %s",
                    self.label, _fmt_s(elapsed), _fmt_s(self.time_budget_s), it if it is not None else "?",
                    "Stopping it and keeping the model trained so far (resume from snapshot)." if self.on_limit is not None
                    else "It cannot be stopped without losing the model here (no snapshot / not on the main thread), so it keeps running.",
                    gpu_s,
                )
            self._request_limit(f"time budget {_fmt_s(self.time_budget_s)} exceeded")


# ----------------------------------------------------------------------------------------------------------------------
# Fit-time guard used by the training loop
# ----------------------------------------------------------------------------------------------------------------------


class CatBoostGpuFitGuard:
    """Context manager around one CatBoost fit; a no-op for anything that is not a GPU CatBoost fit.

    On enter (GPU only): strips ``fit_params["callbacks"]``, logs the once-per-process notice, redirects ``train_dir`` to a
    unique temp dir and starts the monitor. On exit (also on exception): stops the monitor, restores the model's
    ``train_dir`` / ``allow_writing_files`` and removes the temp dir.
    """

    def __init__(self, model: Any, model_obj: Any, model_type_name: str, fit_params: Dict[str, Any], *, interval_s: Optional[float] = None):
        self.model = model
        self.est = model_obj if model_obj is not None else model
        self.model_type_name = model_type_name
        self.fit_params = fit_params
        self.interval_s = monitor_interval_from_env() if interval_s is None else float(interval_s)
        self.active = False
        self.stripped: List[Any] = []
        self.monitor: Optional[CatBoostGpuFitMonitor] = None
        self._tmp_dir: Optional[str] = None
        self._restore: Optional[Dict[str, Any]] = None
        self.snapshot_file: Optional[str] = None
        self.limit_reason: Optional[str] = None
        self._fit_running = False
        self._lock = threading.Lock()

    def __enter__(self) -> "CatBoostGpuFitGuard":
        try:
            if not (cb_model_is_gpu(self.est) or (self.model is not self.est and cb_model_is_gpu(self.model))):
                return self
            self.active = True
            cbs = self.fit_params.pop("callbacks", None) if isinstance(self.fit_params, dict) else None
            self.stripped = list(cbs or [])
            budget_s = _time_budget_from_callbacks(self.stripped)
            _log_notice_once(self.model_type_name, self.stripped, _native_es_summary(self.est), budget_s)
            if self.interval_s > 0:
                self._redirect_train_dir()
                from ._cb_gpu_budget import enable_snapshots, runaway_factor_from_env

                runaway = runaway_factor_from_env()
                enforce = bool(budget_s or runaway > 0) and enable_snapshots(self)
                params = self.est.get_params()
                total = params.get("iterations") or params.get("n_estimators") or params.get("num_boost_round")
                self.monitor = CatBoostGpuFitMonitor(
                    params.get("train_dir") or "catboost_info",
                    interval_s=self.interval_s, label=self.model_type_name,
                    total_iterations=int(total) if total else None, time_budget_s=budget_s,
                    on_limit=self._request_interrupt if enforce else None, runaway_factor=runaway if enforce else 0.0,
                ).start()
        except Exception as e:  # best-effort: the guard must never prevent the fit
            logger.debug("CatBoostGpuFitGuard enter failed: %s", e)
        return self

    @property
    def tmp_dir(self) -> Optional[str]:
        return self._tmp_dir

    def ensure_tmp_dir(self) -> str:
        if self._tmp_dir is None:
            # A unique dir per fit: concurrent fits in one cwd would otherwise interleave rows in the shared catboost_info/.
            self._tmp_dir = tempfile.mkdtemp(prefix="mlframe_cb_gpu_")
        return self._tmp_dir

    def remember_original(self, keys: Any) -> None:
        """Record the current values of ``keys`` (first time only) so ``__exit__`` restores them on the estimator."""
        params = self.est.get_params()
        if self._restore is None:
            self._restore = {}
        for k in keys:
            self._restore.setdefault(k, params.get(k))

    def set_fit_running(self, running: bool) -> None:
        with self._lock:
            self._fit_running = bool(running)

    def stop_monitor(self) -> None:
        if self.monitor is not None:
            self.monitor.stop()

    def _request_interrupt(self, reason: str) -> bool:
        """Monitor callback on a limit breach: interrupt the (main-thread) fit, only while it runs and once a snapshot exists."""
        import _thread

        if not (self.snapshot_file and os.path.exists(self.snapshot_file)):
            return False  # nothing to resume from yet: interrupting now would lose the model; retried at the next poll
        with self._lock:
            if not self._fit_running or self.limit_reason:
                return self.limit_reason is not None
            self.limit_reason = reason
            _thread.interrupt_main()
        return True

    def _redirect_train_dir(self) -> None:
        params = self.est.get_params()
        if params.get("allow_writing_files") is not False and params.get("train_dir"):
            return  # the caller chose a train_dir explicitly: read it, leave it alone
        self.remember_original(("allow_writing_files", "train_dir"))
        self.est.set_params(allow_writing_files=True, train_dir=self.ensure_tmp_dir())

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            if self.monitor is not None:
                self.monitor.stop()
                if exc_type is None:
                    try:
                        self.monitor.poll_once()  # final line: iterations actually run, elapsed
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            if self._restore is not None:
                # Edits _init_params directly: set_params raises "You can't change params of fitted model" once the fit
                # succeeded, and a param that was never set must be REMOVED, not set to None (CatBoost serialises None and the
                # next fit / clone fails with "Can't parse parameter train_dir with value: null").
                init_params = getattr(self.est, "_init_params", None)
                for k, v in self._restore.items():
                    if isinstance(init_params, dict):
                        if v is None:
                            init_params.pop(k, None)
                        else:
                            init_params[k] = v
                    elif v is not None:
                        self.est.set_params(**{k: v})
        except Exception as e:
            logger.debug("CatBoostGpuFitGuard could not restore train_dir params: %s", e)
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        return False
