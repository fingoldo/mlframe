"""CatBoost GPU fits: no Python callbacks reach fit, the once-per-process notice fires, and the side monitor flags slow fits.

All CPU-only: the GPU fit is simulated by a CatBoostClassifier(task_type="GPU") whose ``fit`` records its kwargs, and the
monitor is driven through a fake ``train_dir`` with synthetic ``time_left.tsv`` rows and a fake clock.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import pytest

cb = pytest.importorskip("catboost")

from mlframe.training.cb import _cb_gpu_monitor as m


class _RecordingGpuCatBoost(cb.CatBoostClassifier):
    """Mimics CatBoost's GPU behaviour: any ``callbacks`` raise the real error message; records what fit received."""

    def fit(self, X, y=None, **kwargs):
        """Record the fit kwargs and params; raise CatBoost's GPU error when callbacks are passed."""
        calls = self.__dict__.setdefault("_calls", [])
        calls.append({"kwargs": dict(kwargs), "params": dict(self.get_params())})
        if kwargs.get("callbacks"):
            raise cb.CatBoostError("User defined callbacks are not supported for GPU")
        return self


class _Dummy:
    """Callback stand-in carrying only a 30-minute time budget."""
    time_budget_mins = 30


def _fit_via_training_loop(model, fit_params):
    """Fit ``model`` through the training loop's fallback wrapper on a tiny frame."""
    from mlframe.training._training_loop import _train_model_with_fallback

    X = pd.DataFrame({"a": np.arange(40, dtype=float), "b": np.arange(40, dtype=float) % 3})
    y = np.arange(40) % 2
    return _train_model_with_fallback(model, model, "CatBoostClassifier", X, y, fit_params, verbose=False)


def test_gpu_catboost_fit_never_receives_callbacks(caplog, monkeypatch):
    """A GPU CatBoost fit gets no callbacks, is attempted once per call, and the notice is logged once per process."""
    monkeypatch.setenv("MLFRAME_CB_GPU_MONITOR_S", "0")
    monkeypatch.setattr(m, "_NOTICE_LOGGED", False)
    model = _RecordingGpuCatBoost(task_type="GPU", iterations=5, early_stopping_rounds=50)
    with caplog.at_level(logging.WARNING):
        _fit_via_training_loop(model, {"callbacks": [_Dummy()]})
        _fit_via_training_loop(model, {"callbacks": [_Dummy()]})
    calls = model.__dict__["_calls"]
    # One fit attempt per training call: no guaranteed-to-fail first attempt, no retry.
    assert len(calls) == 2
    assert all("callbacks" not in c["kwargs"] for c in calls)
    notices = [r for r in caplog.records if "WITHOUT mlframe Python callbacks" in r.getMessage()]
    assert len(notices) == 1, "the GPU notice must be logged once per process, not per fit"
    msg = notices[0].getMessage()
    assert "early_stopping_rounds=50" in msg and "time budget 30 min" in msg


def test_cpu_catboost_keeps_callbacks(monkeypatch):
    """A CPU CatBoost fit keeps its callbacks and the guard stays inactive."""
    monkeypatch.setenv("MLFRAME_CB_GPU_MONITOR_S", "0")
    model = _RecordingGpuCatBoost(task_type="CPU", iterations=5)
    marker = object()
    guard = m.CatBoostGpuFitGuard(model, model, "CatBoostClassifier", {"callbacks": [marker]})
    with guard:
        assert guard.fit_params["callbacks"] == [marker]
    assert not guard.active


def test_guard_redirects_train_dir_and_restores_after_fit(tmp_path):
    """The guard redirects train_dir to a temp dir during the fit and restores params, stops the monitor and removes the dir even when the fit raises."""
    model = _RecordingGpuCatBoost(task_type="GPU", iterations=5, allow_writing_files=False)
    fit_params = {"callbacks": []}
    guard = m.CatBoostGpuFitGuard(model, model, "CatBoostClassifier", fit_params, interval_s=3600)
    with pytest.raises(RuntimeError):
        with guard:
            p = model.get_params()
            assert p["allow_writing_files"] is True and os.path.isdir(p["train_dir"])
            tmp_dir = p["train_dir"]
            assert guard.monitor is not None and guard.monitor.alive
            raise RuntimeError("fit blew up")
    assert not guard.monitor.alive, "monitor thread must stop even when the fit raises"
    p = model.get_params()
    assert p["allow_writing_files"] is False and "train_dir" not in p
    assert not os.path.exists(tmp_dir)


def test_two_guards_get_distinct_train_dirs():
    """Two concurrent guards get distinct temp train_dirs."""
    a = _RecordingGpuCatBoost(task_type="GPU", iterations=5)
    b = _RecordingGpuCatBoost(task_type="GPU", iterations=5)
    with m.CatBoostGpuFitGuard(a, a, "A", {}, interval_s=3600), m.CatBoostGpuFitGuard(b, b, "B", {}, interval_s=3600):
        assert a.get_params()["train_dir"] != b.get_params()["train_dir"]


# ----------------------------------------------------------------------------------------------------------------------
# Monitor on synthetic progress files
# ----------------------------------------------------------------------------------------------------------------------


class _Clock:
    """Manually advanced monotonic clock."""
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _write_rows(d, rows, partial=None):
    """Write a CatBoost ``time_left.tsv`` with ``rows`` and an optional trailing partial row."""
    with open(os.path.join(d, "time_left.tsv"), "w", encoding="utf-8") as f:
        f.write("iter\tPassed\tRemaining\n")
        for it, passed_ms, rem_ms in rows:
            f.write(f"{it}\t{passed_ms}\t{rem_ms}\n")
        if partial:
            f.write(partial)


def _fake_gpu():
    """GPU snapshot with a near-full VRAM GPU and another process holding memory."""
    return {
        "gpus": [{"index": 0, "util_pct": 97.0, "mem_used_mb": 3900.0, "mem_total_mb": 4096.0}],
        "processes": [{"gpu": 0, "pid": 424242, "name": "other_trainer.exe", "mem_mb": 2500.0}],
    }


def test_read_time_left_tail_skips_partial_row(tmp_path):
    """The tail reader skips a partially written last row and returns None for a missing file."""
    _write_rows(str(tmp_path), [(0, 100, 5000), (1, 200, 4000)], partial="2\t30")
    assert m.read_time_left_tail(str(tmp_path)) == (1, 0.2, 4.0)
    assert m.read_time_left_tail(str(tmp_path / "missing")) is None


def test_monitor_detects_throughput_collapse_and_names_other_gpu_process(tmp_path, caplog):
    """A drop from 10 it/s to 0.2 it/s is flagged as a collapse naming the other GPU process and VRAM pressure."""
    d = str(tmp_path)
    clock = _Clock()
    mon = m.CatBoostGpuFitMonitor(d, interval_s=60, label="CB", total_iterations=5000, clock=clock, gpu_probe=_fake_gpu)
    mon.start()  # interval>0 starts a thread; stop it and drive polls by hand
    mon.stop()
    rows = []
    it = 0
    with caplog.at_level(logging.INFO, logger=m.logger.name):
        # 3 healthy minutes at 10 it/s ...
        for _ in range(3):
            clock.t += 60
            it += 600
            rows.append((it, int((clock.t - 1000) * 1000), 100000))
            _write_rows(d, rows)
            mon.poll_once()
        assert not mon.warnings
        # ... then 0.2 it/s.
        clock.t += 60
        it += 12
        rows.append((it, int((clock.t - 1000) * 1000), 9000000))
        _write_rows(d, rows)
        stats = mon.poll_once()
    assert stats["rate"] == pytest.approx(0.2)
    assert any("THROUGHPUT COLLAPSE" in w and "other_trainer.exe[424242]" in w and "VRAM pressure" in w for w in mon.warnings)
    assert any("[cb-gpu-monitor] CB: iter=" in r.getMessage() for r in caplog.records)


def test_monitor_detects_stall_and_budget_overrun(tmp_path):
    """Repeated polls without progress warn STALLED, and the time-budget overrun is warned exactly once."""
    d = str(tmp_path)
    clock = _Clock()
    mon = m.CatBoostGpuFitMonitor(d, interval_s=60, label="CB", time_budget_s=150, clock=clock, gpu_probe=lambda: None)
    _write_rows(d, [(100, 60000, 1000)])
    for _ in range(4):
        clock.t += 60
        mon.poll_once()
    assert any("STALLED" in w for w in mon.warnings)
    assert sum("EXCEEDED the configured time budget" in w for w in mon.warnings) == 1


def test_monitor_no_file_and_broken_probe_never_raise(tmp_path):
    """A missing progress file and a raising GPU probe make poll_once return None instead of raising."""
    def _boom():
        """GPU probe that always fails."""
        raise OSError("nvml gone")

    mon = m.CatBoostGpuFitMonitor(str(tmp_path / "nope"), interval_s=60, clock=_Clock(), gpu_probe=_boom)
    assert mon.poll_once() is None


def test_monitor_thread_polls_and_stops(tmp_path):
    """The monitor thread polls on its own and stops when asked."""
    _write_rows(str(tmp_path), [(5, 1000, 1000)])
    mon = m.CatBoostGpuFitMonitor(str(tmp_path), interval_s=0.02, gpu_probe=lambda: None).start()
    import time

    deadline = time.time() + 5
    while mon.polls < 2 and time.time() < deadline:
        time.sleep(0.01)
    mon.stop()
    assert mon.polls >= 2 and not mon.alive


def test_guard_restores_params_on_really_fitted_model(monkeypatch):
    """set_params is refused on a fitted CatBoost model; the restore must still leave no temp train_dir behind."""
    monkeypatch.setattr(m, "cb_model_is_gpu", lambda est: True)
    model = cb.CatBoostClassifier(iterations=5, verbose=0, allow_writing_files=False)
    X = np.random.default_rng(0).random((200, 3))
    y = np.arange(200) % 2
    fit_params = {"callbacks": [object()]}
    with m.CatBoostGpuFitGuard(model, model, "CatBoostClassifier", fit_params, interval_s=3600):
        model.fit(X, y, **fit_params)
    p = model.get_params()
    assert p["allow_writing_files"] is False and "train_dir" not in p
    from sklearn.base import clone

    clone(model).fit(X, y)  # a clone must fit without a dangling/None train_dir


def test_monitor_pickles_to_a_stopped_copy(tmp_path):
    """The monitor holds a threading.Event; pickling it must yield a stopped copy instead of raising TypeError."""
    import pickle

    mon = m.CatBoostGpuFitMonitor(str(tmp_path), interval_s=0)
    clone = pickle.loads(pickle.dumps(mon))
    assert clone.train_dir == str(tmp_path)
    assert clone._stop.is_set() and not clone.alive


def test_monitor_poll_failure_is_logged_not_silent(tmp_path, caplog, monkeypatch):
    """A failing poll is logged at WARNING instead of disappearing."""
    mon = m.CatBoostGpuFitMonitor(str(tmp_path), interval_s=0, gpu_probe=lambda: None)

    def _broken():
        """poll_once replacement that always fails."""
        raise RuntimeError("poll exploded")

    monkeypatch.setattr(mon, "poll_once", _broken)
    with caplog.at_level(logging.WARNING, logger=m.logger.name):
        mon._safe_poll()
    assert any("poll exploded" in r.getMessage() for r in caplog.records)


def test_guard_pickles_to_an_inactive_copy():
    """The guard holds a threading.Lock; pickling it must yield a usable copy with a fresh lock instead of raising."""
    import pickle

    model = _RecordingGpuCatBoost(task_type="CPU", iterations=5)
    guard = m.CatBoostGpuFitGuard(model, model, "CatBoostClassifier", {}, interval_s=0)
    clone = pickle.loads(pickle.dumps(guard))
    assert clone.monitor is None and clone.model_type_name == "CatBoostClassifier"
    clone.set_fit_running(True)
    assert clone._fit_running
