"""CatBoost GPU fits: a time budget / runaway stop that keeps the model trained so far.

CatBoost rejects callbacks on GPU and an interrupted fit is left unfitted. The guard runs the fit with snapshots on; on a
limit the monitor interrupts it and the fit resumes from the snapshot with ``iterations`` capped at the iterations done.
These run a REAL CatBoost fit on CPU with the guard's GPU check patched to True (this box has no GPU CatBoost): the
snapshot / interrupt / resume mechanics are CatBoost's own and identical on CPU. depth=8 matters: there CatBoost derives
max_ctr_complexity from the planned iteration count, so a capped resume fails unless the snapshot's value is pinned.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

catboost = pytest.importorskip("catboost")

import mlframe.training.cb._cb_gpu_monitor as mon
from mlframe.training.cb._cb_gpu_budget import fit_with_cb_gpu_guard, snapshot_params


def _data(n=40_000, seed=0):
    """Train / validation regression split large enough that a deep CatBoost fit runs for many seconds."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12))
    y = 2 * X[:, 0] + np.sin(X[:, 1]) + rng.normal(size=n)
    return X[: int(n * 0.8)], y[: int(n * 0.8)], X[int(n * 0.8) :], y[int(n * 0.8) :]


def _plain_fit(model, model_obj, name, X, y, fit_params, verbose=False):
    """Stand-in for the unguarded training-loop fit: a plain ``model.fit``."""
    model.fit(X, y, **fit_params)
    return model


@pytest.fixture
def fake_gpu(monkeypatch):
    """Treat every CatBoost fit as a GPU fit, with a fast monitor and 1 s snapshots."""
    monkeypatch.setattr(mon, "cb_model_is_gpu", lambda est: True)
    monkeypatch.setenv("MLFRAME_CB_GPU_MONITOR_S", "0.5")
    monkeypatch.setenv("MLFRAME_CB_GPU_SNAPSHOT_S", "1")
    monkeypatch.setattr(mon, "_NOTICE_LOGGED", False)


@pytest.mark.usefixtures("fake_gpu")
def test_time_budget_stops_fit_and_keeps_model(caplog):
    """Budget exceeded -> interrupted -> resumed from snapshot: a FITTED model, far fewer trees, original params restored."""
    Xt, yt, Xv, yv = _data()
    model = catboost.CatBoostRegressor(iterations=100_000, learning_rate=0.02, depth=8, od_type="Iter", od_wait=90_000,
                                       use_best_model=True, verbose=0, thread_count=2)
    budget_cb = SimpleNamespace(time_budget_mins=5 / 60.0)  # 5 s; stripped by the guard, its budget is read
    fit_params = {"eval_set": (Xv, yv), "callbacks": [budget_cb]}
    out = fit_with_cb_gpu_guard(_plain_fit, model, model, "CatBoostRegressor", Xt, yt, fit_params)
    assert out is model and model.is_fitted()
    assert 0 < model.tree_count_ < 100_000
    # Stopped by the budget rather than by a fast machine: the resume log names the time budget as the reason.
    assert any("resuming from its snapshot" in r.getMessage() and "time budget" in r.getMessage() for r in caplog.records)
    r2 = 1 - np.mean((model.predict(Xv) - yv) ** 2) / np.var(yv)
    assert r2 > 0.3, f"the kept model is not a usable fit (R2={r2:.3f})"
    p = model.get_params()
    assert p.get("iterations") == 100_000, "the capped iteration count leaked into the model's params (clones would be capped)"
    for k in ("save_snapshot", "snapshot_file", "snapshot_interval", "max_ctr_complexity", "train_dir"):
        assert p.get(k) is None, f"{k} was not restored: {p.get(k)!r}"
    assert any("resuming from its snapshot" in r.getMessage() for r in caplog.records)


@pytest.mark.usefixtures("fake_gpu")
def test_genuine_keyboard_interrupt_still_propagates(monkeypatch):
    """A Ctrl+C that the monitor did not issue must not be swallowed as a budget stop."""
    import _thread

    monkeypatch.setenv("MLFRAME_CB_GPU_RUNAWAY_FACTOR", "0")
    Xt, yt, Xv, yv = _data()
    model = catboost.CatBoostRegressor(iterations=100_000, learning_rate=0.02, depth=6, verbose=0, thread_count=2)
    threading.Timer(3.0, _thread.interrupt_main).start()
    with pytest.raises(KeyboardInterrupt):
        fit_with_cb_gpu_guard(_plain_fit, model, model, "CatBoostRegressor", Xt, yt, {"eval_set": (Xv, yv)})


def test_no_interrupt_before_a_snapshot_exists(tmp_path):
    """Interrupting with nothing to resume from would lose the model: the request is declined and retried later."""
    guard = mon.CatBoostGpuFitGuard(None, SimpleNamespace(get_params=lambda: {}), "CatBoostRegressor", {})
    guard.snapshot_file = str(tmp_path / "missing.snap")
    guard.set_fit_running(True)
    assert guard._request_interrupt("time budget") is False
    assert guard.limit_reason is None


def test_monitor_runaway_rule_requests_a_stop(tmp_path):
    """Elapsed beyond runaway_factor x the full-budget projection (and > 5 min) triggers on_limit, retried until accepted."""
    calls = []
    clock = {"t": 0.0}
    m = mon.CatBoostGpuFitMonitor(str(tmp_path), interval_s=60, total_iterations=1000, clock=lambda: clock["t"],
                                  gpu_probe=lambda: None, on_limit=lambda reason: calls.append(reason) or len(calls) >= 2,
                                  runaway_factor=3.0)
    m.start()
    m.stop()
    tl = tmp_path / "time_left.tsv"
    rows = ["iter\tPassed\tRemaining"]
    # early: 10 it/s -> the whole 1000-iteration budget would take 100 s; runaway beyond 300 s AND past the 300 s floor
    for t, it in ((10, 100), (20, 200), (400, 230), (460, 240), (520, 250)):
        rows.append(f"{it}\t{t * 1000}\t1000")
        tl.write_text("\n".join(rows) + "\n")
        clock["t"] = float(t)
        m.poll_once()
    assert len(calls) == 2 and "runaway" in calls[0], calls  # first declined (e.g. no snapshot yet), then accepted, then silent


def test_snapshot_params_reads_catboost_serialised_params(tmp_path):
    """The resume pins max_ctr_complexity from the snapshot's own params: the reader must find them in a real snapshot."""
    Xt, yt, _, _ = _data(n=5_000)
    snap = str(tmp_path / "s.snap")
    catboost.CatBoostRegressor(iterations=20, depth=4, verbose=0, save_snapshot=True, snapshot_file=snap, train_dir=str(tmp_path / "td")).fit(Xt, yt)
    params = snapshot_params(snap)
    assert params is not None and "max_ctr_complexity" in (params.get("cat_feature_params") or {})


@pytest.mark.usefixtures("fake_gpu")
def test_the_resumed_fit_does_not_run_under_the_guards_interrupt():
    """A production traceback showed a later, unrelated TypeError "During handling of" a KeyboardInterrupt: the resume ran
    inside the handler, so everything it raised carried the guard's interrupt as context and read as a user's Ctrl+C."""
    Xt, yt, Xv, yv = _data()
    model = catboost.CatBoostRegressor(iterations=100_000, learning_rate=0.02, depth=8, od_type="Iter", od_wait=90_000,
                                       use_best_model=True, verbose=0, thread_count=2)
    calls = {"n": 0}

    def _fit_then_fail_on_resume(model, model_obj, name, X, y, fit_params, verbose=False):
        calls["n"] += 1
        if calls["n"] > 1:
            raise ValueError("the resumed fit failed")
        model.fit(X, y, **fit_params)
        return model

    fit_params = {"eval_set": (Xv, yv), "callbacks": [SimpleNamespace(time_budget_mins=5 / 60.0)]}
    with pytest.raises(ValueError) as excinfo:
        fit_with_cb_gpu_guard(_fit_then_fail_on_resume, model, model, "CatBoostRegressor", Xt, yt, fit_params)
    assert calls["n"] == 2, "the budget stop must have triggered a resume"
    assert not isinstance(excinfo.value.__context__, KeyboardInterrupt)
