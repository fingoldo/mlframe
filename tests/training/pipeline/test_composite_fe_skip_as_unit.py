"""Regression: a failure in any split skips the ma-crossover / event-proximity / nearest-past-join step for ALL splits."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.training._preprocessing_configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline import _event_proximity_decay_composite_fe as ev
from mlframe.training.pipeline import _ma_crossover_composite_fe as ma
from mlframe.training.pipeline import _nearest_past_join_composite_fe as npj


def _fail_on_second_call(real):
    """Wrap ``real`` so the second call (the val split) raises."""
    state = {"n": 0}

    def wrapper(*args, **kwargs):
        """Delegate to the real function except on the second call."""
        state["n"] += 1
        if state["n"] == 2:
            raise RuntimeError("boom on val")
        return real(*args, **kwargs)

    return wrapper


def _run_ma(monkeypatch):
    """Run the ma-crossover step with a val-split failure."""
    monkeypatch.setattr(ma, "ma_crossover_features", _fail_on_second_call(ma.ma_crossover_features))
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"val_col": rng.normal(size=300).cumsum()})
    g, ts = rng.integers(0, 10, 300), np.arange(300, dtype=np.float64)
    cfg = PreprocessingExtensionsConfig(ma_crossover_columns=["val_col"], ma_crossover_windows=[3, 5])
    ti, vi, si = np.arange(0, 200), np.arange(200, 250), np.arange(250, 300)
    frames = [df.iloc[i].reset_index(drop=True) for i in (ti, vi, si)]
    return frames, ma.apply_ma_crossover_composite_fe(*frames, cfg, g, ts, ti, vi, si, verbose=0)


def _run_ev(monkeypatch):
    """Run the event-proximity step with a val-split failure."""
    monkeypatch.setattr(ev, "event_proximity_decay_features", _fail_on_second_call(ev.event_proximity_decay_features))
    df = pd.DataFrame({"dummy": np.arange(200)})
    ts = np.asarray(pd.date_range("2024-01-01", periods=200, freq="D"))
    cfg = PreprocessingExtensionsConfig(event_proximity_decay_event_dates=["2024-01-15", "2024-03-01"], event_proximity_decay_cap=10)
    ti, vi, si = np.arange(0, 150), np.arange(150, 175), np.arange(175, 200)
    frames = [df.iloc[i].reset_index(drop=True) for i in (ti, vi, si)]
    return frames, ev.apply_event_proximity_decay_composite_fe(*frames, cfg, ts, ti, vi, si, verbose=0)


def _run_npj(monkeypatch):
    """Run the nearest-past-join step with a val-split failure."""
    monkeypatch.setattr(npj, "nearest_past_join", _fail_on_second_call(npj.nearest_past_join))
    rng = np.random.default_rng(0)
    right = pd.DataFrame({"entity": np.repeat(np.arange(20), 10), "t": np.tile(np.arange(10), 20), "known_value": rng.normal(size=200)})
    left = pd.DataFrame({"entity": rng.integers(0, 20, 100), "t": rng.integers(0, 10, 100).astype(float) + 0.5})
    cfg = PreprocessingExtensionsConfig(nearest_past_join_on="t", nearest_past_join_by=["entity"], nearest_past_join_value_cols=["known_value"])
    frames = [left.iloc[:60].reset_index(drop=True), left.iloc[60:80].reset_index(drop=True), left.iloc[80:].reset_index(drop=True)]
    return frames, npj.apply_nearest_past_join_composite_fe(*frames, cfg, right, verbose=0)


@pytest.mark.parametrize("runner", [_run_ma, _run_ev, _run_npj])
def test_composite_fe_skip_as_unit_val_failure_leaves_all_splits_unchanged(runner, monkeypatch, caplog):
    """A val-split failure must leave train, val and test with identical, original column sets, logged at WARNING."""
    with caplog.at_level(logging.WARNING):
        frames, out = runner(monkeypatch)
    for before, after in zip(frames, out):
        assert list(after.columns) == list(before.columns)
    assert any(r.levelno == logging.WARNING and "skipping the step for all splits" in r.getMessage() for r in caplog.records)
