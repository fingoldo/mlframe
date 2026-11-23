"""Cluster 37 regression: NaN-gain specs are dropped by the honest-gain floor, and a skipped back-merge is logged (audit 2026-11-20, findings 37-F26, 37-F29)."""

import logging
from types import SimpleNamespace

import pandas as pd

from mlframe.training.core._calib_frame_replay import _back_merge_new_columns
from mlframe.training.core._phase_composite_discovery_gates import _drop_below_honest_gain_floor


def _cfg():
    """Discovery config with a constant floor and no noise multiplier."""
    return SimpleNamespace(min_honest_gain_to_train=0.001, min_honest_gain_z=0.0)


def test_nan_gain_spec_is_dropped_not_trained():
    """A spec whose honest RMSE gain is NaN does not pass the floor (37-F26)."""
    pending = [{"name": "nan_spec", "rmse_gain": True, "gain": float("nan")}, {"name": "good", "rmse_gain": True, "gain": 0.05}]
    kept = [p["name"] for p in _drop_below_honest_gain_floor(pending, _cfg())]
    assert kept == ["good"]


def test_gain_above_floor_is_kept_control():
    """Control: a real gain above the constant floor is kept (37-F26)."""
    pending = [{"name": "good", "rmse_gain": True, "gain": 0.05}]
    assert [p["name"] for p in _drop_below_honest_gain_floor(pending, _cfg())] == ["good"]


def test_skipped_back_merge_logs_a_warning(caplog):
    """When row counts differ the merge is skipped with a WARNING naming the dropped columns (37-F29)."""
    pre = pd.DataFrame({"a": [1, 2, 3]})
    post = pd.DataFrame({"a": [1, 2, 3, 4], "eng": [0.1, 0.2, 0.3, 0.4]})
    with caplog.at_level(logging.WARNING):
        out = _back_merge_new_columns(pre, post)
    assert list(out.columns) == ["a"]
    assert any("row count changed" in r.getMessage() and "eng" in r.getMessage() for r in caplog.records)
