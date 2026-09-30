"""Unit + integration tests for the index-only chronological reorder of the train split (``TrainingSplitConfig.chronological_train_order``)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training import _chronological_order as co
from mlframe.training._chronological_order import chronological_order_index
from mlframe.training._preprocessing_configs import TrainingSplitConfig
from mlframe.training.core._phase_helpers_fit_split import _phase_train_val_test_split


def _dt(n, seed=0):
    """Hourly datetime64[ns] timestamps."""
    base = np.datetime64("2024-01-01T00:00:00", "ns")
    return base + (np.arange(n) * 3_600_000_000_000).astype("timedelta64[ns]")


def test_already_sorted_returns_same_object_and_skips_argsort(monkeypatch):
    """Already sorted returns same object and skips argsort."""
    idx = np.array([0, 2, 3, 7, 9])
    ts = _dt(10)

    def _boom(*a, **k):
        """Fail if argsort is called."""
        raise AssertionError("argsort must not run on an already-chronological index")

    monkeypatch.setattr(co.np, "argsort", _boom)
    out, status = chronological_order_index(idx, ts)
    assert out is idx and status == "sorted"


@pytest.mark.parametrize("kind", ["numpy", "pandas", "polars", "int", "pandas_tz"])
def test_unsorted_becomes_chronological_all_timestamp_containers(kind):
    """Unsorted becomes chronological all timestamp containers."""
    n = 12
    perm = np.random.default_rng(1).permutation(n)
    ts = _dt(n)[perm]
    if kind == "pandas":
        ts = pd.Series(ts)
    elif kind == "pandas_tz":
        ts = pd.Series(ts).dt.tz_localize("UTC")
    elif kind == "polars":
        ts = pl.Series(ts)
    elif kind == "int":
        ts = perm.astype(np.int64)
    idx = np.array([0, 1, 2, 3, 5, 8, 9, 11])
    out, status = chronological_order_index(idx, ts)
    assert status == "reordered"
    assert sorted(out.tolist()) == idx.tolist()
    ranks = perm[out]
    assert np.all(np.diff(ranks) > 0)


def test_ties_are_stable_in_row_order():
    """Ties are stable in row order."""
    ts = np.array([5, 1, 5, 1, 3, 5])
    out, _ = chronological_order_index(np.arange(6), ts)
    assert out.tolist() == [1, 3, 4, 0, 2, 5]


def test_missing_timestamps_go_last_stable():
    """Missing timestamps go last stable."""
    ts = np.array(["2024-01-03", "NaT", "2024-01-01", "NaT", "2024-01-02"], dtype="datetime64[ns]")
    out, status = chronological_order_index(np.arange(5), ts)
    assert status == "reordered" and out.tolist() == [2, 4, 0, 1, 3]
    fl = np.array([3.0, np.nan, 1.0, np.nan, 2.0])
    out, _ = chronological_order_index(np.arange(5), fl)
    assert out.tolist() == [2, 4, 0, 1, 3]
    # NaN already trailing -> counts as sorted
    _, status = chronological_order_index(np.arange(3), np.array([1.0, 2.0, np.nan]))
    assert status == "sorted"


def test_skipped_when_no_timestamps_or_misaligned():
    """Skipped when no timestamps or misaligned."""
    idx = np.array([3, 1, 2])
    out, status = chronological_order_index(idx, None)
    assert out is idx and status == "skipped"
    assert chronological_order_index(idx, np.arange(2))[1] == "skipped"  # timestamps shorter than the largest index


class _Cfg:
    """Split-config stand-in holding the chronological_train_order flag and a real config dump."""
    def __init__(self, order=True):
        self.chronological_train_order = order
        self.d = TrainingSplitConfig(test_size=0.2, val_size=0.2, chronological_train_order=order).model_dump()

    def model_dump(self, exclude=None):
        """Return a copy of the stored config dict."""
        return dict(self.d)

    def __getattr__(self, name):
        try:
            return self.__dict__["d"][name]
        except KeyError as e:
            raise AttributeError(name) from e


def _run_phase(df, ts, order, tmp_path=None):
    """Run the train/val/test split phase with the ordering flag and return its result."""
    n = len(df)
    y = np.arange(n, dtype=float)
    md: dict = {}
    res = _phase_train_val_test_split(
        df=df, target_by_type={"REGRESSION": {"y": y}}, timestamps=ts, group_ids=None, group_ids_raw=None, artifacts=None, sequences=None,
        split_config=TrainingSplitConfig(test_size=0.2, val_size=0.2, chronological_train_order=order), behavior_config=type("B", (), {"fairness_features": None})(),
        metadata=md, data_dir=None, models_dir=None, target_name="y", model_name="m", df_size_mb=1.0, verbose=False,
    )
    return res, md


@pytest.mark.parametrize("frame", ["pandas", "polars"])
def test_real_split_orders_train_chronologically_and_keeps_membership(frame):
    """Real split orders train chronologically and keeps membership."""
    n = 400
    perm = np.random.default_rng(3).permutation(n)
    ts = _dt(n)[perm]
    data = {"id": np.arange(n), "x": np.random.default_rng(4).normal(size=n)}
    df = pd.DataFrame(data) if frame == "pandas" else pl.DataFrame(data)
    on, md_on = _run_phase(df, ts, True)
    off, md_off = _run_phase(df, ts, False)
    assert md_on["train_chronological_order"] == "reordered" and "train_chronological_order" not in md_off
    t_on = np.asarray(on.train_idx)
    assert np.all(np.diff(ts[t_on].astype(np.int64)) >= 0)
    # membership of every split identical, val/test order untouched
    assert set(t_on.tolist()) == set(np.asarray(off.train_idx).tolist())
    np.testing.assert_array_equal(on.val_idx, off.val_idx)
    np.testing.assert_array_equal(on.test_idx, off.test_idx)
    assert np.all(np.diff(np.asarray(off.train_idx)) > 0)  # opt-out keeps source-row order
    # the taken train frame follows the reordered index (row alignment by id)
    ids = np.asarray(on.train_df["id"]) if frame == "polars" else on.train_df["id"].to_numpy()
    np.testing.assert_array_equal(ids, t_on)


def test_chronological_frame_costs_only_the_check():
    """Chronological frame costs only the check."""
    n = 200
    ts = _dt(n)
    _, md = _run_phase(pd.DataFrame({"x": np.arange(n)}), ts, True)
    assert md["train_chronological_order"] == "sorted"


def test_default_is_on():
    """Default is on."""
    assert TrainingSplitConfig().chronological_train_order is True
