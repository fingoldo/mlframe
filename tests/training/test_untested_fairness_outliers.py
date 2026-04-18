"""Tests for _compute_fairness_subgroups and _apply_outlier_detection_global."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.core import (
    _apply_outlier_detection_global,
    _compute_fairness_subgroups,
)
from mlframe.training.configs import TrainingBehaviorConfig


# ----- _compute_fairness_subgroups -----

def test_fairness_no_features_returns_none():
    bc = TrainingBehaviorConfig()  # fairness_features default None
    df = pd.DataFrame({"a": [1, 2, 3]})
    subs, feats = _compute_fairness_subgroups(df, bc)
    assert subs is None
    assert feats == []


def test_fairness_categorical_feature_pandas():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "grp": rng.choice(["A", "B"], size=n),
        "x": rng.standard_normal(n),
    })
    bc = TrainingBehaviorConfig(
        fairness_features=["grp"],
        cont_nbins=3,
        fairness_min_pop_cat_thresh=1,
    )
    subs, feats = _compute_fairness_subgroups(df, bc)
    assert subs is not None
    assert feats == ["grp"]


def test_fairness_missing_columns_skipped():
    # Only special markers -> df_subset is empty frame but function still runs
    bc = TrainingBehaviorConfig(
        fairness_features=["**ORDER**"],
        cont_nbins=3,
        fairness_min_pop_cat_thresh=1,
    )
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    subs, feats = _compute_fairness_subgroups(df, bc)
    # special marker triggers create_fairness_subgroups path; subs may be dict
    assert feats == ["**ORDER**"]


def test_fairness_polars_input():
    rng = np.random.default_rng(1)
    n = 150
    df = pl.DataFrame({
        "grp": rng.choice(["x", "y", "z"], size=n).tolist(),
        "v": rng.standard_normal(n).tolist(),
    })
    bc = TrainingBehaviorConfig(
        fairness_features=["grp"],
        cont_nbins=3,
        fairness_min_pop_cat_thresh=1,
    )
    subs, feats = _compute_fairness_subgroups(df, bc)
    assert subs is not None


# ----- _apply_outlier_detection_global -----

class _MockOD:
    """Trivial mock outlier detector — marks final row as outlier."""
    def fit(self, X):
        self._n = len(X)
        return self

    def predict(self, X):
        n = len(X)
        out = np.ones(n, dtype=int)
        if n > 0:
            out[-1] = -1  # last sample is outlier
        return out


def test_outlier_none_returns_inputs_unchanged():
    df = pd.DataFrame({"a": [1, 2, 3]})
    idx = np.arange(3)
    tr, va, tr_idx, va_idx, tr_mask, va_mask = _apply_outlier_detection_global(
        train_df=df, val_df=None, train_idx=idx, val_idx=None,
        outlier_detector=None, od_val_set=False, verbose=False,
    )
    assert tr is df and va is None
    assert tr_mask is None and va_mask is None


def test_outlier_filters_train_pandas():
    rng = np.random.default_rng(0)
    n = 20
    df = pd.DataFrame({"a": rng.standard_normal(n)})
    idx = np.arange(n)
    tr, va, tr_idx, va_idx, tr_mask, va_mask = _apply_outlier_detection_global(
        train_df=df, val_df=None, train_idx=idx, val_idx=None,
        outlier_detector=_MockOD(), od_val_set=False, verbose=False,
    )
    assert len(tr) == n - 1
    assert tr_mask.sum() == n - 1
    assert tr_mask[-1] == False
    assert len(tr_idx) == n - 1


def test_outlier_filters_val_when_od_val_set():
    n_tr, n_va = 10, 6
    tr_df = pd.DataFrame({"a": np.arange(n_tr, dtype=float)})
    va_df = pd.DataFrame({"a": np.arange(n_va, dtype=float)})
    tr_idx = np.arange(n_tr)
    va_idx = np.arange(n_tr, n_tr + n_va)
    tr, va, tr_i, va_i, tr_m, va_m = _apply_outlier_detection_global(
        train_df=tr_df, val_df=va_df, train_idx=tr_idx, val_idx=va_idx,
        outlier_detector=_MockOD(), od_val_set=True, verbose=False,
    )
    assert len(tr) == n_tr - 1
    assert len(va) == n_va - 1
    assert va_m is not None
    assert va_m.sum() == n_va - 1


def test_outlier_skips_val_when_flag_false():
    n_tr, n_va = 8, 4
    tr_df = pd.DataFrame({"a": np.arange(n_tr, dtype=float)})
    va_df = pd.DataFrame({"a": np.arange(n_va, dtype=float)})
    tr, va, tr_i, va_i, tr_m, va_m = _apply_outlier_detection_global(
        train_df=tr_df, val_df=va_df,
        train_idx=np.arange(n_tr), val_idx=np.arange(n_tr, n_tr + n_va),
        outlier_detector=_MockOD(), od_val_set=False, verbose=False,
    )
    # val untouched
    assert va is va_df
    assert len(va) == n_va
    assert va_m is None


def test_outlier_polars_path():
    n = 12
    tr_df = pl.DataFrame({"a": list(range(n))})
    tr_idx = np.arange(n)
    tr, va, tr_i, va_i, tr_m, va_m = _apply_outlier_detection_global(
        train_df=tr_df, val_df=None, train_idx=tr_idx, val_idx=None,
        outlier_detector=_MockOD(), od_val_set=False, verbose=False,
    )
    assert isinstance(tr, pl.DataFrame)
    assert tr.height == n - 1


# ----- Catastrophic-rejection guard -----
# Sensor for: misconfigured outlier detector (contamination too high, sign
# convention bug) used to silently filter train_df to 0 rows. Downstream
# CatBoost/LightGBM then crashed 5+ minutes later with opaque errors.
# Guard must raise ValueError immediately with a diagnostic message.


class _RejectAllOD:
    """Mock OD that marks every sample as an outlier."""
    def fit(self, X):
        self._n = len(X)

    def predict(self, X):
        return np.full(len(X), -1, dtype=int)


class _RejectAllButOneOD:
    """Mock OD that keeps exactly one sample (below 1% threshold when n>>100)."""
    def fit(self, X):
        self._n = len(X)

    def predict(self, X):
        out = np.full(len(X), -1, dtype=int)
        out[0] = 1
        return out


def test_outlier_all_rejected_raises():
    """100% of train flagged as outliers -> loud ValueError."""
    n = 200
    df = pd.DataFrame({"a": np.arange(n, dtype=float)})
    idx = np.arange(n)
    with pytest.raises(ValueError, match="Outlier detector rejected"):
        _apply_outlier_detection_global(
            train_df=df, val_df=None, train_idx=idx, val_idx=None,
            outlier_detector=_RejectAllOD(), od_val_set=False, verbose=False,
        )


def test_outlier_below_one_percent_raises():
    """Keeping <1% of rows also trips the guard (not just 0 rows)."""
    n = 500  # 1% = 5; keeping 1 row is below
    df = pd.DataFrame({"a": np.arange(n, dtype=float)})
    idx = np.arange(n)
    with pytest.raises(ValueError, match="likely misconfigured"):
        _apply_outlier_detection_global(
            train_df=df, val_df=None, train_idx=idx, val_idx=None,
            outlier_detector=_RejectAllButOneOD(), od_val_set=False, verbose=False,
        )


def test_outlier_all_rejected_polars_also_raises():
    """Polars path must hit the same guard — the check runs before the filter."""
    n = 300
    df = pl.DataFrame({"a": list(range(n))})
    idx = np.arange(n)
    with pytest.raises(ValueError, match="Outlier detector rejected"):
        _apply_outlier_detection_global(
            train_df=df, val_df=None, train_idx=idx, val_idx=None,
            outlier_detector=_RejectAllOD(), od_val_set=False, verbose=False,
        )


def test_outlier_error_message_includes_numbers():
    """Diagnostic must name the counts so operators can see the scale."""
    n = 150
    df = pd.DataFrame({"a": np.arange(n, dtype=float)})
    idx = np.arange(n)
    with pytest.raises(ValueError) as excinfo:
        _apply_outlier_detection_global(
            train_df=df, val_df=None, train_idx=idx, val_idx=None,
            outlier_detector=_RejectAllOD(), od_val_set=False, verbose=False,
        )
    msg = str(excinfo.value)
    assert "150" in msg  # rejected count or input count
    assert "0" in msg    # kept count
    assert "contamination" in msg or "misconfigured" in msg


def test_outlier_drops_one_row_does_not_trip_guard():
    """Boundary sanity: normal single-row rejection must NOT raise."""
    n = 200
    df = pd.DataFrame({"a": np.arange(n, dtype=float)})
    idx = np.arange(n)
    tr, va, tr_i, va_i, tr_m, va_m = _apply_outlier_detection_global(
        train_df=df, val_df=None, train_idx=idx, val_idx=None,
        outlier_detector=_MockOD(), od_val_set=False, verbose=False,
    )
    assert len(tr) == n - 1  # dropped exactly one; guard did not fire
