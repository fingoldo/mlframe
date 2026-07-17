"""Regression sensor for A1#6: RFECV cv_shuffle=True must override temporal auto-detect.

Pre-fix: when caller passed cv=int + cv_shuffle=True AND the input was polars with a single
monotonic datetime column (or pandas DatetimeIndex), ``_resolve_cv_and_val_cv`` silently
swapped to TimeSeriesSplit, voiding cv_shuffle.

Fix: respect explicit cv_shuffle=True; emit WARN-level log explaining that the explicit
shuffle choice overrides temporal auto-detect (and that the temporal-leakage guarantee is
consequently the caller's responsibility).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from mlframe.feature_selection.wrappers.rfecv._cv_setup import _resolve_cv_and_val_cv


def _make_pd_datetime_indexed_xy():
    n = 60
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    X = pd.DataFrame({"f0": np.arange(n), "f1": np.arange(n)[::-1] * 0.5}, index=idx)
    y = (np.arange(n) % 2).astype(int)
    return X, y


def test_rfecv_cv_shuffle_true_overrides_temporal_pandas(caplog):
    X, y = _make_pd_datetime_indexed_xy()
    est = LogisticRegression(max_iter=50)
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.wrappers.rfecv"):
        cv, _val, _ = _resolve_cv_and_val_cv(
            cv=3,
            X=X,
            y=y,
            groups=None,
            estimator=est,
            cv_shuffle=True,
            random_state=42,
            fit_params={},
            early_stopping_val_nsplits=0,
            early_stopping_rounds=0,
            _polars_time_series_hint=False,
            verbose=1,
        )
    assert not isinstance(cv, TimeSeriesSplit), f"cv_shuffle=True must NOT swap to TSS; got {cv!r}"
    assert isinstance(cv, StratifiedKFold)
    assert cv.shuffle is True
    # WARN-level message must explain the override
    warn_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("cv_shuffle=True" in m and "TimeSeriesSplit" in m for m in warn_messages), f"expected explicit cv_shuffle override warning; got {warn_messages}"


def test_rfecv_cv_shuffle_false_still_swaps_to_tss_on_temporal_pandas(caplog):
    """The opposite direction: cv_shuffle=False (default) on temporal pandas must still auto-swap."""
    X, y = _make_pd_datetime_indexed_xy()
    est = LogisticRegression(max_iter=50)
    cv, _val, _ = _resolve_cv_and_val_cv(
        cv=3,
        X=X,
        y=y,
        groups=None,
        estimator=est,
        cv_shuffle=False,
        random_state=None,
        fit_params={},
        early_stopping_val_nsplits=0,
        early_stopping_rounds=0,
        _polars_time_series_hint=False,
        verbose=0,
    )
    assert isinstance(cv, TimeSeriesSplit), f"cv_shuffle=False must auto-swap to TSS; got {cv!r}"


def test_rfecv_cv_shuffle_true_overrides_polars_hint(caplog):
    X, y = _make_pd_datetime_indexed_xy()
    est = LogisticRegression(max_iter=50)
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.wrappers.rfecv"):
        cv, _val, _ = _resolve_cv_and_val_cv(
            cv=3,
            X=X,
            y=y,
            groups=None,
            estimator=est,
            cv_shuffle=True,
            random_state=42,
            fit_params={},
            early_stopping_val_nsplits=0,
            early_stopping_rounds=0,
            _polars_time_series_hint=True,
            verbose=0,
        )
    assert not isinstance(cv, TimeSeriesSplit)
    assert isinstance(cv, StratifiedKFold)
    warn_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("cv_shuffle=True" in m for m in warn_messages)


def test_rfecv_cv_shuffle_true_overrides_timestamps_hint(caplog):
    X, y = _make_pd_datetime_indexed_xy()
    est = LogisticRegression(max_iter=50)
    ts = np.arange(len(y))
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.wrappers.rfecv"):
        cv, _val, _ = _resolve_cv_and_val_cv(
            cv=3,
            X=X,
            y=y,
            groups=None,
            estimator=est,
            cv_shuffle=True,
            random_state=42,
            fit_params={"timestamps": ts},
            early_stopping_val_nsplits=0,
            early_stopping_rounds=0,
            _polars_time_series_hint=False,
            verbose=0,
        )
    assert not isinstance(cv, TimeSeriesSplit)
    warn_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("cv_shuffle=True" in m for m in warn_messages)
