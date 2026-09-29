"""RFECV ``cv`` resolution: splitter names / classes / instances, the early-stopping ``val_cv`` and timestamp-ordered folds."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold, LeaveOneOut, StratifiedKFold, TimeSeriesSplit

from mlframe.feature_selection.wrappers.rfecv._cv_setup import _resolve_cv_and_val_cv
from mlframe.feature_selection.wrappers.rfecv._timestamp_ordered_split import TimestampOrderedSplit

LOGGER = "mlframe.feature_selection.wrappers.rfecv"


def _resolve(cv, *, n=40, estimator=None, groups=None, fit_params=None, cv_shuffle=False, val_nsplits=10):
    X = pd.DataFrame({"a": np.arange(n, dtype=float), "b": np.arange(n, dtype=float) % 7})
    y = np.arange(n) % 2
    return _resolve_cv_and_val_cv(
        cv=cv, X=X, y=y, groups=groups, estimator=estimator if estimator is not None else LogisticRegression(),
        cv_shuffle=cv_shuffle, random_state=0, fit_params=fit_params if fit_params is not None else {},
        early_stopping_val_nsplits=val_nsplits, early_stopping_rounds=None, _polars_time_series_hint=False, verbose=0,
    )


@pytest.mark.parametrize(
    "cv",
    [KFold(n_splits=4), KFold(n_splits=4, shuffle=True, random_state=3), StratifiedKFold(n_splits=4), TimeSeriesSplit(n_splits=4, gap=2), GroupKFold(n_splits=4)],
    ids=["kfold", "kfold_shuffled", "stratified", "tss_gap", "groupkfold"],
)
def test_sklearn_splitter_instance_builds_val_cv_without_warning(cv, caplog):
    """sklearn splitters expose no get_params(), so every one of them used to hit the 'does not accept n_splits' warning."""
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        out_cv, val_cv, es_rounds = _resolve(cv, val_nsplits=10)
    assert out_cv is cv
    assert type(val_cv) is type(cv) and val_cv is not cv
    assert val_cv.n_splits == 10 and cv.n_splits == 4
    for attr in ("shuffle", "random_state", "gap"):
        if hasattr(cv, attr):
            assert getattr(val_cv, attr) == getattr(cv, attr)
    assert es_rounds == 20
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], [r.getMessage() for r in caplog.records]


def test_leave_one_out_still_warns_that_val_nsplits_cannot_apply(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _cv, val_cv, _ = _resolve(LeaveOneOut(), val_nsplits=5)
    assert isinstance(val_cv, LeaveOneOut)
    assert any("has no n_splits" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("spec", ["KFold", KFold, "5"], ids=["name", "class", "numeric_string"])
def test_cv_given_as_name_class_or_numeric_string_resolves_to_a_splitter(spec, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        cv, val_cv, _ = _resolve(spec, estimator=Ridge(), val_nsplits=4)
    assert isinstance(cv, KFold)
    assert cv.n_splits == (5 if spec == "5" else 3)
    assert isinstance(val_cv, KFold) and val_cv.n_splits == 4
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_unknown_splitter_name_raises():
    with pytest.raises(ValueError, match="not a known CV splitter name"):
        _resolve("NoSuchSplit")


def test_unsorted_timestamps_hint_gives_timestamp_ordered_folds():
    """A non-monotonic timestamps hint used to fall through to (Stratified)KFold, discarding the temporal signal."""
    n = 60
    ts = np.random.default_rng(0).permutation(n)
    cv, val_cv, _ = _resolve(3, n=n, fit_params={"timestamps": ts}, val_nsplits=4)
    assert isinstance(cv, TimestampOrderedSplit)
    for tr, te in cv.split(np.zeros((n, 1))):
        assert ts[tr].max() < ts[te].min()
    assert isinstance(val_cv, TimestampOrderedSplit) and val_cv.timestamps is None and val_cv.n_splits == 4


def test_unsorted_timestamps_hint_with_groups_orders_groups_by_time():
    n = 60
    rng = np.random.default_rng(1)
    ts = rng.permutation(n)
    groups = ts // 5
    cv, _val, _ = _resolve(3, n=n, fit_params={"timestamps": ts}, groups=groups, val_nsplits=0)
    assert isinstance(cv, TimestampOrderedSplit)
    for tr, te in cv.split(np.zeros((n, 1)), groups=groups):
        assert not set(groups[tr]) & set(groups[te])
        assert ts[tr].max() < ts[te].min()


def test_timestamp_ordered_split_matches_time_series_split_on_sorted_rows():
    n = 50
    ref = [(tr.tolist(), te.tolist()) for tr, te in TimeSeriesSplit(n_splits=4).split(np.zeros(n))]
    got = [(tr.tolist(), te.tolist()) for tr, te in TimestampOrderedSplit(n_splits=4, timestamps=np.arange(n)).split(np.zeros(n))]
    assert got == ref


def test_timestamp_ordered_split_on_shuffled_datetimes_returns_chronological_folds():
    n = 80
    order = np.random.default_rng(2).permutation(n)
    ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="h").to_numpy()[order])
    splitter = TimestampOrderedSplit(n_splits=3, timestamps=ts)
    for tr, te in splitter.split(np.zeros((n, 1))):
        assert ts.iloc[tr].max() < ts.iloc[te].min()
        assert ts.iloc[tr].is_monotonic_increasing and ts.iloc[te].is_monotonic_increasing


def test_timestamp_ordered_split_length_mismatch_warns_and_uses_row_order(caplog):
    splitter = TimestampOrderedSplit(n_splits=3, timestamps=np.arange(10)[::-1])
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        folds = list(splitter.split(np.zeros(20)))
    assert folds[0][0].tolist() == list(range(folds[0][0].size))
    assert any("holds 10 timestamps" in r.getMessage() for r in caplog.records)


def test_timestamp_ordered_split_shares_timestamps_on_deepcopy_and_drops_them_from_pickle(caplog):
    import copy
    import pickle

    ts = np.arange(30)[::-1].copy()
    splitter = TimestampOrderedSplit(n_splits=3, timestamps=ts)
    assert copy.deepcopy(splitter).timestamps is ts
    restored = pickle.loads(pickle.dumps(splitter))
    assert restored.timestamps is None
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        list(restored.split(np.zeros(30)))
    assert any("were not pickled" in r.getMessage() for r in caplog.records)
