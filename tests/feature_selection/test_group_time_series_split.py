"""GroupTimeSeriesSplit: entity isolation AND temporal fold order, plus RFECV auto-routing on groups+temporal.

Fixes the wave-6 P2 leak: with groups + a temporal signal RFECV previously chose GroupKFold, which isolates
entities but can train on a future group and test on a past one. GroupTimeSeriesSplit forward-chains at the group
level so every test group is strictly later than every train group and no group straddles the boundary.
"""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.wrappers.rfecv._group_time_series_split import GroupTimeSeriesSplit


def _time_ordered_groups(n_groups=8, per=10):
    # group g occupies a contiguous, time-ordered block of rows (row order = time)
    groups = np.repeat(np.arange(n_groups), per)
    return groups


def test_no_group_straddles_and_test_is_future():
    groups = _time_ordered_groups(8, 10)
    order = {g: i for i, g in enumerate(pd.unique(groups))}
    splitter = GroupTimeSeriesSplit(n_splits=4)
    folds = list(splitter.split(np.zeros((len(groups), 2)), groups=groups))
    assert len(folds) == 4
    for tr, te in folds:
        tr_groups = set(groups[tr])
        te_groups = set(groups[te])
        # entity isolation: no group in both sides
        assert tr_groups.isdisjoint(te_groups)
        # temporal: every test group later than every train group
        assert min(order[g] for g in te_groups) > max(order[g] for g in tr_groups)


def test_reproduces_time_series_split_when_each_row_is_its_own_group():
    from sklearn.model_selection import TimeSeriesSplit

    n = 40
    groups = np.arange(n)  # each row a distinct group, in time order
    gts = list(GroupTimeSeriesSplit(n_splits=5).split(np.zeros((n, 1)), groups=groups))
    tss = list(TimeSeriesSplit(n_splits=5).split(np.zeros((n, 1))))
    assert len(gts) == len(tss)
    for (a_tr, a_te), (b_tr, b_te) in zip(gts, tss):
        assert np.array_equal(np.sort(a_tr), np.sort(b_tr))
        assert np.array_equal(np.sort(a_te), np.sort(b_te))


def test_gap_embargoes_groups_between_train_and_test():
    groups = _time_ordered_groups(10, 5)
    order = {g: i for i, g in enumerate(pd.unique(groups))}
    for tr, te in GroupTimeSeriesSplit(n_splits=3, gap=1).split(np.zeros((len(groups), 1)), groups=groups):
        gap = min(order[g] for g in set(groups[te])) - max(order[g] for g in set(groups[tr]))
        assert gap >= 2, "gap=1 must leave at least one embargoed group between train and test"


def test_max_train_groups_is_a_rolling_window():
    groups = _time_ordered_groups(12, 4)
    for tr, _te in GroupTimeSeriesSplit(n_splits=3, max_train_groups=2).split(np.zeros((len(groups), 1)), groups=groups):
        assert len(set(groups[tr])) <= 2


def test_too_few_groups_raises():
    groups = np.array([0, 0, 1, 1])  # 2 groups, n_splits=4 impossible
    with pytest.raises(ValueError, match="too few"):
        list(GroupTimeSeriesSplit(n_splits=4).split(np.zeros((4, 1)), groups=groups))


def test_rfecv_auto_routes_to_group_time_series_on_groups_plus_temporal():
    from sklearn.ensemble import RandomForestRegressor
    from mlframe.feature_selection.wrappers.rfecv._cv_setup import _resolve_cv_and_val_cv

    n = 60
    X = pd.DataFrame({"f0": np.arange(n, dtype=float), "f1": np.arange(n, dtype=float)})
    groups = np.repeat(np.arange(12), 5)
    ts = np.arange(n)  # monotonic timestamps hint
    cv, _val, _es = _resolve_cv_and_val_cv(
        cv=3,
        estimator=RandomForestRegressor(n_estimators=3),
        X=X,
        y=np.arange(n, dtype=float),
        groups=groups,
        cv_shuffle=False,
        random_state=0,
        verbose=False,
        fit_params={"timestamps": ts},
        early_stopping_val_nsplits=0,
        early_stopping_rounds=None,
        _polars_time_series_hint=False,
    )
    assert isinstance(cv, GroupTimeSeriesSplit), f"expected GroupTimeSeriesSplit, got {type(cv).__name__}"
