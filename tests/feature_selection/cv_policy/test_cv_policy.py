"""Decision table and builders of the shared feature-selector split policy."""
import copy
import logging
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

from mlframe.feature_selection.cv_policy import (
    BoundGroupKFold, CVPolicy, TimestampOrderedSplit, apply_cv_param, build_cv_splitter, cv_is_temporal, decide_cv_policy, get_cv_policy,
    holdout_indices, partition_folds, wire_selector_policy,
)

N = 120


def _split(**kw):
    base = dict(cv_strategy="random", test_size=0.2, val_size=0.2, shuffle_test=False, shuffle_val=False,
                test_sequential_fraction=None, val_sequential_fraction=None, use_groups=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _hp(**kw):
    ns = SimpleNamespace(**kw)
    ns.model_fields_set = set(kw)
    return ns


@pytest.fixture
def shuffled_ts():
    return np.random.default_rng(0).permutation(N)


@pytest.mark.parametrize(
    "has_ts,split,hp,override,expected",
    [
        (False, _split(), _hp(), None, False),
        (True, _split(), _hp(), None, True),
        (True, _split(shuffle_test=True, shuffle_val=True), _hp(), None, False),
        (True, _split(shuffle_test=True, shuffle_val=True, cv_strategy="timeseries"), _hp(), None, True),
        (True, _split(shuffle_test=True, shuffle_val=True), _hp(has_time=True), None, True),
        (True, _split(), _hp(has_time=False), None, False),
        (False, _split(), _hp(has_time=True), None, True),
        (True, _split(), _hp(), "caller set cv", False),
        (True, None, _hp(), None, True),
    ],
)
def test_cv_is_temporal_decision_table(has_ts, split, hp, override, expected):
    temporal, reason = cv_is_temporal(np.arange(N) if has_ts else None, split, hp, override)
    assert temporal is expected and reason


def test_decide_cv_policy_kinds(shuffled_ts):
    groups = np.repeat(np.arange(12), 10)
    temporal = decide_cv_policy(timestamps=shuffled_ts, train_idx=None, groups=groups, split_config=_split(), hyperparams_config=_hp())
    assert temporal.kind == "temporal" and temporal.groups is groups
    grouped = decide_cv_policy(timestamps=None, train_idx=None, groups=groups, split_config=_split(use_groups=True), hyperparams_config=_hp())
    assert grouped.kind == "grouped"
    ignoring = decide_cv_policy(timestamps=None, train_idx=None, groups=groups, split_config=_split(use_groups=False), hyperparams_config=_hp())
    assert ignoring.kind == "iid"
    assert decide_cv_policy(timestamps=None, train_idx=None).kind == "iid"


def test_decide_cv_policy_slices_arrays_to_train_rows(shuffled_ts):
    train_idx = np.arange(0, N, 2)
    policy = decide_cv_policy(timestamps=pd.Series(shuffled_ts), train_idx=train_idx, split_config=_split(), hyperparams_config=_hp())
    assert len(policy.timestamps) == train_idx.size
    assert np.array_equal(np.asarray(policy.timestamps), shuffled_ts[train_idx])


def test_temporal_splitter_folds_are_chronological_on_unsorted_rows(shuffled_ts):
    cv = build_cv_splitter(CVPolicy("temporal", "t", shuffled_ts), 4)
    assert isinstance(cv, TimestampOrderedSplit) and cv.deterministic_folds
    for tr, te in cv.split(np.zeros((N, 1))):
        assert shuffled_ts[tr].max() < shuffled_ts[te].min()


def test_temporal_splitter_with_bound_groups_isolates_groups_in_time_order(shuffled_ts):
    groups = np.repeat(np.arange(12), 10)
    ts = np.arange(N).astype(float)
    cv = build_cv_splitter(CVPolicy("temporal", "t", ts, groups), 3)
    folds = list(cv.split(np.zeros((N, 1))))
    assert len(folds) == 3
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te])
        assert ts[tr].min() <= ts[te].min()


def test_grouped_splitter_isolates_groups_without_passing_groups_to_split():
    groups = np.repeat(np.arange(10), 12)
    cv = build_cv_splitter(CVPolicy("grouped", "g", None, groups), 4)
    assert isinstance(cv, BoundGroupKFold)
    for tr, te in cv.split(np.zeros((N, 1))):
        assert not set(groups[tr]) & set(groups[te])


def test_iid_splitter_is_stratified_for_classification_else_plain():
    assert isinstance(build_cv_splitter(CVPolicy("iid", "x"), 3, classification=True), StratifiedKFold)
    assert type(build_cv_splitter(None, 3)) is KFold


def test_holdout_indices_temporal_takes_newest_rows(shuffled_ts):
    search, hold = holdout_indices(CVPolicy("temporal", "t", shuffled_ts), N, 0.25)
    assert hold.size == 30 and search.size == 90
    assert shuffled_ts[search].max() < shuffled_ts[hold].min()
    assert np.all(np.diff(search) > 0) and np.all(np.diff(hold) > 0)


def test_holdout_indices_grouped_isolates_groups():
    groups = np.repeat(np.arange(12), 10)
    search, hold = holdout_indices(CVPolicy("grouped", "g", None, groups), N, 0.25, random_state=1)
    assert not set(groups[search]) & set(groups[hold])


def test_holdout_indices_none_for_iid_and_for_row_count_mismatch(caplog):
    assert holdout_indices(None, N, 0.3) is None
    assert holdout_indices(CVPolicy("iid", "x"), N, 0.3) is None
    with caplog.at_level(logging.WARNING):
        assert holdout_indices(CVPolicy("temporal", "t", np.arange(10)), N, 0.3) is None
    assert any("i.i.d. split" in r.getMessage() for r in caplog.records)


def test_partition_folds_temporal_blocks_partition_all_rows(shuffled_ts):
    folds = partition_folds(CVPolicy("temporal", "t", shuffled_ts), N, 4)
    assert folds is not None
    tests = np.concatenate([te for _, te in folds])
    assert np.array_equal(np.sort(tests), np.arange(N))
    ends = [shuffled_ts[te].max() for _, te in folds]
    starts = [shuffled_ts[te].min() for _, te in folds]
    assert all(e < s for e, s in zip(ends[:-1], starts[1:])), "test blocks must be contiguous, ordered in time"
    for tr, te in folds:
        assert not set(tr) & set(te)


def test_partition_folds_grouped_and_iid_fallback():
    groups = np.repeat(np.arange(12), 10)
    folds = partition_folds(CVPolicy("grouped", "g", None, groups), N, 3)
    assert folds and all(not set(groups[tr]) & set(groups[te]) for tr, te in folds)
    assert partition_folds(CVPolicy("iid", "x"), N, 3) is None


def test_apply_cv_param_only_replaces_suite_chosen_cv(shuffled_ts):
    policy = CVPolicy("temporal", "t", shuffled_ts)
    for original in (None, 4):
        sel = SimpleNamespace(cv=original)
        assert apply_cv_param(sel, policy) and isinstance(sel.cv, TimestampOrderedSplit)
        assert sel.cv.n_splits == (5 if original is None else 4)
    sel = SimpleNamespace(cv=TimeSeriesSplit(n_splits=6))
    assert apply_cv_param(sel, policy) and sel.cv.n_splits == 6 and isinstance(sel.cv, TimestampOrderedSplit)
    own = GroupKFold(n_splits=3)
    sel = SimpleNamespace(cv=own)
    assert not apply_cv_param(sel, policy) and sel.cv is own
    sel = SimpleNamespace(cv=None)
    assert not apply_cv_param(sel, CVPolicy("iid", "x")) and sel.cv is None


def test_wire_selector_policy_stamps_and_skips_iid(shuffled_ts):
    sel = SimpleNamespace(cv=None)
    assert wire_selector_policy(sel, CVPolicy("temporal", "t", shuffled_ts))
    assert get_cv_policy(sel).kind == "temporal" and isinstance(sel.cv, TimestampOrderedSplit)
    other = SimpleNamespace(cv=None)
    assert not wire_selector_policy(other, CVPolicy("iid", "x")) and get_cv_policy(other) is None


def test_policy_shares_arrays_on_deepcopy_and_drops_them_from_pickle(shuffled_ts):
    policy = CVPolicy("temporal", "t", shuffled_ts)
    assert copy.deepcopy(policy).timestamps is shuffled_ts
    restored = pickle.loads(pickle.dumps(policy))
    assert restored.timestamps is None and restored.kind == "temporal"
    assert holdout_indices(restored, N, 0.25) is None


def test_subset_restricts_rows_and_degrades_when_arrays_are_short(shuffled_ts):
    idx = np.arange(0, N, 3)
    sub = CVPolicy("temporal", "t", shuffled_ts).subset(idx)
    assert np.array_equal(sub.timestamps, shuffled_ts[idx])
    assert CVPolicy("temporal", "t", shuffled_ts[:10]).subset(idx).kind == "iid"
