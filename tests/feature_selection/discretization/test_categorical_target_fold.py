"""``fold_column_by_target``: levels merge when the target cannot tell them apart and stay apart when it can, however few rows they have."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.discretization._categorical_target_fold import fold_column_by_target, target_classes
from mlframe.feature_selection.filters.discretization import cap_categorical_cardinality


def _frame(seed=0, n_big=1380, thin_levels=40, fraud_levels=6, base_rate=0.03):
    """Five big levels at the base rate, ``thin_levels`` levels of three rows each, ``fraud_levels`` of them all positive."""
    rng = np.random.default_rng(seed)
    big = rng.integers(0, 5, n_big)
    thin = np.repeat(np.arange(5, 5 + thin_levels), 3)
    y_big = (rng.random(n_big) < base_rate).astype(int)
    y_thin = np.concatenate([np.ones(3, dtype=int) if k < fraud_levels else np.zeros(3, dtype=int) for k in range(thin_levels)])
    return np.concatenate([big, thin]).astype(np.float64), np.concatenate([y_big, y_thin])


def test_thin_levels_of_the_rare_class_stay_apart_from_the_base_levels():
    """Six three-row levels that are all of a 3 % class share one code, and no base level shares it: min_support would have folded all of them into 'other'."""
    lvl, y = _frame()
    out = fold_column_by_target(lvl, target_classes(y), 127)
    fraud_codes = set(out[(lvl >= 5) & (lvl < 11)].tolist())
    base_codes = set(out[lvl < 5].tolist())
    assert len(fraud_codes) == 1 and not (fraud_codes & base_codes)
    folded = cap_categorical_cardinality(lvl.reshape(-1, 1), 127, min_support=5).ravel()
    assert set(folded[(lvl >= 5) & (lvl < 11)].tolist()) == set(folded[(lvl >= 11)].tolist())  # ... the same bucket as the thin levels without signal


def test_levels_with_one_class_mix_merge_and_codes_are_dense():
    """Ten levels drawn from one distribution collapse to (almost) one group; the codes run 0..k-1 in order of size; a missing code stays -1."""
    rng = np.random.default_rng(3)
    lvl = rng.integers(0, 10, 3000).astype(np.float64)
    y = (rng.random(3000) < 0.4).astype(int)
    lvl[:5] = -1
    out = fold_column_by_target(lvl, target_classes(y), 127)
    assert (out[:5] == -1).all()
    codes = np.unique(out[out >= 0])
    assert list(codes) == list(range(codes.size)) and codes.size <= 3
    sizes = np.bincount(out[out >= 0].astype(int))
    assert list(sizes) == sorted(sizes, reverse=True)


def test_a_constant_target_or_no_classes_leaves_the_column_alone():
    """With fewer than two classes there is nothing to split on: the column comes back unchanged (and ``target_classes`` reports a constant target as ``None``)."""
    lvl = np.arange(20, dtype=np.float64) % 7
    assert target_classes(np.ones(20)) is None
    assert np.array_equal(fold_column_by_target(lvl, np.zeros(20, dtype=np.int64), 127), lvl)


def test_the_cap_still_binds_when_the_target_splits_everything():
    """Every level of a 60-level column is a pure class-0 / class-1 mix of its own; with a cap of 10 the merge continues past the BIC threshold until at most 9 groups remain."""
    rng = np.random.default_rng(4)
    levels = np.repeat(np.arange(60), 40).astype(np.float64)
    rate = rng.random(60)
    y = np.concatenate([(rng.random(40) < r).astype(int) for r in rate])
    out = fold_column_by_target(levels, target_classes(y), 10)
    assert np.unique(out).size <= 9


def test_continuous_targets_get_quantile_classes():
    """A target with more than 32 distinct values becomes 8 quantile classes; missing values map to -1."""
    y = np.random.default_rng(5).standard_normal(2000)
    y[:3] = np.nan
    codes = target_classes(y)
    assert (codes[:3] == -1).all() and codes[3:].max() <= 7 and np.unique(codes[3:]).size >= 7
