"""Regression (wave6 P2): overflow/domain/empty guards surfaced by the numerical-stability audit.

- batch_pair/triple_mi_prange: the joint-cardinality OOM cap was tested AFTER forming nb_a*nb_b(*nb_c) in int64,
  which wraps silently on billion-scale cardinalities and defeats the cap. The cap is now tested via division
  BEFORE multiplying (pathological pairs return MI=0.0, no allocation).
- _bayesian_blocks_bin_edges: math.log(73.53 * p0 * ...) raised a domain error on the public p0<=0; now validated.
- merge_vars: freqs / len(factors_data) divided by zero on an empty frame; now returns an empty freq vector.
- RFECV _resolve_cv_and_val_cv: groups + a temporal signal silently chose GroupKFold (no time ordering); now warns.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.info_theory._batch_kernels import (
    batch_pair_mi_prange,
    batch_triple_mi_prange,
    MAX_JOINT_CARDINALITY,
)
from mlframe.feature_selection.filters.info_theory._class_encoding import merge_vars


def _tiny_batch_inputs(nbins_val):
    n = 40
    rng = np.random.default_rng(0)
    factors = rng.integers(0, 2, size=(n, 3)).astype(np.int32)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    freqs_y = np.bincount(y).astype(np.float64) / n
    nbins = np.array([nbins_val, nbins_val, nbins_val], dtype=np.int64)
    return factors, nbins, y, freqs_y


def test_batch_pair_mi_huge_cardinality_returns_zero_not_oom():
    factors, nbins, y, freqs_y = _tiny_batch_inputs(nbins_val=1_000_000)  # 1e6 * 1e6 = 1e12 > cap
    a = np.array([0], dtype=np.int64)
    b = np.array([1], dtype=np.int64)
    out = batch_pair_mi_prange(factors, a, b, nbins, y, freqs_y)
    assert out[0] == 0.0, "pathological-cardinality pair must be skipped to MI=0.0 without allocating"


def test_batch_triple_mi_huge_cardinality_returns_zero():
    factors, nbins, y, freqs_y = _tiny_batch_inputs(nbins_val=1_000_000)
    a = np.array([0], dtype=np.int64)
    b = np.array([1], dtype=np.int64)
    c = np.array([2], dtype=np.int64)
    out = batch_triple_mi_prange(factors, a, b, c, nbins, y, freqs_y)
    assert out[0] == 0.0


def test_batch_pair_mi_normal_cardinality_still_computes():
    factors, nbins, y, freqs_y = _tiny_batch_inputs(nbins_val=2)  # in-range
    out = batch_pair_mi_prange(factors, np.array([0]), np.array([1]), nbins, y, freqs_y)
    assert np.isfinite(out[0]) and out[0] >= 0.0


def test_bayesian_blocks_rejects_nonprobability_p0():
    from mlframe.feature_selection.filters.discretization._discretization_edges import _bayesian_blocks_bin_edges

    a = np.random.default_rng(1).normal(size=200)
    for bad in (0.0, -0.1, 1.0, 2.0):
        with pytest.raises(ValueError, match="p0"):
            _bayesian_blocks_bin_edges(a, p0=bad)
    # a valid p0 still works
    edges = _bayesian_blocks_bin_edges(a, p0=0.05)
    assert edges.ndim == 1 and edges.size >= 2


def test_merge_vars_empty_frame_no_divide_by_zero():
    factors = np.empty((0, 2), dtype=np.int32)
    fc, freqs, ncl = merge_vars(
        factors, np.array([0, 1]), np.array([False, False]), np.array([2, 2], dtype=np.int64)
    )
    assert len(fc) == 0
    assert freqs.size == 0 and np.isfinite(freqs).all()


def test_rfecv_routes_groups_plus_temporal_to_group_time_series():
    # groups + a temporal signal now ROUTES to GroupTimeSeriesSplit (entity isolation AND time-ordered folds)
    # instead of the old silent GroupKFold. Full guarantees are exercised in test_group_time_series_split.py.
    from sklearn.ensemble import RandomForestRegressor
    from mlframe.feature_selection.wrappers.rfecv._cv_setup import _resolve_cv_and_val_cv
    from mlframe.feature_selection.wrappers.rfecv._group_time_series_split import GroupTimeSeriesSplit

    n = 30
    X = pd.DataFrame({"f0": np.arange(n, dtype=float), "f1": np.arange(n, dtype=float)})
    groups = np.repeat(np.arange(6), 5)  # 6 groups >= n_splits+1
    ts = np.arange(n)  # monotonic timestamps hint
    cv, _val, _es = _resolve_cv_and_val_cv(
        cv=3, estimator=RandomForestRegressor(n_estimators=3), X=X, y=np.arange(n, dtype=float),
        groups=groups, cv_shuffle=False, random_state=0, verbose=False,
        fit_params={"timestamps": ts}, early_stopping_val_nsplits=0, early_stopping_rounds=None,
        _polars_time_series_hint=False,
    )
    assert isinstance(cv, GroupTimeSeriesSplit), f"expected GroupTimeSeriesSplit, got {type(cv).__name__}"
