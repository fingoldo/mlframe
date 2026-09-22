"""Regression tests for the training-log audit 2026-09-20, ``SEN-08``.

``per_group_prior`` / ``per_group_mean`` predicted the RAW per-group mean, so a group with one training row emitted
exactly 0.0 or 1.0 for a binary target. Any unbounded proper scoring rule then explodes on a single wrong-and-certain
row: a production run's constant-prior dummy reported ``exploss=0.88`` on val and ``exploss=18.29`` on test -- a 20x
move -- while its log loss moved only 0.57 -> 0.64.

Size-weighted shrinkage toward the global mean removes the 0/1 endpoints, leaves large groups untouched, and makes
the baseline harder to beat, which is what a baseline is for.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.metrics.classification._gains_lift import exploss
from mlframe.training.baselines._dummy_baseline_compute import (
    PER_GROUP_SMOOTHING_MAX_PSEUDOCOUNTS,
    PER_GROUP_SMOOTHING_MIN_PSEUDOCOUNTS,
    _empirical_bayes_pseudocounts,
    _per_group_predict,
)

N_SINGLETON = 3000
GLOBAL_RATE = 0.3


def _singleton_group_split(seed: int = 0):
    """Train where every group holds ONE row, and a val split re-using those groups with independent labels.

    That is the production shape underneath ``high_entity_overlap=1.00``: the bulk of val rows sit in well-populated
    groups, but singleton groups still exist in the tail and it is the tail that decides an unbounded scoring rule.
    """
    rng = np.random.default_rng(seed)
    groups = np.array([f"g{i}" for i in range(N_SINGLETON)])
    y_train = (rng.random(N_SINGLETON) < GLOBAL_RATE).astype(float)
    y_val = (rng.random(N_SINGLETON) < GLOBAL_RATE).astype(float)
    frame = pd.DataFrame({"g": groups})
    return frame, y_train, y_val


def test_singleton_groups_no_longer_emit_zero_or_one():
    frame, y_train, _y_val = _singleton_group_split()
    _train, val_pred, _test, _diag = _per_group_predict(frame, frame, frame, y_train, "g", "binary_classification")
    assert not np.any(val_pred == 0.0), "a one-row group must not predict a certain negative"
    assert not np.any(val_pred == 1.0), "a one-row group must not predict a certain positive"
    # Every group is a singleton, so nothing supports trusting a group over the global prior and the estimated
    # shrinkage goes to the ceiling: predictions collapse onto the observed prior.
    prior = float(y_train.mean())
    reach = 1.0 / (1.0 + PER_GROUP_SMOOTHING_MAX_PSEUDOCOUNTS)
    assert val_pred.min() >= prior - reach - 1e-6
    assert val_pred.max() <= prior + reach + 1e-6


def test_exploss_no_longer_explodes_on_singleton_groups():
    """The metric the production log showed jumping 0.88 -> 18.29 must stay in the same order as log loss."""
    frame, y_train, y_val = _singleton_group_split()
    _train, val_pred, _test, _diag = _per_group_predict(frame, frame, frame, y_train, "g", "binary_classification")
    assert exploss(y_val, val_pred) < 2.0, "an unbounded scoring rule must not be dominated by the clip constant"


def test_strong_group_signal_survives_shrinkage():
    """Shrinkage must not destroy a real group effect, or it replaces a useful baseline with the global mean.

    This is the regime an m=10 constant broke: 600 groups x ~6.7 rows, offsets std 5 against noise std 1. The
    estimated strength drops to its floor here, because the data says the groups are worth trusting.
    """
    rng = np.random.default_rng(0)
    n_groups, n = 600, 4000
    offsets = rng.normal(0, 5, n_groups)
    gid = rng.integers(0, n_groups, n)
    y = offsets[gid] + rng.normal(0, 1, n)
    frame = pd.DataFrame({"g": [f"grp_{g}" for g in gid]})
    _train, val_pred, _test, _diag = _per_group_predict(frame, frame, frame, y, "g", "regression")
    group_rmse = float(np.sqrt(np.mean((y - val_pred) ** 2)))
    assert group_rmse < 0.5 * float(np.std(y)), (group_rmse, np.std(y))


def test_shrinkage_strength_tracks_the_signal_to_noise_ratio():
    """``m`` is the hierarchical-model weight, so it must fall when groups separate and rise when they do not."""
    rng = np.random.default_rng(1)
    sizes = np.full(200, 10.0)
    gid = np.repeat(np.arange(200), 10)
    strong = rng.normal(0, 5, 200)[gid] + rng.normal(0, 1, 2000)
    weak = rng.normal(0, 0.01, 200)[gid] + rng.normal(0, 1, 2000)

    def _m(y):
        means = pd.Series(y).groupby(gid).mean().to_numpy()
        return _empirical_bayes_pseudocounts(y, sizes, means, float(y.mean()))

    assert _m(strong) == pytest.approx(PER_GROUP_SMOOTHING_MIN_PSEUDOCOUNTS, abs=1e-9) or _m(strong) < 1.0
    assert _m(weak) > _m(strong) * 10


def test_polars_and_pandas_paths_agree():
    """The polars fast path and the pandas path must stay numerically interchangeable."""
    frame, y_train, _y_val = _singleton_group_split(seed=3)
    pl_frame = pl.DataFrame({"g": frame["g"].to_numpy()})
    _t1, v1, s1, _d1 = _per_group_predict(frame, frame, frame, y_train, "g", "binary_classification")
    _t2, v2, s2, _d2 = _per_group_predict(pl_frame, pl_frame, pl_frame, y_train, "g", "binary_classification")
    assert np.allclose(v1, v2)
    assert np.allclose(s1, s2)


def test_unseen_groups_still_fall_back_to_the_global_mean():
    frame, y_train, _y_val = _singleton_group_split(seed=5)
    unseen = pd.DataFrame({"g": np.array([f"unseen{i}" for i in range(200)])})
    _train, val_pred, _test, _diag = _per_group_predict(frame, unseen, unseen, y_train, "g", "binary_classification")
    assert np.allclose(val_pred, y_train.mean())


@pytest.mark.parametrize("target_type", ["regression", "binary_classification"])
def test_shrinkage_is_a_convex_blend_of_group_and_global_mean(target_type):
    """Every prediction must lie between its own group mean and the global mean -- shrinkage, never extrapolation."""
    rng = np.random.default_rng(9)
    groups = np.repeat([f"k{i}" for i in range(20)], 7)
    y = rng.normal(100.0, 30.0, groups.size)
    frame = pd.DataFrame({"g": groups})
    _train, val_pred, _test, _diag = _per_group_predict(frame, frame, frame, y, "g", target_type)
    global_mean = float(y.mean())
    raw = pd.Series(y).groupby(groups).mean()
    for grp, pred in zip(groups, val_pred):
        lo, hi = sorted((float(raw[grp]), global_mean))
        assert lo - 1e-9 <= pred <= hi + 1e-9, (grp, pred, lo, hi)
