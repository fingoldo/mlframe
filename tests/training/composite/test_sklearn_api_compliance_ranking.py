"""sklearn-contract regression tests for CompositeRankEstimator (SK3).

- ``clone(est)`` round-trips (no learned state leaks into get_params).
- ``predict`` before ``fit`` raises NotFittedError (check_is_fitted).
- learned attributes end with a trailing underscore.
- ``group`` is a keyword with a None default so ``clone(est).fit(X, y)`` is
  reachable (and raises a clear error when group is omitted), while behavior
  with an explicit group is unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

from mlframe.training.composite.ranking import CompositeRankEstimator


def _toy(n_groups: int = 6, per: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    groups = []
    ys = []
    for g in range(n_groups):
        base = rng.standard_normal(per)
        feat = rng.standard_normal(per)
        y = base + 0.5 * feat + 0.1 * rng.standard_normal(per)
        X = np.column_stack([base, feat])
        rows.append(X)
        groups.extend([g] * per)
        ys.append(y)
    return np.vstack(rows), np.concatenate(ys), np.asarray(groups)


def test_sk3_clone_round_trips():
    est = CompositeRankEstimator(base_column=0)
    cloned = clone(est)
    assert cloned.get_params() == est.get_params()


def test_sk3_predict_before_fit_raises_not_fitted():
    est = CompositeRankEstimator(base_column=0)
    X, _, _ = _toy()
    with pytest.raises(NotFittedError):
        est.predict(X)


def test_sk3_fit_without_group_raises_clear_error():
    # group is now a keyword default None so clone().fit(X, y) is callable; it must raise a clear error rather than a TypeError on missing-positional.
    est = CompositeRankEstimator(base_column=0)
    X, y, _ = _toy()
    with pytest.raises(ValueError, match="group"):
        est.fit(X, y)


@pytest.mark.parametrize(
    "inner",
    [None, LogisticRegression(max_iter=200)],
    ids=["default_inner", "pairwise_logistic"],
)
def test_sk3_learned_attrs_have_trailing_underscore(inner):
    est = CompositeRankEstimator(base_column=0, base_estimator=inner)
    X, y, group = _toy()
    est.fit(X, y, group=group)

    for attr in ("inner_", "kind_", "fitted_residual_mode_", "n_features_in_"):
        assert hasattr(est, attr), attr
    # No legacy non-underscore learned attrs remain.
    for legacy in ("_inner", "_kind", "_pairwise_w", "_pairwise_b", "_fitted_residual_mode"):
        assert not hasattr(est, legacy), legacy

    # predict works after fit and returns one score per row.
    scores = est.predict(X, group=group)
    assert scores.shape == (X.shape[0],)
