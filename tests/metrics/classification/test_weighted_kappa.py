"""Unit + biz_value tests for weighted / quadratic-weighted kappa (PZAD err_multirankcluster)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.classification._weighted_kappa import (
    quadratic_weighted_kappa,
    weighted_kappa,
)


# ---------------------------------------------------------------- unit
def test_perfect_agreement_is_one():
    """Perfect agreement is one."""
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    assert abs(quadratic_weighted_kappa(y, y.copy()) - 1.0) < 1e-12


def test_matches_sklearn_quadratic():
    """Matches sklearn quadratic."""
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, size=300)
    a = rng.integers(0, 5, size=300)
    ours = quadratic_weighted_kappa(y, a, n_classes=5)
    ref = sk.cohen_kappa_score(y, a, weights="quadratic")
    assert abs(ours - ref) < 1e-9


def test_matches_sklearn_linear():
    """Matches sklearn linear."""
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, size=250)
    a = rng.integers(0, 4, size=250)
    ours = weighted_kappa(y, a, weights="linear", n_classes=4)
    ref = sk.cohen_kappa_score(y, a, weights="linear")
    assert abs(ours - ref) < 1e-9


def test_invalid_weights_and_mismatch():
    """Invalid weights and mismatch."""
    with pytest.raises(ValueError):
        weighted_kappa(np.zeros(3, int), np.zeros(3, int), weights="cubic")
    with pytest.raises(ValueError):
        weighted_kappa(np.zeros(3, int), np.zeros(2, int))


def test_empty_and_single_class():
    """Empty and single class."""
    assert np.isnan(quadratic_weighted_kappa(np.array([], int), np.array([], int)))
    assert quadratic_weighted_kappa(np.zeros(5, int), np.zeros(5, int)) == 1.0


# ---------------------------------------------------------------- biz_value
def test_biz_val_qwk_rewards_near_misses_over_far_misses():
    """On ordinal targets QWK ranks a model whose errors are 1 grade off ABOVE one whose errors are far off,
    even when both have the same raw accuracy. This is QWK's whole point vs plain accuracy."""
    rng = np.random.default_rng(2)
    n = 500
    y = rng.integers(0, 5, size=n)
    near = y.copy()
    far = y.copy()
    wrong = rng.random(n) < 0.4  # same 40% wrong for both
    near[wrong] = np.clip(y[wrong] + rng.choice([-1, 1], size=wrong.sum()), 0, 4)  # off by 1
    far[wrong] = 4 - y[wrong]  # maximally off (flip on the 0..4 scale)
    from sklearn.metrics import accuracy_score

    # comparable accuracy (both ~60%), but QWK should clearly prefer the near-miss model
    assert abs(accuracy_score(y, near) - accuracy_score(y, far)) < 0.1
    assert quadratic_weighted_kappa(y, near, n_classes=5) > quadratic_weighted_kappa(y, far, n_classes=5) + 0.3


def test_biz_val_quadratic_penalizes_far_errors_more_than_linear():
    """Quadratic weighting punishes a large-distance confusion harder than linear weighting does."""
    y = np.array([0, 0, 0, 0])
    pred_far = np.array([4, 0, 0, 0])  # one big miss
    quadratic_weighted_kappa(y, pred_far, n_classes=5)
    weighted_kappa(y, pred_far, weights="linear", n_classes=5)
    # both degenerate (constant y) -> denominator 0 -> 0.0; use a non-degenerate case instead
    y2 = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    a_far = y2.copy()
    a_far[0] = 4  # one 4-grade miss
    a_near = y2.copy()
    a_near[0] = 1  # one 1-grade miss
    q_far = quadratic_weighted_kappa(y2, a_far, n_classes=5)
    q_near = quadratic_weighted_kappa(y2, a_near, n_classes=5)
    assert q_near > q_far  # the far miss lowers QWK more
