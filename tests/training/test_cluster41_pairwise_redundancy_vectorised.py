"""Cluster 41 regression: the vectorised pairwise-redundancy scan returns what the nested loop did (audit 2026-11-20, 41-F41)."""

import math

import numpy as np

from mlframe.training.targets._target_distribution_analyzer_features import _pairwise_redundant_features


def _loop_reference(X, names, threshold):
    """The pre-fix nested loop."""
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(X, rowvar=False)
    out = []
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            c = float(C[i, j])
            if math.isfinite(c) and abs(c) >= threshold:
                out.append((names[i], names[j], abs(c)))
    out.sort(key=lambda t: -t[2])
    return out


def test_matches_the_nested_loop_with_constant_and_correlated_columns():
    """Correlated, anti-correlated, constant (NaN correlation) and noise columns: same pairs, same order, same values (41-F41)."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(400, 1))
    X = np.hstack([base, base + 0.05 * rng.normal(size=(400, 1)), -base, np.ones((400, 1)), rng.normal(size=(400, 6))])
    names = [f"f{i}" for i in range(X.shape[1])]
    assert _pairwise_redundant_features(X, names, 0.9) == _loop_reference(X, names, 0.9)
    assert len(_pairwise_redundant_features(X, names, 0.9)) >= 3


def test_fewer_than_two_columns_is_empty():
    """A single column has no pairs (41-F41)."""
    assert _pairwise_redundant_features(np.ones((10, 1)), ["a"]) == []
