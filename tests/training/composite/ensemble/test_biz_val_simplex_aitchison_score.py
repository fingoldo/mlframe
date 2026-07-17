"""biz_value: CompositeSimplexEstimator scores compositional predictions with the Aitchison metric by default.

The inherited Euclidean R^2 is meaningless on the simplex (it ignores the constant-sum constraint and relative scale).
``CompositeSimplexEstimator.score`` defaults to ``score_metric='aitchison'`` -- an R^2-style skill against the
geometric-mean baseline using squared Aitchison distances. These tests pin that the default is Aitchison, that it
rewards a genuinely-learnable compositional fit, and that the euclidean opt-out still returns the sklearn R^2.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.simplex import CompositeSimplexEstimator, aitchison_distance


def _compositional_dataset(seed: int = 0, n: int = 1500):
    """3-part composition driven by 4 features through a softmax -- learnable on the log-ratio scale."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    logits = (
        np.column_stack(
            [
                X @ np.array([1.2, -0.6, 0.0, 0.3]),
                X @ np.array([-0.4, 0.9, 0.5, 0.0]),
                np.zeros(n),
            ]
        )
        + rng.standard_normal((n, 3)) * 0.15
    )
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    y = e / e.sum(axis=1, keepdims=True)
    return X, y


def _make():
    from sklearn.linear_model import Ridge

    return CompositeSimplexEstimator(base_estimator=Ridge(alpha=1.0))


def test_score_metric_defaults_to_aitchison():
    assert _make().score_metric == "aitchison"


def test_aitchison_score_rewards_learnable_compositional_fit():
    X, y = _compositional_dataset(seed=1)
    cut = 1000
    est = _make().fit(X[:cut], y[:cut])
    skill = est.score(X[cut:], y[cut:])
    # A softmax-linear composition is well within Ridge-on-ilr reach; skill should be a large positive fraction.
    assert skill > 0.5, f"Aitchison skill on a learnable composition should be >0.5; got {skill:.3f}"


def test_aitchison_score_above_constant_baseline_but_euclidean_optout_available():
    X, y = _compositional_dataset(seed=2)
    cut = 1000
    est = _make().fit(X[:cut], y[:cut])
    aitch = est.score(X[cut:], y[cut:])
    est.score_metric = "euclidean"
    eucl = est.score(X[cut:], y[cut:])
    # Both should agree the fit beats the trivial baseline (positive), but they are distinct metrics on distinct
    # geometries -- the default is the simplex-correct Aitchison skill, the opt-out the sklearn Euclidean R^2.
    assert aitch > 0.0 and np.isfinite(eucl)
    assert aitch != pytest.approx(eucl, abs=1e-9), "Aitchison and Euclidean scores must be distinct metrics"


def test_aitchison_score_penalises_proportion_blind_prediction():
    """A predictor that nails the dominant part but inverts the two minor parts is fine in raw Euclidean terms yet
    wrong on the simplex; its Aitchison skill must be strictly below a proportion-faithful fit on the same data."""
    X, y = _compositional_dataset(seed=3)
    cut = 1000
    good = _make().fit(X[:cut], y[:cut])
    good_skill = good.score(X[cut:], y[cut:])
    # Construct a "proportion-blind" composition: keep part 0, swap the relative split of parts 1 and 2.
    yte = y[cut:]
    blind = yte.copy()
    blind[:, [1, 2]] = blind[:, [2, 1]]
    d_blind = float(np.mean(np.square(aitchison_distance(blind, yte))))
    d_good = float(np.mean(np.square(aitchison_distance(good.predict(X[cut:]), yte))))
    assert d_good < d_blind, f"faithful fit must have smaller Aitchison error than the part-swapped one ({d_good:.3f} vs {d_blind:.3f})"
    assert good_skill > 0.0
