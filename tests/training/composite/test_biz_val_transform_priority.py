"""biz_value test for ``training.composite.transform_priority.recommend_transform_candidates``.

The win: ``CompositeTargetDiscovery``'s transform search is exhaustive/order-independent (every configured
transform is CV/MI-scored and the best wins regardless of trial order), so this helper's value is COMPUTE
reduction, not outcome improvement -- for a strictly-positive, scale-like (volatility-style) target/base
pair, the additive ``diff``/``linear_residual`` transforms are known (Optiver 1st place) not to help and are
a wasted CV-scoring pass. This test validates BOTH halves of the claim: (1) the premise -- ratio genuinely
beats diff by CV R^2 on a volatility-style synthetic, confirming the pruning doesn't discard a viable
transform, and (2) the mechanism -- the helper actually drops the additive candidates for such pairs,
reducing the number of transforms a caller needs to evaluate.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from mlframe.training.composite.transform_priority import recommend_transform_candidates
from mlframe.training.composite.transforms.simple import _diff_forward, _ratio_fit, _ratio_forward


def _make_volatility_style_dataset(n: int, seed: int):
    # A strictly-positive, multiplicative target: true_vol scales with base by a feature-dependent factor,
    # not an additive offset -- exactly Optiver's "target / realized_volatility gave a real improvement,
    # plain additive residual did NOT work" regime.
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.5, 2.0, size=(n, 3))
    base = rng.uniform(1.0, 5.0, size=n)
    multiplier = 1.0 + 0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2]
    y = base * multiplier * np.exp(rng.normal(scale=0.05, size=n))  # strictly positive, multiplicative
    return X, y, base


def test_biz_val_ratio_transform_beats_diff_on_volatility_style_target():
    X, y, base = _make_volatility_style_dataset(n=500, seed=0)

    ratio_params = _ratio_fit(y, base)
    T_ratio = _ratio_forward(y, base, ratio_params)
    auc_ratio = float(np.mean(cross_val_score(Ridge(alpha=1.0), X, T_ratio, cv=5, scoring="r2")))

    T_diff = _diff_forward(y, base, {})
    auc_diff = float(np.mean(cross_val_score(Ridge(alpha=1.0), X, T_diff, cv=5, scoring="r2")))

    assert auc_ratio > auc_diff + 0.1, f"expected ratio to materially beat diff on a volatility-style multiplicative target, got ratio_r2={auc_ratio:.4f} vs diff_r2={auc_diff:.4f}"


def test_recommend_transform_candidates_drops_additive_for_positive_scale_pair():
    _, y, base = _make_volatility_style_dataset(n=200, seed=1)
    candidates = ["diff", "linear_residual", "ratio", "logratio"]
    recommended = recommend_transform_candidates(y, base, candidates)
    assert "diff" not in recommended
    assert "linear_residual" not in recommended
    assert "ratio" in recommended and "logratio" in recommended
    assert len(recommended) < len(candidates)


def test_recommend_transform_candidates_keeps_additive_when_not_strictly_positive():
    rng = np.random.default_rng(2)
    y = rng.normal(size=200)  # can be negative
    base = rng.uniform(1.0, 5.0, size=200)
    candidates = ["diff", "linear_residual", "ratio", "logratio"]
    recommended = recommend_transform_candidates(y, base, candidates)
    assert recommended == candidates


def test_recommend_transform_candidates_never_drops_below_one_when_no_multiplicative_alternative():
    y = np.abs(np.random.default_rng(3).normal(size=100)) + 0.1
    base = np.abs(np.random.default_rng(4).normal(size=100)) + 0.1
    candidates = ["diff", "linear_residual"]
    recommended = recommend_transform_candidates(y, base, candidates)
    assert recommended == candidates
