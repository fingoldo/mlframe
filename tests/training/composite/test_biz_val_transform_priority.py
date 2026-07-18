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
    """Make volatility style dataset."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.5, 2.0, size=(n, 3))
    base = rng.uniform(1.0, 5.0, size=n)
    multiplier = 1.0 + 0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2]
    y = base * multiplier * np.exp(rng.normal(scale=0.05, size=n))  # strictly positive, multiplicative
    return X, y, base


def test_biz_val_ratio_transform_beats_diff_on_volatility_style_target():
    """Biz val ratio transform beats diff on volatility style target."""
    X, y, base = _make_volatility_style_dataset(n=500, seed=0)

    ratio_params = _ratio_fit(y, base)
    T_ratio = _ratio_forward(y, base, ratio_params)
    auc_ratio = float(np.mean(cross_val_score(Ridge(alpha=1.0), X, T_ratio, cv=5, scoring="r2")))

    T_diff = _diff_forward(y, base, {})
    auc_diff = float(np.mean(cross_val_score(Ridge(alpha=1.0), X, T_diff, cv=5, scoring="r2")))

    assert (
        auc_ratio > auc_diff + 0.1
    ), f"expected ratio to materially beat diff on a volatility-style multiplicative target, got ratio_r2={auc_ratio:.4f} vs diff_r2={auc_diff:.4f}"


def test_recommend_transform_candidates_drops_additive_for_positive_scale_pair():
    """Recommend transform candidates drops additive for positive scale pair."""
    _, y, base = _make_volatility_style_dataset(n=200, seed=1)
    candidates = ["diff", "linear_residual", "ratio", "logratio"]
    recommended = recommend_transform_candidates(y, base, candidates)
    assert "diff" not in recommended
    assert "linear_residual" not in recommended
    assert "ratio" in recommended and "logratio" in recommended
    assert len(recommended) < len(candidates)


def test_recommend_transform_candidates_keeps_additive_when_not_strictly_positive():
    """Recommend transform candidates keeps additive when not strictly positive."""
    rng = np.random.default_rng(2)
    y = rng.normal(size=200)  # can be negative
    base = rng.uniform(1.0, 5.0, size=200)
    candidates = ["diff", "linear_residual", "ratio", "logratio"]
    recommended = recommend_transform_candidates(y, base, candidates)
    assert recommended == candidates


def test_recommend_transform_candidates_never_drops_below_one_when_no_multiplicative_alternative():
    """Recommend transform candidates never drops below one when no multiplicative alternative."""
    y = np.abs(np.random.default_rng(3).normal(size=100)) + 0.1
    base = np.abs(np.random.default_rng(4).normal(size=100)) + 0.1
    candidates = ["diff", "linear_residual"]
    recommended = recommend_transform_candidates(y, base, candidates)
    assert recommended == candidates


def test_recommend_transform_candidates_auto_detect_is_opt_in_noop_when_omitted():
    # Bit-identical default: a caller who never passes ``auto_detect`` gets the exact old behavior, even for
    # a genuinely multiplicative pair where the caller forgot to list "ratio" among candidate_transforms.
    """Recommend transform candidates auto detect is opt in noop when omitted."""
    _, y, base = _make_volatility_style_dataset(n=200, seed=5)
    candidates = ["diff", "linear_residual"]  # no multiplicative candidate offered
    recommended = recommend_transform_candidates(y, base, candidates)
    assert recommended == candidates  # unchanged: no auto_detect, no multiplicative alternative present


def test_biz_val_recommend_transform_candidates_auto_detect_finds_multiplicative_regime():
    # The gap this closes: WITHOUT auto_detect, a caller who only offers additive candidates (never having
    # asserted "this pair is multiplicative") gets stuck with diff/linear_residual only, even though the
    # pair is genuinely ratio-stationary (Optiver-style volatility regime). auto_detect probes the data
    # itself and both (a) surfaces "ratio" without the caller asserting the regime, and (b) proves that
    # doing so is a real downstream quality win, not just a label change.
    """Biz val recommend transform candidates auto detect finds multiplicative regime."""
    X, y, base = _make_volatility_style_dataset(n=500, seed=6)
    candidates = ["diff", "linear_residual"]  # caller never asserts a multiplicative regime

    recommended_naive = recommend_transform_candidates(y, base, candidates)
    assert recommended_naive == candidates  # confirms the gap: no auto_detect -> stuck on additive-only

    recommended_auto = recommend_transform_candidates(y, base, candidates, auto_detect=True)
    assert "ratio" in recommended_auto
    assert "diff" not in recommended_auto and "linear_residual" not in recommended_auto

    # Downstream quality: the additive-only baseline a naive caller would have evaluated vs. the
    # auto-detected ratio candidate, on genuinely held-out CV folds.
    T_diff = _diff_forward(y, base, {})
    r2_diff_baseline = float(np.mean(cross_val_score(Ridge(alpha=1.0), X, T_diff, cv=5, scoring="r2")))

    ratio_params = _ratio_fit(y, base)
    T_ratio = _ratio_forward(y, base, ratio_params)
    r2_ratio_auto = float(np.mean(cross_val_score(Ridge(alpha=1.0), X, T_ratio, cv=5, scoring="r2")))

    assert r2_ratio_auto > r2_diff_baseline + 0.1, (
        f"expected auto-detected ratio to materially beat the additive-only baseline a naive caller would "
        f"be stuck with, got ratio_r2={r2_ratio_auto:.4f} vs diff_baseline_r2={r2_diff_baseline:.4f}"
    )
