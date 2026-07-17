"""Biz_value + unit tests for the normalized (locally-adaptive) conformal default.

``calibrate_conformal`` defaults to ``score="normalized"``: it divides residuals by
a binned conditional residual-scale ``sigma_hat(x)`` so the band widens where the
model is noisier. On heteroscedastic targets this restores CONDITIONAL coverage,
where the legacy ``score="absolute"`` (constant-width band) under-covers the
high-variance region and over-covers the low-variance region.

Bench: ``mlframe/training/composite/_benchmarks/bench_conformal_normalized_vs_absolute.py``
(25/25 het cells, worst-bin coverage gap 0.042 vs 0.227). These tests exercise the
REAL estimator API and pin the win quantitatively so a regression to absolute-only
trips them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression

from mlframe.training.composite import CompositeTargetEstimator


def _fit_het(seed, score, alpha=0.1, n=6000):
    """Heteroscedastic residual setup: noise scale grows with the base column.

    Returns (worst-bin coverage gap across x-bins, marginal coverage, mean width).
    """
    rng = np.random.default_rng(seed)
    b = rng.uniform(0.0, 1.0, n)
    sigma = 0.2 + 4.0 * b  # noise scale tied to b -> strongly heteroscedastic
    y = 10.0 * b + rng.normal(0.0, 1.0, n) * sigma
    X = pd.DataFrame({"b": b})
    nf = n // 3
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="linear_residual",
        base_column="b",
    )
    est.fit(X.iloc[:nf], y[:nf])
    est.calibrate_conformal(X.iloc[nf : 2 * nf], y[nf : 2 * nf], alpha=alpha, score=score)
    lo, hi = est.predict_interval(X.iloc[2 * nf :], alpha)
    ye = y[2 * nf :]
    be = b[2 * nf :]
    covered = (ye >= lo) & (ye <= hi)
    marg = float(covered.mean())
    width = float(np.mean(hi - lo))
    edges = np.quantile(be, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(edges, be, side="right") - 1, 0, 9)
    gaps = [abs(covered[idx == k].mean() - (1 - alpha)) for k in range(10) if (idx == k).sum() >= 20]
    return max(gaps), marg, width


class TestNormalizedDefault:
    def test_default_score_is_normalized(self) -> None:
        rng = np.random.default_rng(0)
        b = rng.uniform(0, 1, 900)
        y = 10 * b + rng.normal(0, 1, 900) * (0.2 + 4 * b)
        X = pd.DataFrame({"b": b})
        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="linear_residual",
            base_column="b",
        )
        est.fit(X.iloc[:300], y[:300])
        est.calibrate_conformal(X.iloc[300:600], y[300:600], alpha=0.1)
        # Default must populate the sigma model (normalized path), not just a scalar radius.
        assert getattr(est, "_conformal_sigma_", None), "default calibrate_conformal must be normalized"
        lo, hi = est.predict_interval(X.iloc[600:], 0.1)
        # Normalized -> variable-width band; absolute would be constant width.
        w = hi - lo
        assert float(w.max() - w.min()) > 1e-6, "normalized band must vary in width"

    def test_absolute_opt_out_is_constant_width(self) -> None:
        rng = np.random.default_rng(1)
        b = rng.uniform(0, 1, 900)
        y = 10 * b + rng.normal(0, 1, 900) * (0.2 + 4 * b)
        X = pd.DataFrame({"b": b})
        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="linear_residual",
            base_column="b",
        )
        est.fit(X.iloc[:300], y[:300])
        est.calibrate_conformal(X.iloc[300:600], y[300:600], alpha=0.1, score="absolute")
        assert not getattr(est, "_conformal_sigma_", {}), "absolute must not store a sigma model"
        lo, hi = est.predict_interval(X.iloc[600:], 0.1)
        w = hi - lo
        # Constant-width before envelope clipping; clipping can trim a few rows, so
        # assert the bulk of widths coincide.
        assert float(np.median(np.abs(w - np.median(w)))) < 1e-6, "absolute band must be constant width"

    def test_invalid_score_raises(self) -> None:
        rng = np.random.default_rng(2)
        X = pd.DataFrame({"b": rng.uniform(0, 1, 300)})
        y = 10 * X["b"].to_numpy() + rng.normal(0, 1, 300)
        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="linear_residual",
            base_column="b",
        )
        est.fit(X.iloc[:150], y[:150])
        with pytest.raises(ValueError):
            est.calibrate_conformal(X.iloc[150:], y[150:], alpha=0.1, score="bogus")


class TestNormalizedBizValue:
    def test_biz_normalized_better_conditional_coverage(self) -> None:
        """Normalized must beat absolute on worst-bin (conditional) coverage gap on a
        heteroscedastic target, on the majority of seeds. Bench: gap 0.042 vs 0.227;
        floor here is a conservative absolute_gap - 0.05 margin so noise doesn't trip it."""
        wins = 0
        for s in range(5):
            gap_n, _, _ = _fit_het(s, "normalized")
            gap_a, _, _ = _fit_het(s, "absolute")
            if gap_n < gap_a - 0.05:
                wins += 1
        assert wins >= 4, f"normalized beat absolute on conditional coverage in only {wins}/5 seeds"

    def test_biz_both_keep_marginal_coverage(self) -> None:
        """The flip preserves the split-conformal marginal guarantee for both scores."""
        for s in range(3):
            _, marg_n, _ = _fit_het(s, "normalized")
            _, marg_a, _ = _fit_het(s, "absolute")
            assert marg_n >= 0.86, f"normalized under-covered marginally: {marg_n:.3f}"
            assert marg_a >= 0.86, f"absolute under-covered marginally: {marg_a:.3f}"

    def test_biz_normalized_not_wider(self) -> None:
        """Normalized is also sharper (not wider) than absolute on heteroscedastic data."""
        for s in range(3):
            _, _, w_n = _fit_het(s, "normalized")
            _, _, w_a = _fit_het(s, "absolute")
            assert w_n <= w_a * 1.02, f"normalized width {w_n:.2f} should not exceed absolute {w_a:.2f}"
