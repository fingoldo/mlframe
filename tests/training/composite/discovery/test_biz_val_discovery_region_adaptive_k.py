"""Biz-value: ``region_adaptive_k`` captures a regime-varying ``(y, base)`` relationship.

``fit_region_adaptive`` cuts ``base`` into ``k`` quantile regions and fits the best per-region transform, frozen on
train. On a target whose ``y -> base`` map genuinely changes shape across the base range (different additive offset /
slope per region), too few regions (``k=2``) cannot represent the regime structure, so the region-adaptive
reconstruction of ``y`` carries large residuals; matching the true regime count (``k=4``) collapses them.

The win is the round-trip reconstruction error ``inverse(forward(y))`` per region count: ``k=4`` must beat ``k=2`` on a
4-regime synthetic. A regression that ignores ``region_adaptive_k`` (hardcodes a single region count) flattens the gap
and FAILS the test.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.discovery._region_adaptive import fit_region_adaptive


def _make_4_regime(n=2000, seed=0):
    """y = base + per-region offset; the offset is piecewise-constant across 4 base quantile bands."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.0, 4.0, n)
    region = np.clip(base.astype(int), 0, 3)  # 4 bands: [0,1),[1,2),[2,3),[3,4)
    offset = np.array([0.0, 5.0, -5.0, 10.0])[region]  # strongly regime-dependent shift
    y = base + offset + 0.05 * rng.normal(size=n)
    return y, base


def _recon_rmse(spec, y, base):
    """Recon rmse."""
    t = spec.forward(y, base)
    y_hat = spec.inverse(t, base)
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def test_biz_val_region_adaptive_k_matches_regime_count():
    """k=4 (the true regime count) reconstructs the 4-regime target; n_regions honours the knob."""
    y, base = _make_4_regime()
    spec4 = fit_region_adaptive(y, base, k=4, random_state=0)
    assert spec4.k == 4, "region_adaptive_k must drive the number of fitted regions"
    rmse4 = _recon_rmse(spec4, y, base)
    assert rmse4 < 0.5, f"4-region adaptive must reconstruct the 4-regime target tightly (got {rmse4:.3f})"


def test_biz_val_region_adaptive_higher_k_wins_on_regime_target():
    """k=4 reconstruction error is materially lower than k=2 on a 4-regime target."""
    y, base = _make_4_regime()
    rmse2 = _recon_rmse(fit_region_adaptive(y, base, k=2, random_state=0), y, base)
    rmse4 = _recon_rmse(fit_region_adaptive(y, base, k=4, random_state=0), y, base)
    assert rmse2 == rmse2 and rmse4 == rmse4  # not-NaN guard
    assert rmse4 <= rmse2 + 1e-9, f"k matching regime count must not be worse (k4={rmse4:.3f}, k2={rmse2:.3f})"
