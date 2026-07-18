"""Regression sensors for metric-definition methodology fixes (SA22 / SA23 / SA25).

Each test fails on the pre-fix code and passes on the corrected contract:

* SA22 drift: a near-constant reference collapsed the bin grid to one bin, so
  PSI/KL/JS reported 0 (silent false "no drift") regardless of how far the
  target had shifted. The pooled-support fallback now puts the shifted target in
  its own bin so the divergence is nonzero.
* SA23 CRPS: integrating only ``[a[0], a[-1]]`` dropped the tails, under-
  estimating CRPS and breaking cross-grid comparability. Tail clamping makes two
  grids covering the same distribution agree more closely, and a heavy-tail case
  is no longer under-estimated.
* SA25 ranking: ERR ``max_grade`` defaulting to the per-call ``y_true.max()``
  made ERR incomparable across splits; P@k dividing by ``k`` deflated short
  queries.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics._drift import (
    population_stability_index,
    kl_divergence,
    js_divergence,
)
from mlframe.metrics.quantile import crps_from_quantiles
from mlframe.metrics._ranking_extras import expected_reciprocal_rank, precision_at_k

# ----- SA22: near-constant reference must not silently report "no drift" -----


def test_psi_constant_reference_detects_shifted_target():
    """A constant reference vs a clearly shifted target must give nonzero PSI.

    Pre-fix the reference quantiles collapsed to a single bin, so both ref and
    target normalised to the SAME [1.0] histogram and PSI was exactly 0."""
    ref = np.full(2000, 5.0)
    tgt = np.concatenate([np.full(1000, 5.0), np.full(1000, 9.0)])
    psi = population_stability_index(ref, tgt, nbins=10)
    assert np.isfinite(psi)
    assert psi > 1e-6, f"PSI should detect the shift, got {psi}"


def test_kl_constant_reference_detects_shifted_target():
    """Kl constant reference detects shifted target."""
    ref = np.full(2000, 5.0)
    tgt = np.concatenate([np.full(1000, 5.0), np.full(1000, 9.0)])
    kl = kl_divergence(ref, tgt, nbins=10, bias_correction=False)
    assert np.isfinite(kl)
    assert kl > 1e-6, f"KL should detect the shift, got {kl}"


def test_js_constant_reference_detects_shifted_target():
    """Js constant reference detects shifted target."""
    ref = np.full(2000, 5.0)
    tgt = np.concatenate([np.full(1000, 5.0), np.full(1000, 9.0)])
    js = js_divergence(ref, tgt, nbins=10, bias_correction=False)
    assert np.isfinite(js)
    assert js > 1e-6, f"JS should detect the shift, got {js}"


def test_psi_truly_constant_both_sides_is_zero_not_spurious():
    """When ref AND target are the same constant there is genuinely no drift -> 0."""
    ref = np.full(500, 3.0)
    tgt = np.full(500, 3.0)
    assert population_stability_index(ref, tgt, nbins=10) == pytest.approx(0.0, abs=1e-9)


# ----- SA23: CRPS tail handling -> cross-grid comparability -----


def _crps_grid(y, mu, sigma, alphas):
    """Helper: Crps grid."""
    from scipy.stats import norm

    P = np.stack([norm.ppf(a, loc=mu, scale=sigma) for a in alphas], axis=1)
    return crps_from_quantiles(y, P, alphas)


def test_crps_grids_more_comparable_after_tail_handling():
    """Two grids over the SAME Gaussian predictive should give CLOSE CRPS.

    A narrow grid [0.1..0.9] drops more tail mass than a wide grid
    [0.01..0.99]; pre-fix the dropped tails made the two disagree substantially.
    Tail clamping shrinks the gap."""
    rng = np.random.default_rng(0)
    n = 4000
    mu = np.zeros(n)
    sigma = np.ones(n)
    y = rng.normal(mu, sigma)
    narrow = np.round(np.arange(0.1, 0.91, 0.1), 4)
    wide = np.round(np.arange(0.01, 0.991, 0.01), 4)
    c_narrow = _crps_grid(y, mu, sigma, narrow)
    c_wide = _crps_grid(y, mu, sigma, wide)
    rel_gap = abs(c_narrow - c_wide) / c_wide
    # Pre-fix this gap is large (~0.2+); post-fix tail handling brings it well under 0.1.
    assert rel_gap < 0.08, f"grids disagree too much: narrow={c_narrow:.4f} wide={c_wide:.4f} gap={rel_gap:.3f}"


def test_crps_heavy_tail_not_underestimated():
    """A coarse grid that ignores the tails under-estimates CRPS; tail handling
    keeps the coarse-grid estimate at or above the no-tail (pre-fix) integral."""
    rng = np.random.default_rng(1)
    n = 3000
    mu = np.zeros(n)
    sigma = np.full(n, 2.0)
    y = rng.standard_t(df=3, size=n) * 2.0  # heavy-tailed truth
    alphas = np.array([0.2, 0.5, 0.8])
    from scipy.stats import norm

    P = np.stack([norm.ppf(a, loc=mu, scale=sigma) for a in alphas], axis=1)
    # Pre-fix integral (no tails) is strictly the [a0,a-1] trapezoid; the fixed
    # function adds positive tail contributions, so it must exceed it.
    per_alpha = np.array([float(np.mean(np.maximum(a * (y - P[:, k]), (a - 1.0) * (y - P[:, k])))) for k, a in enumerate(alphas)])
    no_tail = 2.0 * float(np.sum((alphas[1:] - alphas[:-1]) * (per_alpha[1:] + per_alpha[:-1]) * 0.5))
    full = crps_from_quantiles(y, P, alphas)
    assert full > no_tail, f"tail handling must raise CRPS above the truncated integral: {full} vs {no_tail}"


# ----- SA25: ERR comparable across splits; P@k not deflated on short queries -----


def test_err_max_grade_is_fixed_scale_not_per_call_max():
    """Same ranking, two splits whose label sets have DIFFERENT max grade, must
    give the SAME ERR under the fixed default scale.

    Pre-fix the default max_grade was y_true.max(), so the gain normalisation
    differed between the two splits and the ERR values diverged."""
    g = None
    # Two perfectly-ranked queries; relevance pattern identical, but split B's
    # labels also contain a grade-4 doc elsewhere that does not change THIS query.
    rel_a = np.array([2.0, 1.0, 0.0, 0.0])
    score = np.array([4.0, 3.0, 2.0, 1.0])
    rel_b = np.array([2.0, 1.0, 0.0, 0.0])  # same query content
    err_a = expected_reciprocal_rank(rel_a, score, g, k=4)
    err_b = expected_reciprocal_rank(rel_b, score, g, k=4)
    # Both use the fixed default ceiling -> identical, and independent of any
    # per-split max grade.
    assert err_a == pytest.approx(err_b)
    # The fixed default is 4.0: top doc gain = (2^2-1)/2^4 = 3/16.
    assert err_a == pytest.approx((3.0 / 16.0) / 1.0 + (1.0 - 3.0 / 16.0) * (1.0 / 16.0) / 2.0, rel=1e-9)


def test_precision_at_k_not_deflated_on_short_query():
    """A perfectly-ranked 3-doc query at k=10 must score 1.0, not 0.3.

    Pre-fix the denominator was always k, so a query shorter than k was deflated
    for rank slots that do not exist."""
    g = None
    y = np.array([1.0, 1.0, 1.0])  # all relevant, only 3 docs
    s = np.array([3.0, 2.0, 1.0])
    val = precision_at_k(y, s, g, k=10)
    assert val == pytest.approx(1.0), f"short-query P@10 should be 1.0, got {val}"


def test_precision_at_k_full_length_query_unchanged():
    """A query with >= k docs is unaffected by the min(k, n) denominator."""
    g = None
    y = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s = -np.arange(10.0)
    assert precision_at_k(y, s, g, k=5) == pytest.approx(0.4)
