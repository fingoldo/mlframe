"""Unit tests for `bayesian_alpha_fit_bootstrap`.

The conjugate variant is already covered by `test_composite_bayesian_alpha.py`
and `test_composite_bayesian_alpha_conjugate.py`. The bootstrap variant
(opt-in fallback for non-Gaussian residuals) had no dedicated test file.

Contract covered:
1. Return dict shape matches the documented keys.
2. Seed determinism: identical (y, base, random_state) -> identical means/CIs.
3. n_bootstrap=1 edge: returns a 1-sample posterior with NaN std.
4. CI containment: under a known linear DGP the 95% CI for alpha contains the truth.
5. CI ordering: ci_low <= ci_high for both alpha and beta.
6. Degenerate n<4: returns the documented NaN-std fallback.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite_bayesian import bayesian_alpha_fit_bootstrap


_REQUIRED_KEYS = {
    "alpha_mean", "alpha_std", "alpha_ci_low", "alpha_ci_high",
    "beta_mean", "beta_std", "beta_ci_low", "beta_ci_high",
    "alpha_samples", "beta_samples", "n_bootstrap", "ci_level",
}


def _synthetic_linear(n: int = 500, alpha: float = 2.5, beta: float = -1.0, noise_sd: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    y = alpha * base + beta + rng.normal(scale=noise_sd, size=n)
    return y, base


def test_bootstrap_returns_documented_dict_shape():
    y, base = _synthetic_linear(n=200, seed=1)
    out = bayesian_alpha_fit_bootstrap(y, base, n_bootstrap=50, random_state=7)
    missing = _REQUIRED_KEYS - set(out.keys())
    assert not missing, f"missing keys: {missing}"
    # Type / shape contract.
    assert out["n_bootstrap"] == 50
    assert out["alpha_samples"].shape == (50,)
    assert out["beta_samples"].shape == (50,)
    assert out["ci_level"] == 0.95


def test_bootstrap_seed_determinism():
    y, base = _synthetic_linear(n=300, seed=11)
    a = bayesian_alpha_fit_bootstrap(y, base, n_bootstrap=60, random_state=42)
    b = bayesian_alpha_fit_bootstrap(y, base, n_bootstrap=60, random_state=42)
    assert a["alpha_mean"] == b["alpha_mean"]
    assert a["beta_mean"] == b["beta_mean"]
    assert np.array_equal(a["alpha_samples"], b["alpha_samples"])
    assert np.array_equal(a["beta_samples"], b["beta_samples"])


def test_bootstrap_n_bootstrap_one_edge_nan_std():
    y, base = _synthetic_linear(n=200, seed=2)
    out = bayesian_alpha_fit_bootstrap(y, base, n_bootstrap=1, random_state=0)
    assert out["n_bootstrap"] == 1
    # ddof=1 std on a 1-sample is undefined; module pins NaN.
    assert np.isnan(out["alpha_std"])
    assert np.isnan(out["beta_std"])
    # Single sample arrays.
    assert out["alpha_samples"].shape == (1,)
    assert out["beta_samples"].shape == (1,)


def test_bootstrap_ci_contains_truth_on_well_specified_dgp():
    alpha_true, beta_true = 2.5, -1.0
    y, base = _synthetic_linear(n=2000, alpha=alpha_true, beta=beta_true, noise_sd=0.3, seed=3)
    out = bayesian_alpha_fit_bootstrap(y, base, n_bootstrap=300, random_state=0)
    assert out["alpha_ci_low"] <= alpha_true <= out["alpha_ci_high"], (
        f"alpha truth {alpha_true} outside CI [{out['alpha_ci_low']}, {out['alpha_ci_high']}]"
    )
    assert out["beta_ci_low"] <= beta_true <= out["beta_ci_high"], (
        f"beta truth {beta_true} outside CI [{out['beta_ci_low']}, {out['beta_ci_high']}]"
    )
    # CI ordering invariant.
    assert out["alpha_ci_low"] <= out["alpha_ci_high"]
    assert out["beta_ci_low"] <= out["beta_ci_high"]
    # Posterior std must be positive on the well-specified case.
    assert out["alpha_std"] > 0
    assert out["beta_std"] > 0


def test_bootstrap_degenerate_n_lt_4():
    # 3 observations is below the floor n>=4; the function pins a documented
    # NaN-std degenerate-posterior return so callers can detect the case.
    y = np.array([1.0, 2.0, 3.0])
    base = np.array([0.5, 1.0, 1.5])
    out = bayesian_alpha_fit_bootstrap(y, base, n_bootstrap=10)
    assert out["n_bootstrap"] == 0
    assert np.isnan(out["alpha_std"])
    assert np.isnan(out["alpha_ci_low"])
    assert np.isnan(out["alpha_ci_high"])
    assert np.isnan(out["beta_std"])
    # Point estimate still returned.
    assert np.isfinite(out["alpha_mean"])
    assert np.isfinite(out["beta_mean"])


def test_bootstrap_subsample_n_caps_per_draw_size():
    y, base = _synthetic_linear(n=1000, seed=4)
    # subsample_n smaller than n: each bootstrap draw uses subsample_n rows.
    # The result must still be valid (finite, sensible CIs).
    out = bayesian_alpha_fit_bootstrap(
        y, base, n_bootstrap=80, random_state=0, subsample_n=200,
    )
    assert np.isfinite(out["alpha_mean"])
    assert np.isfinite(out["beta_mean"])
    # Posterior is wider than full-sample (less data per draw -> more variance);
    # but we only assert the CI width is positive.
    assert out["alpha_ci_high"] > out["alpha_ci_low"]


def test_bootstrap_ci_level_parameter_widens_or_narrows():
    # Wider CI level -> wider interval. 99% must be wider than 50% on the same data.
    y, base = _synthetic_linear(n=500, seed=5)
    narrow = bayesian_alpha_fit_bootstrap(y, base, n_bootstrap=200, ci_level=0.50, random_state=0)
    wide = bayesian_alpha_fit_bootstrap(y, base, n_bootstrap=200, ci_level=0.99, random_state=0)
    narrow_width = narrow["alpha_ci_high"] - narrow["alpha_ci_low"]
    wide_width = wide["alpha_ci_high"] - wide["alpha_ci_low"]
    assert wide_width > narrow_width
