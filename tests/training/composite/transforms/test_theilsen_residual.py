"""Unit + biz_value tests for the ``theilsen_residual`` composite transform.

``theilsen_residual`` is a high-breakdown robust variant of
``linear_residual``: the slope is the median of pairwise slopes (Theil-Sen),
the intercept the median of ``y - alpha*base``. On a target with gross
outliers in EITHER y or base it should recover the true slope far better than
OLS (``linear_residual``) and, by extension, drive a lower downstream RMSE on
clean test rows.

Per CLAUDE.md each new transform ships unit + a quantitative biz_value test;
the biz_value assertions pin a measured win with margin.
"""

from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from mlframe.training.composite.transforms import TRANSFORMS_REGISTRY
from mlframe.training.composite.transforms.linear import (
    _linear_residual_fit,
    _theilsen_residual_fit,
)

# ---------------------------------------------------------------------------
# Unit: registry wiring, fit contract, round-trip.
# ---------------------------------------------------------------------------


def test_theilsen_registered_and_uses_linear_forward_inverse():
    """Theilsen registered and uses linear forward inverse."""
    t = TRANSFORMS_REGISTRY["theilsen_residual"]
    assert t.requires_base is True
    assert t.requires_groups is False
    # Forward / inverse are the shared linear_residual algebra.
    lin = TRANSFORMS_REGISTRY["linear_residual"]
    assert t.forward is lin.forward
    assert t.inverse is lin.inverse
    assert t.domain_check is lin.domain_check


def test_theilsen_fit_returns_alpha_beta_dict():
    """Theilsen fit returns alpha beta dict."""
    rng = np.random.default_rng(0)
    base = np.linspace(0.0, 10.0, 200)
    y = 2.0 * base + 3.0 + rng.standard_normal(200) * 0.1
    p = _theilsen_residual_fit(y, base)
    assert set(p) == {"alpha", "beta"}
    assert abs(p["alpha"] - 2.0) < 0.05
    assert abs(p["beta"] - 3.0) < 0.2


def test_theilsen_fit_round_trip_identity_on_clean_data():
    """Theilsen fit round trip identity on clean data."""
    rng = np.random.default_rng(1)
    base = np.linspace(0.0, 10.0, 150)
    y = 1.3 * base - 0.7 + rng.standard_normal(150) * 0.05
    t = TRANSFORMS_REGISTRY["theilsen_residual"]
    p = t.fit(y, base)
    resid = t.forward(y, base, p)
    y_back = t.inverse(resid, base, p)
    assert np.allclose(y_back, y, atol=1e-9)


def test_theilsen_fit_does_not_mutate_inputs():
    """Theilsen fit does not mutate inputs."""
    base = np.linspace(0.0, 10.0, 80)
    y = 0.5 * base + np.linspace(-1, 1, 80)
    y_snap, base_snap = y.copy(), base.copy()
    _theilsen_residual_fit(y, base)
    np.testing.assert_array_equal(y, y_snap)
    np.testing.assert_array_equal(base, base_snap)


def test_theilsen_fit_handles_non_finite_rows():
    """Theilsen fit handles non finite rows."""
    base = np.linspace(0.0, 10.0, 50)
    y = 2.0 * base + 1.0
    y[0] = np.nan
    base[1] = np.inf
    p = _theilsen_residual_fit(y, base)
    assert np.isfinite(p["alpha"]) and np.isfinite(p["beta"])
    assert abs(p["alpha"] - 2.0) < 0.05


def test_theilsen_fit_constant_base_degenerate():
    """Theilsen fit constant base degenerate."""
    base = np.full(50, 5.0)
    y = np.linspace(0.0, 10.0, 50)
    p = _theilsen_residual_fit(y, base)
    # No slope information -> alpha 0, beta = median(y).
    assert p["alpha"] == 0.0
    assert abs(p["beta"] - float(np.median(y))) < 1e-9


def test_theilsen_fit_subsamples_for_large_n_and_is_deterministic():
    # Force a tiny pair cap so the subsample branch executes on a small array.
    """Theilsen fit subsamples for large n and is deterministic."""
    import mlframe.training.composite.transforms as tmod
    import mlframe.training.composite.transforms.linear as lin

    rng = np.random.default_rng(7)
    base = np.linspace(0.0, 10.0, 300)
    y = 1.7 * base + 0.4 + rng.standard_normal(300) * 0.1
    orig = lin._THEILSEN_MAX_PAIRS
    try:
        lin._THEILSEN_MAX_PAIRS = 2000  # << 300*299/2 = 44850 full pairs
        p1 = _theilsen_residual_fit(y, base)
        p2 = _theilsen_residual_fit(y, base)
    finally:
        lin._THEILSEN_MAX_PAIRS = orig
    # Deterministic (seeded generator) and still close to the true slope.
    assert p1 == p2
    assert abs(p1["alpha"] - 1.7) < 0.1
    # The module-level constant default is unchanged.
    assert tmod._THEILSEN_MAX_PAIRS == orig


# ---------------------------------------------------------------------------
# biz_value: 10% gross outliers in (y, base) -> Theil-Sen recovers the true
# slope >=3x better than OLS, and downstream RMSE on clean test rows improves.
# ---------------------------------------------------------------------------


def _make_contaminated(rng, n=600, true_alpha=2.0, true_beta=5.0, frac=0.10):
    """Make contaminated."""
    base = rng.uniform(-10.0, 10.0, size=n)
    y = true_alpha * base + true_beta + rng.standard_normal(n) * 0.3
    n_out = int(frac * n)
    idx = rng.choice(n, size=n_out, replace=False)
    # Gross outliers in BOTH y and base (leverage + vertical outliers).
    y[idx] += rng.standard_normal(n_out) * 80.0 + 60.0
    base[idx] += rng.standard_normal(n_out) * 40.0
    return base, y, idx


def test_biz_val_theilsen_recovers_slope_3x_better_than_ols():
    """With 10% gross outliers in (y, base), Theil-Sen slope error must be at
    least 3x lower than OLS slope error. Measured ratio is far above 3x; the
    floor absorbs seed variation."""
    rng = np.random.default_rng(2024)
    true_alpha = 2.0
    ts_errs, ols_errs = [], []
    for _ in range(5):
        base, y, _ = _make_contaminated(rng, true_alpha=true_alpha)
        p_ts = _theilsen_residual_fit(y, base)
        p_ols = _linear_residual_fit(y, base)
        ts_errs.append(abs(p_ts["alpha"] - true_alpha))
        ols_errs.append(abs(p_ols["alpha"] - true_alpha))
    ts_err = float(np.median(ts_errs))
    ols_err = float(np.median(ols_errs))
    assert (
        ols_err / max(ts_err, 1e-9) >= 3.0
    ), f"Theil-Sen slope err {ts_err:.4f} should be >=3x better than OLS {ols_err:.4f} (ratio {ols_err / max(ts_err, 1e-9):.2f})"
    assert ts_err < 0.15, f"Theil-Sen slope err {ts_err:.4f} too high"


def test_biz_val_theilsen_lowers_clean_test_rmse_vs_ols():
    """Train both fits on contaminated rows, then evaluate the linear model
    alpha*base+beta on CLEAN held-out rows. Theil-Sen's RMSE on the clean
    test set must beat OLS by a clear margin."""
    rng = np.random.default_rng(99)
    true_alpha, true_beta = 2.0, 5.0
    # Contaminated train.
    base_tr, y_tr, _ = _make_contaminated(
        rng,
        n=600,
        true_alpha=true_alpha,
        true_beta=true_beta,
    )
    p_ts = _theilsen_residual_fit(y_tr, base_tr)
    p_ols = _linear_residual_fit(y_tr, base_tr)
    # Clean test rows (no contamination).
    base_te = rng.uniform(-10.0, 10.0, size=400)
    y_te = true_alpha * base_te + true_beta + rng.standard_normal(400) * 0.3

    def _rmse(p):
        """Rmse."""
        pred = p["alpha"] * base_te + p["beta"]
        return float(np.sqrt(np.mean((pred - y_te) ** 2)))

    rmse_ts = _rmse(p_ts)
    rmse_ols = _rmse(p_ols)
    assert rmse_ts < rmse_ols * 0.5, f"Theil-Sen clean-test RMSE {rmse_ts:.3f} should be well below OLS {rmse_ols:.3f}"
