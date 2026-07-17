"""biz_val tests for ``mlframe.training.composite.discovery.screening`` +
``mlframe.training.composite_transforms`` -- pure functions tested
outside of a pipeline fit.

Per CLAUDE.md: each test asserts a SYNTHETIC measurable WIN.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# _residualise (from composite_screening.py) -- removes linear component
# ---------------------------------------------------------------------------


def test_biz_val_residualise_removes_linear_component():
    """y = 3*x + noise -> after _residualise, |corr(residual, x)| ~ 0
    AND mean(residual) ~ 0. Catches regressions in the regression
    projection."""
    from mlframe.training.composite.discovery.screening import _residualise

    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(size=n)
    y = 3.0 * x + 0.5 * rng.normal(size=n)
    resid = _residualise(y, x)
    corr = float(np.corrcoef(resid, x)[0, 1])
    assert abs(corr) < 1e-6, f"residual must have zero correlation with x; got {corr:.6f}"
    assert abs(float(np.mean(resid))) < 1e-12, f"mean residual must be zero; got {np.mean(resid):.6e}"


@pytest.mark.parametrize("n_samples", [50, 200, 1000])
def test_biz_val_residualise_scales_with_size(n_samples):
    """_residualise must handle small / medium / large N."""
    from mlframe.training.composite.discovery.screening import _residualise

    rng = np.random.default_rng(42)
    x = rng.normal(size=n_samples)
    y = 1.5 * x + 0.3 * rng.normal(size=n_samples)
    resid = _residualise(y, x)
    assert len(resid) == n_samples
    assert np.all(np.isfinite(resid))


# ---------------------------------------------------------------------------
# _safe_corr (from composite_screening.py)
# ---------------------------------------------------------------------------


def test_biz_val_safe_corr_perfect_correlation():
    """``_safe_corr`` with y = x must return 1.0."""
    from mlframe.training.composite.discovery.screening import _safe_corr

    rng = np.random.default_rng(42)
    x = rng.normal(size=200)
    corr = _safe_corr(x, x)
    assert abs(corr - 1.0) < 1e-12, f"y=x correlation must be 1; got {corr:.6f}"


def test_biz_val_safe_corr_constant_input():
    """Constant input must NOT crash _safe_corr (zero-variance guard).
    Returns NaN or 0 rather than division-by-zero."""
    from mlframe.training.composite.discovery.screening import _safe_corr

    x = np.ones(200, dtype=np.float64)
    y = np.random.default_rng(42).normal(size=200)
    corr = _safe_corr(x, y)
    # Behavioural: zero-variance input must return a finite sentinel (0.0 or NaN) per the documented guard, NOT
    # raise and NOT return inf. ``is not None`` alone passed even when the guard returned a divide-by-zero inf.
    assert corr is not None, "_safe_corr returned None on zero-variance input"
    assert np.isnan(corr) or corr == 0.0, f"_safe_corr on zero-variance input must return NaN or 0.0, got {corr!r}"


# ---------------------------------------------------------------------------
# Transform forward/inverse roundtrip (from composite_transforms.py)
# ---------------------------------------------------------------------------


def test_biz_val_diff_transform_roundtrip_identity():
    """``_diff_forward`` then ``_diff_inverse`` must recover the
    original y exactly (diff is just y - base, so inversion is
    t_hat + base = y)."""
    from mlframe.training.composite import (
        _diff_fit,
        _diff_forward,
        _diff_inverse,
    )

    rng = np.random.default_rng(42)
    y = np.cumsum(rng.normal(0, 0.5, size=100)).astype(np.float64)
    base = np.roll(y, shift=1).astype(np.float64)
    params = _diff_fit(y, base)
    t_hat = _diff_forward(y, base, params)
    y_roundtrip = _diff_inverse(t_hat, base, params)
    # Perfect recovery: diff(t, base) = t - base; inverse = t_hat + base = y
    max_err = float(np.max(np.abs(y_roundtrip - y)))
    assert max_err < 1e-12, f"diff roundtrip must be exact; got max|error|={max_err:.2e}"


def test_biz_val_linear_residual_transform_roundtrip():
    """``_linear_residual_forward`` then ``_linear_residual_inverse``
    must recover the original y within numerical precision."""
    from mlframe.training.composite import (
        _linear_residual_fit,
        _linear_residual_forward,
        _linear_residual_inverse,
    )

    rng = np.random.default_rng(42)
    n = 100
    x = rng.normal(size=n).astype(np.float64)
    y = (1.0 + 2.5 * x + 0.2 * rng.normal(size=n)).astype(np.float64)
    params = _linear_residual_fit(y, x)
    t_hat = _linear_residual_forward(y, x, params)
    y_roundtrip = _linear_residual_inverse(t_hat, x, params)
    max_err = float(np.max(np.abs(y_roundtrip - y)))
    assert max_err < 1e-10, f"linear_residual roundtrip must be nearly exact; got max|error|={max_err:.2e}"


def test_biz_val_linear_residual_fit_recovers_true_coefficients():
    """On y = 1.0 + 2.5*x + tiny noise, ``_linear_residual_fit``
    must recover slope ~2.5, intercept ~1.0. Note: in this codebase
    ``alpha`` = slope, ``beta`` = intercept (inverted naming)."""
    from mlframe.training.composite import _linear_residual_fit

    rng = np.random.default_rng(42)
    n = 500
    x = rng.normal(size=n).astype(np.float64)
    y = (1.0 + 2.5 * x + 0.05 * rng.normal(size=n)).astype(np.float64)
    params = _linear_residual_fit(y, x)
    slope = params.get("alpha")  # misnomer: alpha = slope in this codebase
    intercept = params.get("beta")  # beta = intercept
    assert slope is not None and abs(slope - 2.5) < 0.1, f"slope must be ~2.5; got {slope:.4f}"
    assert abs(intercept - 1.0) < 0.1, f"intercept must be ~1.0; got {intercept:.4f}"


def test_biz_val_ratio_transform_finite_roundtrip():
    """``_ratio_forward`` / ``_ratio_inverse`` roundtrip on
    positive inputs is algebraically exact."""
    from mlframe.training.composite import (
        _ratio_fit,
        _ratio_forward,
        _ratio_inverse,
    )

    rng = np.random.default_rng(42)
    y = np.abs(rng.normal(loc=10.0, scale=2.0, size=100).astype(np.float64))
    base = np.abs(rng.normal(loc=5.0, scale=1.0, size=100).astype(np.float64))
    params = _ratio_fit(y, base)
    t_hat = _ratio_forward(y, base, params)
    y_roundtrip = _ratio_inverse(t_hat, base, params)
    max_err = float(np.max(np.abs(y_roundtrip - y)))
    assert max_err < 1e-12, f"ratio roundtrip must be exact; got max|error|={max_err:.2e}"
