"""Regression tests for numerical-stability audit fixes (audit2_numstab).

Each asserts the fixed behavior on the input regime that triggered the pre-fix bug:
cubic-mean negative-base NaN, Pearson product-overflow collapse, and chao-shen
bincount crash / rare-category cancellation.
"""


import numpy as np

from mlframe.feature_engineering._numerical_numba import compute_numerical_aggregates_numba
from mlframe.metrics.regression._regression_corr import fast_pearson_corr
from mlframe.feature_selection.filters._chao_shen import chao_shen_mi, chao_shen_entropy


def test_cubic_mean_net_negative_column_is_signed_cube_root_not_nan():
    """F1: mean-of-cubes is negative for a net-negative column; a bare (-x)**(1/3) returns NaN.
    The signed cube root must recover the real value (e.g. cube-root of mean(-25) = -2.924...)."""
    neg = np.array([-2.0, -3.0, -1.0, -4.0], dtype=np.float64)
    res = list(compute_numerical_aggregates_numba(neg, whiten_means=False))
    expected_qubic = -(25.0 ** (1.0 / 3.0))  # mean of cubes = -100/4 = -25
    assert any(np.isfinite(v) and abs(v - expected_qubic) < 1e-6 for v in res), (
        f"signed cubic-mean {expected_qubic:.4f} not found among finite aggregates {res}"
    )


def test_quadratic_cubic_mean_large_scale_do_not_overflow_to_inf():
    """F2: naive sum of x^2 / x^3 overflows to inf for extreme-scale columns (|x|~1e154 -> x^2,
    |x|~1e103 -> x^3), so RMS/cubic-mean become inf even though the true power-mean is finite. The
    scaled-recompute-on-overflow fallback must return the finite true value."""
    # Cubic overflows first (x^3 at ~1e103): a constant column's RMS and cubic-mean both equal the value.
    cubic_scale = np.full(8, 3e103, dtype=np.float64)
    res_c = list(compute_numerical_aggregates_numba(cubic_scale, whiten_means=False))
    assert np.isclose(res_c[13], 3e103, rtol=1e-9), f"quadratic_mean overflowed/incorrect: {res_c[13]:.3e}"
    assert np.isclose(res_c[14], 3e103, rtol=1e-9), f"qubic_mean overflowed/incorrect: {res_c[14]:.3e}"

    # Quadratic overflow regime (x^2 at ~1e154).
    quad_scale = np.full(8, 2e154, dtype=np.float64)
    res_q = list(compute_numerical_aggregates_numba(quad_scale, whiten_means=False))
    assert np.isclose(res_q[13], 2e154, rtol=1e-9), f"quadratic_mean overflowed: {res_q[13]:.3e}"
    assert np.isclose(res_q[14], 2e154, rtol=1e-9), f"qubic_mean overflowed: {res_q[14]:.3e}"

    # Normal-scale must be UNCHANGED (fast path) and match numpy exactly.
    np.random.seed(0)
    a = (np.random.randn(2000) * 5 + 2).astype(np.float64)
    res_n = list(compute_numerical_aggregates_numba(a, whiten_means=False))
    exp_quad = np.sqrt(np.mean(a**2))
    m3 = np.mean(a**3)
    exp_qubic = np.sign(m3) * np.abs(m3) ** (1 / 3)
    assert np.isclose(res_n[13], exp_quad, rtol=1e-9)
    assert np.isclose(res_n[14], exp_qubic, rtol=1e-9)


def test_pearson_corr_large_scale_not_collapsed_to_zero():
    """F3: sqrt(sxx*syy) overflows to inf on large-scale data, collapsing corr to sxy/inf==0.
    sqrt(sxx)*sqrt(syy) keeps a genuine correlation ~1."""
    x = (np.arange(1, 9, dtype=np.float64)) * 1e120
    y = 2.0 * x
    corr = fast_pearson_corr(x, y)
    assert corr > 0.999, f"large-scale corr collapsed to {corr} (product-overflow bug)"
    # sanity: normal-scale still correct
    xs = np.arange(1, 9, dtype=np.float64)
    assert fast_pearson_corr(xs, 2.0 * xs) > 0.999


def test_chao_shen_mi_tolerates_negative_codes():
    """F4: negative codes (e.g. NaN sentinels) made x_b*K_y+y_b negative and crashed np.bincount."""
    x_binned = np.array([-1, 0, 1, 0, 1, -1, 0], dtype=np.int64)
    y = np.array([0, 1, 0, 1, 0, 1, 0], dtype=np.float64)
    mi = chao_shen_mi(x_binned, y)  # must not raise
    assert np.isfinite(mi) and mi >= 0.0


def test_chao_shen_entropy_rare_category_large_n_not_dropped():
    """F5: 1-(1-p)**N cancels for small p*N (rare category, large N), silently dropping it via the
    <=1e-12 guard. The stable -expm1(N*log1p(-p)) form keeps the category's contribution."""
    n = 2_000_000
    x = np.zeros(n, dtype=np.int64)
    x[0] = 1  # a single-occurrence rare category among 2M
    h = chao_shen_entropy(x)
    # a genuine (rare) second category must lift entropy above the degenerate single-category 0.0
    assert h > 0.0, "rare category silently dropped (cancellation bug)"
