"""Regression tests for numerical-stability audit fixes (audit2_numstab).

Each asserts the fixed behavior on the input regime that triggered the pre-fix bug:
cubic-mean negative-base NaN, Pearson product-overflow collapse, and chao-shen
bincount crash / rare-category cancellation.
"""
import math

import numpy as np
import pytest

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
