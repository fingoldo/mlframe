"""Hypothesis property tests for MI bounds + DPI invariants in cat-FE.

Tier 1.4: catches regressions in info_theory kernels by checking
information-theoretic invariants on random inputs.

Reference: Cover & Thomas, "Elements of Information Theory" (2nd ed.):
- Theorem 2.6.4: ``I(X; Y) <= min(H(X), H(Y))``
- Theorem 2.6.5: ``H(X) <= log(|X|)``
- Theorem 2.8.1 (DPI): ``I(X1, X2; Y) >= max(I(X1; Y), I(X2; Y))``
- MI is invariant under bijective relabeling of either alphabet
  (i.e. ``I(merge_vars(X1, X2); Y) == I([X1, X2]; Y)``)
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from mlframe.feature_selection.filters.info_theory import (
    compute_mi_from_classes,
    merge_vars,
)


# Hypothesis-flake budget: use bounded n / cardinality + fixed deadline
HYP_SETTINGS = settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


@HYP_SETTINGS
@given(
    n=st.integers(min_value=200, max_value=1500),
    k_x=st.integers(min_value=2, max_value=8),
    k_y=st.integers(min_value=2, max_value=6),
    seed=st.integers(min_value=0, max_value=10_000),
)
@pytest.mark.fast
def test_mi_upper_bound_log_min_alphabet(n, k_x, k_y, seed):
    """Cover-Thomas Thm 2.6.4: ``I(X;Y) <= min(log(|X|), log(|Y|))``.

    Catches regressions in entropy / MI computation that could produce
    impossible MI values (e.g. negative joint counts, missing log).
    """
    rng = np.random.default_rng(seed)
    x = rng.integers(0, k_x, n).astype(np.int32)
    y = rng.integers(0, k_y, n).astype(np.int32)
    data = np.column_stack([x, y]).astype(np.int32)
    nbins = np.array([k_x, k_y], dtype=np.int64)
    cls_x, fq_x, _ = merge_vars(
        factors_data=data,
        vars_indices=np.array([0], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=nbins,
        dtype=np.int32,
    )
    cls_y, fq_y, _ = merge_vars(
        factors_data=data,
        vars_indices=np.array([1], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=nbins,
        dtype=np.int32,
    )
    mi = compute_mi_from_classes(
        classes_x=cls_x,
        freqs_x=fq_x,
        classes_y=cls_y,
        freqs_y=fq_y,
        dtype=np.int32,
    )
    bound = min(np.log(k_x), np.log(k_y)) + 1e-6
    assert mi <= bound, f"MI={mi:.4f} exceeded Cover-Thomas bound {bound:.4f} (n={n}, k_x={k_x}, k_y={k_y}, seed={seed})"
    assert mi >= -1e-9, f"MI={mi} cannot be negative"


@HYP_SETTINGS
@given(
    n=st.integers(min_value=200, max_value=1500),
    k_a=st.integers(min_value=2, max_value=5),
    k_b=st.integers(min_value=2, max_value=5),
    k_y=st.integers(min_value=2, max_value=4),
    seed=st.integers(min_value=0, max_value=10_000),
)
@pytest.mark.fast
def test_dpi_joint_mi_dominates_marginals(n, k_a, k_b, k_y, seed):
    """Cover-Thomas Thm 2.8.1 (Data Processing Inequality):
    ``I(X1, X2; Y) >= max(I(X1; Y), I(X2; Y))``.

    The joint can never carry LESS information about Y than either
    marginal alone (since marginals are deterministic functions of
    the joint). Catches off-by-sign / formula errors in MI kernels.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(0, k_a, n).astype(np.int32)
    b = rng.integers(0, k_b, n).astype(np.int32)
    y = rng.integers(0, k_y, n).astype(np.int32)
    data = np.column_stack([a, b, y]).astype(np.int32)
    nbins = np.array([k_a, k_b, k_y], dtype=np.int64)
    cls_y, fq_y, _ = merge_vars(
        factors_data=data,
        vars_indices=np.array([2], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=nbins,
        dtype=np.int32,
    )
    # Marginals
    cls_a, fq_a, _ = merge_vars(
        factors_data=data,
        vars_indices=np.array([0], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=nbins,
        dtype=np.int32,
    )
    cls_b, fq_b, _ = merge_vars(
        factors_data=data,
        vars_indices=np.array([1], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=nbins,
        dtype=np.int32,
    )
    # Joint
    cls_ab, fq_ab, _ = merge_vars(
        factors_data=data,
        vars_indices=np.array([0, 1], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=nbins,
        dtype=np.int32,
    )
    mi_a_y = compute_mi_from_classes(cls_a, fq_a, cls_y, fq_y, np.int32)
    mi_b_y = compute_mi_from_classes(cls_b, fq_b, cls_y, fq_y, np.int32)
    mi_ab_y = compute_mi_from_classes(cls_ab, fq_ab, cls_y, fq_y, np.int32)
    tol = 1e-6
    assert mi_ab_y >= mi_a_y - tol, f"I(A,B;Y)={mi_ab_y:.6f} < I(A;Y)={mi_a_y:.6f} violates DPI (n={n}, k_a={k_a}, seed={seed})"
    assert mi_ab_y >= mi_b_y - tol, f"I(A,B;Y)={mi_ab_y:.6f} < I(B;Y)={mi_b_y:.6f} violates DPI (n={n}, k_b={k_b}, seed={seed})"


@HYP_SETTINGS
@given(
    n=st.integers(min_value=200, max_value=1000),
    k=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=10_000),
)
@pytest.mark.fast
def test_mi_self_equals_entropy(n, k, seed):
    """``I(X; X) = H(X)`` -- a column carries all its own information.

    Verifies merge_vars + compute_mi_from_classes are consistent with
    the entropy primitive. Catches off-by-log-base errors and
    normalisation mistakes.
    """
    from mlframe.feature_selection.filters.info_theory import entropy

    rng = np.random.default_rng(seed)
    x = rng.integers(0, k, n).astype(np.int32)
    data = np.column_stack([x, x.copy()]).astype(np.int32)
    nbins = np.array([k, k], dtype=np.int64)
    cls_x, fq_x, _ = merge_vars(
        factors_data=data,
        vars_indices=np.array([0], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=nbins,
        dtype=np.int32,
    )
    h_x = entropy(fq_x)
    mi_xx = compute_mi_from_classes(
        classes_x=cls_x,
        freqs_x=fq_x,
        classes_y=cls_x,
        freqs_y=fq_x,
        dtype=np.int32,
    )
    assert abs(mi_xx - h_x) < 1e-6, f"I(X;X)={mi_xx:.6f} != H(X)={h_x:.6f} (n={n}, k={k}, seed={seed})"


@HYP_SETTINGS
@given(
    n=st.integers(min_value=200, max_value=800),
    k_a=st.integers(min_value=2, max_value=4),
    k_b=st.integers(min_value=2, max_value=4),
    seed=st.integers(min_value=0, max_value=10_000),
)
@pytest.mark.fast
def test_independent_pair_mi_near_zero(n, k_a, k_b, seed):
    """When X1 and X2 are drawn independently with the SAME generator
    (no shared randomness with Y), the joint MI(X1, X2; X1) should
    equal H(X1) (X2 adds nothing), and MI(X1, X2; rng.uniform()-y)
    converges to 0 as n -> infinity.

    Loose threshold to absorb finite-sample noise.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(0, k_a, n).astype(np.int32)
    b = rng.integers(0, k_b, n).astype(np.int32)
    y_indep = rng.integers(0, 2, n).astype(np.int32)
    data = np.column_stack([a, b, y_indep]).astype(np.int32)
    nbins = np.array([k_a, k_b, 2], dtype=np.int64)
    cls_y, fq_y, _ = merge_vars(
        factors_data=data,
        vars_indices=np.array([2], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=nbins,
        dtype=np.int32,
    )
    cls_ab, fq_ab, _ = merge_vars(
        factors_data=data,
        vars_indices=np.array([0, 1], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=nbins,
        dtype=np.int32,
    )
    mi = compute_mi_from_classes(cls_ab, fq_ab, cls_y, fq_y, np.int32)
    # Finite-sample bias on small-n random data: ~0.05 nat typical
    # at n=200, k_a*k_b=16 cells.
    assert mi < 0.1, f"Independent (A, B; Y) joint MI={mi:.4f} should be small (n={n}, k_a={k_a}, k_b={k_b}, seed={seed})"
