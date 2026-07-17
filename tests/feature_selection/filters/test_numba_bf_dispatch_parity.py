"""Regression: the numba_kernel optimizer's integer-dispatched binary functions must be numerically
IDENTICAL to the reference callables in hermite_fe._DEFAULT_BIN_FUNCS -- a formula divergence makes
winners found by the reference-scoring optimizers (cma_batch/optuna/...) unreachable by the kernel.
Found live (2026-07-15 budget-matched A/B): BF_LOGABS used sign(ab)*log1p(|ab|) instead of the
reference sign(ab+eps)*(log(|a|+eps)+log(|b|+eps)), and BF_DIV used a/(|b|+1e-9) instead of the
sign-preserving exact-divide _safe_div -- both made the kernel return None on the cubic-inner case
(mi=0.4813 for every other optimizer, planted-winner probe included)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._numba_polynom_optimizer import _BF_NAME_TO_ID, _bf_dispatch_njit
from mlframe.feature_selection.filters.hermite_fe import _DEFAULT_BIN_FUNCS


@pytest.mark.parametrize("bf_name", sorted(_BF_NAME_TO_ID))
def test_bf_dispatch_matches_reference_callable(bf_name):
    """The numba-dispatched bin-func kernel is bit-identical to its reference callable in _DEFAULT_BIN_FUNCS across adversarial zero/tiny/negative/large inputs, for every registered bf_name."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(4096)
    b = rng.standard_normal(4096)
    # Adversarial values: exact zeros, tiny magnitudes, negatives, large magnitudes.
    a[:8] = [0.0, -0.0, 1e-12, -1e-12, -5.0, 5.0, 0.3, -0.3]
    b[:8] = [0.0, 1e-12, 0.0, -3.0, -0.0, 1e-300, -0.7, 0.7]
    ref = np.asarray(_DEFAULT_BIN_FUNCS[bf_name](a, b), dtype=np.float64)
    got = np.asarray(_bf_dispatch_njit(_BF_NAME_TO_ID[bf_name], a, b), dtype=np.float64)
    np.testing.assert_allclose(got, ref, rtol=0, atol=0, err_msg=f"bf={bf_name} diverges from reference")


def test_numba_kernel_recovers_cubic_inner_winner():
    """End-to-end regression for the three-way parity fix (bf formulas + saturating L2): the kernel must
    recover the cubic-inner logabs winner every reference-scored optimizer finds (mi=0.4813). Pre-fix it
    returned None even with the winner planted into its warm seeds."""
    from mlframe.feature_selection.filters.hermite_fe import optimise_hermite_pair

    r = np.random.default_rng(7)
    a = r.standard_normal(20000)
    b = r.standard_normal(20000)
    y = ((a**3 - 2 * a) * b > 0).astype(np.int64)
    res = optimise_hermite_pair(
        x_a=a,
        x_b=b,
        y=y,
        seed=42,
        optimizer="numba_kernel",
        n_trials=100,
        min_degree=3,
        max_degree=6,
        coef_range=(-2.0, 2.0),
        l2_penalty=0.05,
        sweep_degrees=True,
        basis="chebyshev",
        mi_estimator="plugin",
        discrete_target=True,
        warm_start=True,
        multi_fidelity=False,
    )
    assert res is not None, "kernel must clear the baseline-uplift gate on the cubic-inner case"
    assert res.mi >= 0.45, f"expected ~0.4813 parity with the reference optimizers, got {res.mi:.4f}"


def test_cupy_kernel_recovers_cubic_inner_winner():
    """The GPU generation-batched optimizer must recover the same cubic-inner logabs winner as every
    reference-scored backend (selection-equivalence bar; cp.argsort tie-order is the only permitted
    divergence source and is measure-zero on continuous GEMM outputs)."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters.hermite_fe import optimise_hermite_pair

    r = np.random.default_rng(7)
    a = r.standard_normal(20000)
    b = r.standard_normal(20000)
    y = ((a**3 - 2 * a) * b > 0).astype(np.int64)
    res = optimise_hermite_pair(
        x_a=a,
        x_b=b,
        y=y,
        seed=42,
        optimizer="cupy_kernel",
        n_trials=100,
        min_degree=3,
        max_degree=6,
        coef_range=(-2.0, 2.0),
        l2_penalty=0.05,
        sweep_degrees=True,
        basis="chebyshev",
        mi_estimator="plugin",
        discrete_target=True,
        warm_start=True,
        multi_fidelity=False,
    )
    assert res is not None and res.mi >= 0.45, f"expected ~0.4813, got {None if res is None else res.mi}"
