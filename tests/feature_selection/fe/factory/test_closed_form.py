"""Closed-form shift estimators: agreement with a numpy lstsq reference, CUDA-arithmetic emulation, degenerate input and thread invariance."""

import numba
import numpy as np
from scipy.stats import rankdata

from mlframe.feature_selection._benchmarks.fe_operator_factory.kernel_prototypes._synthetic import make
from mlframe.feature_selection._benchmarks.fe_operator_factory.kernel_prototypes.offset_closed_form import (
    emulate_gram4_cuda,
    interaction_ols_shift,
    zero_cross_shift,
)

N = 4000


def _data():
    """Four (u, v) forms and the rank-scaled target of the case-2 data."""
    U, V, _, _ = make(N, K=4, G=4)
    rng = np.random.default_rng(0)
    c, d = rng.random(N), rng.random(N)
    y = np.log(2 * c) * np.sin(d / 3) + rng.random(N) / 5
    return U, V, rankdata(y) / N


def _ref_ols(u, v, yr):
    """Reference shift ``b_v / b_uv`` (centred) from a dense lstsq fit of ``y ~ [1, u, v, uv]``."""
    mu_u, mu_v = u.mean(), v.mean()
    x, w = u - mu_u, v - mu_v
    A = np.stack([np.ones(N), x, w, x * w], 1)
    co = np.linalg.lstsq(A, yr - yr.mean(), rcond=None)[0]
    return co[2] / co[3] - mu_u


def test_interaction_ols_matches_lstsq():
    """The one-pass 4x4 normal-equation estimator equals the dense lstsq shift to rounding."""
    U, V, yr = _data()
    t = interaction_ols_shift(U, V, yr)
    ref = np.array([_ref_ols(U[k], V[k], yr) for k in range(U.shape[0])])
    assert np.abs(t - ref).max() < 1e-8


def test_case2_shift_has_the_right_sign_and_scale():
    """On ``log(2c) * sin(d/3)`` the (log c, sin d) form needs a positive shift near ln 2 (loose bound: n is small here)."""
    U, V, yr = _data()
    assert abs(interaction_ols_shift(U, V, yr)[0] - np.log(2)) < 0.4


def test_cuda_arithmetic_emulation_matches_njit():
    """The numpy emulation of the CUDA summation order agrees with the njit kernel."""
    U, V, yr = _data()
    t = interaction_ols_shift(U, V, yr)
    assert np.abs(emulate_gram4_cuda(U[:2], V[:2], yr) - t[:2]).max() < 1e-9


def test_zero_cross_is_close_to_ols():
    """The zero-crossing estimate lands near the OLS one on these smooth forms."""
    U, V, yr = _data()
    assert np.abs(zero_cross_shift(U, V, yr, 10) - interaction_ols_shift(U, V, yr)).max() < 0.1


def test_singular_input_returns_nan():
    """A constant operand makes the normal equations singular; both estimators return NaN so the caller can fall back to t = 0."""
    U, _, yr = _data()
    Vc = np.ones((1, N))
    assert np.isnan(interaction_ols_shift(U[:1].copy(), Vc, yr)).all()
    assert np.isnan(zero_cross_shift(U[:1].copy(), Vc, yr, 10)).all()


def test_bit_identical_across_thread_counts():
    """Fixed row blocks plus Neumaier merge make both estimators independent of the thread count."""
    U, V, yr = _data()
    saved = numba.get_num_threads()
    two = min(2, saved)
    try:
        res = {}
        for nt in (1, two):
            numba.set_num_threads(nt)
            res[nt] = (interaction_ols_shift(U, V, yr), zero_cross_shift(U, V, yr, 10))
    finally:
        numba.set_num_threads(saved)
    assert all(np.array_equal(a, b, equal_nan=True) for a, b in zip(res[1], res[two]))
