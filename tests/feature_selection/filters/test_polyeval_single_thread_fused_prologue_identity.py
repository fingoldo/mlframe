"""Bit-identity regression for the fused-prologue single-thread polynomial evaluators.

hermite_fe single-thread _hermeval/_legval/_chebval/_lagval_njit had their prologue fused
(2026-06-24): the x.copy() and the two separate out[i]=c[0] / out[i]+=c[1]*P_1[i] passes were
collapsed into one pass (P_1==x is read-only for hermite/legendre/chebyshev; laguerre fuses the
1-x build with the c[1] accumulation). 1.13-1.39x faster single-thread, must stay bit-identical.

This test pins bit-identity vs. the LEGACY per-degree-allocation form (reproduced verbatim below
from the pre-fix source). It must FAIL on a regression that changes the numerics.
"""
from __future__ import annotations

import numpy as np
import pytest

njit = pytest.importorskip("numba").njit

from mlframe.feature_selection.filters.hermite_fe import (  # noqa: E402
    _hermeval_njit,
    _legval_njit,
    _chebval_njit,
    _lagval_njit,
)


# ---- Verbatim pre-fix legacy forms (the OLD side of the A/B) ----
@njit(cache=False, fastmath=True)
def _herm_old(x, c):
    n = x.shape[0]; out = np.zeros(n); nc = c.shape[0]
    if nc == 0:
        return out
    for i in range(n):
        out[i] = c[0]
    if nc == 1:
        return out
    p_prev = np.ones(n); p_curr = x.copy()
    for i in range(n):
        out[i] += c[1] * p_curr[i]
    for k in range(2, nc):
        p_next = np.empty(n); ck = c[k]; km1 = k - 1
        for i in range(n):
            p_next[i] = x[i] * p_curr[i] - km1 * p_prev[i]
            out[i] += ck * p_next[i]
        p_prev = p_curr; p_curr = p_next
    return out


@njit(cache=False, fastmath=True)
def _leg_old(x, c):
    n = x.shape[0]; out = np.zeros(n); nc = c.shape[0]
    if nc == 0:
        return out
    for i in range(n):
        out[i] = c[0]
    if nc == 1:
        return out
    p_prev = np.ones(n); p_curr = x.copy()
    for i in range(n):
        out[i] += c[1] * p_curr[i]
    for k in range(2, nc):
        p_next = np.empty(n); ck = c[k]; inv_k = 1.0 / k; two_km1 = 2 * k - 1; km1 = k - 1
        for i in range(n):
            p_next[i] = (two_km1 * x[i] * p_curr[i] - km1 * p_prev[i]) * inv_k
            out[i] += ck * p_next[i]
        p_prev = p_curr; p_curr = p_next
    return out


@njit(cache=False, fastmath=True)
def _cheb_old(x, c):
    n = x.shape[0]; out = np.zeros(n); nc = c.shape[0]
    if nc == 0:
        return out
    for i in range(n):
        out[i] = c[0]
    if nc == 1:
        return out
    p_prev = np.ones(n); p_curr = x.copy()
    for i in range(n):
        out[i] += c[1] * p_curr[i]
    for k in range(2, nc):
        p_next = np.empty(n); ck = c[k]
        for i in range(n):
            p_next[i] = 2.0 * x[i] * p_curr[i] - p_prev[i]
            out[i] += ck * p_next[i]
        p_prev = p_curr; p_curr = p_next
    return out


@njit(cache=False, fastmath=True)
def _lag_old(x, c):
    n = x.shape[0]; out = np.zeros(n); nc = c.shape[0]
    if nc == 0:
        return out
    for i in range(n):
        out[i] = c[0]
    if nc == 1:
        return out
    p_prev = np.ones(n); p_curr = np.empty(n)
    for i in range(n):
        p_curr[i] = 1.0 - x[i]
        out[i] += c[1] * p_curr[i]
    for k in range(2, nc):
        p_next = np.empty(n); ck = c[k]; inv_k = 1.0 / k; two_km1 = 2 * k - 1; km1 = k - 1
        for i in range(n):
            p_next[i] = ((two_km1 - x[i]) * p_curr[i] - km1 * p_prev[i]) * inv_k
            out[i] += ck * p_next[i]
        p_prev = p_curr; p_curr = p_next
    return out


_CASES = [
    ("hermite", _hermeval_njit, _herm_old),
    ("legendre", _legval_njit, _leg_old),
    ("chebyshev", _chebval_njit, _cheb_old),
    ("laguerre", _lagval_njit, _lag_old),
]


@pytest.mark.parametrize("name,new_fn,old_fn", _CASES)
@pytest.mark.parametrize("nc", [1, 2, 3, 5])
@pytest.mark.parametrize("n", [1, 500, 2000, 10000])
def test_fused_prologue_bit_identical(name, new_fn, old_fn, nc, n):
    rng = np.random.default_rng(hash((name, nc, n)) & 0xFFFF)
    x = rng.standard_normal(n)
    c = rng.standard_normal(nc)
    new = new_fn(x, c)
    old = old_fn(x, c)
    assert np.array_equal(new, old), f"{name} nc={nc} n={n} diverged: max|d|={np.max(np.abs(new-old))}"


def test_edge_empty_coef():
    x = np.array([1.0, 2.0, 3.0])
    for _, new_fn, old_fn in _CASES:
        assert np.array_equal(new_fn(x, np.empty(0)), old_fn(x, np.empty(0)))
