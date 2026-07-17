"""Identity / equivalence pin for the angle-addition recurrence in
``bases._fourier_eval_njit``.

The recurrence form (sin/cos of the base angle 2*pi*z computed once per
sample, harmonics k=2..K stepped via the angle-addition identity) must
reproduce the direct ``math.sin(2*pi*k*z)`` / ``math.cos(2*pi*k*z)``
summation to ~1 ULP. fastmath is on for this kernel, so the contract is
numerical equivalence (max-abs-diff <= 1e-12), NOT bit-identity --
amply below any FE selection-altering threshold.

This test FAILS on the pre-recurrence kernel only if it regresses
numerically; its real job is to lock the recurrence's correctness so a
future "simplify back to the per-harmonic trig loop" or a buggy
recurrence edit is caught.
"""

import math

import numpy as np
import pytest

from mlframe.feature_selection.filters.bases import _fourier_eval_njit


def _fourier_eval_direct(z, c):
    """Reference: direct per-harmonic math.sin/cos summation (the pre-recurrence body)."""
    n = z.shape[0]
    out = np.zeros(n, dtype=np.float64)
    K = c.shape[0] // 2
    if K == 0:
        return out
    two_pi = 2.0 * math.pi
    for i in range(n):
        zi = z[i]
        s = 0.0
        for k in range(1, K + 1):
            ang = two_pi * k * zi
            s += c[2 * (k - 1)] * math.sin(ang)
            s += c[2 * (k - 1) + 1] * math.cos(ang)
        out[i] = s
    return out


@pytest.mark.parametrize("K", [0, 1, 2, 3, 5, 8])
@pytest.mark.parametrize("n", [1, 37, 1000])
def test_fourier_recurrence_matches_direct(n, K):
    rng = np.random.default_rng(n * 1000 + K)
    z = rng.random(n)  # z in [0, 1) -- the canonical Fourier-fit domain
    c = rng.standard_normal(2 * K) if K > 0 else np.zeros(0, dtype=np.float64)
    ref = _fourier_eval_direct(z, c)
    got = _fourier_eval_njit(z, c)
    assert got.shape == ref.shape
    assert np.max(np.abs(got - ref)) <= 1e-12, f"K={K} n={n}: max diff {np.max(np.abs(got - ref)):.2e}"


def test_fourier_recurrence_handles_extremes():
    """z at domain edges + a high harmonic -- recurrence must still track the direct form."""
    z = np.array([0.0, 0.25, 0.5, 0.75, 0.999999], dtype=np.float64)
    c = np.array([1.0, -2.0, 0.5, 0.3, -0.7, 1.1, 0.2, -0.4], dtype=np.float64)  # K=4
    ref = _fourier_eval_direct(z, c)
    got = _fourier_eval_njit(z, c)
    assert np.max(np.abs(got - ref)) <= 1e-12
