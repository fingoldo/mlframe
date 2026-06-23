"""Identity regression for the _mise_optimal_bandwidth h-grid loop hoist.

The MISE bandwidth grid search (``fastmi(..., bandwidth="mise")`` default path)
was rewritten to hoist the loop-invariant row-max / logsumexp shift out of the
per-bandwidth loop. The function's ONLY output is the selected bandwidth
``best_h``; the downstream ``fastmi`` MI is a pure function of it, so pinning
``best_h`` bit-identical to the reference math pins the whole estimator.

This reference reimplements the PRE-hoist arithmetic inline (full ``log_k``
build + ``log_k.max(axis=1)`` + separate ``exp``) and asserts the live
production function returns the exact same float64 across seeds + correlation
regimes (incl. the unsafe-looking near-independent / strong-dependence cases).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from mlframe.feature_selection.filters._fastmi import _mise_optimal_bandwidth


def _reference_pre_hoist(zx, zy, n_grid=12, h_min_factor=0.2, h_max_factor=1.5):
    """Verbatim pre-optimization arithmetic (full log_k + max + exp per h)."""
    n = zx.size
    h_sil = float(1.0 * (n ** (-1.0 / 6.0)))
    h_grid = np.linspace(h_sil * h_min_factor, h_sil * h_max_factor, n_grid)
    sp = (zx[:, None] - zx[None, :]) ** 2 + (zy[:, None] - zy[None, :]) ** 2
    np.fill_diagonal(sp, np.inf)
    best_h = h_sil
    best_ll = -np.inf
    for h in h_grid:
        log_k = -0.5 * sp / (h * h) - math.log(2.0 * math.pi * h * h)
        m = log_k.max(axis=1)
        f_i = m + np.log(np.exp(log_k - m[:, None]).sum(axis=1))
        f_i = f_i - math.log(n - 1)
        ll = float(np.sum(f_i))
        if ll > best_ll:
            best_ll = ll
            best_h = float(h)
    return best_h


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.95])
def test_mise_bandwidth_bit_identical_to_pre_hoist(seed, rho):
    rng = np.random.default_rng(seed)
    n = 400
    zx = rng.standard_normal(n)
    zy = rho * zx + math.sqrt(max(0.0, 1.0 - rho * rho)) * rng.standard_normal(n)
    expected = _reference_pre_hoist(zx, zy)
    got = _mise_optimal_bandwidth(zx, zy)
    # Bit-identical: the hoist only reorders loop-invariant terms; the selected
    # bandwidth must match exactly (not merely approximately).
    assert got == expected, f"seed={seed} rho={rho}: {got!r} != {expected!r}"
