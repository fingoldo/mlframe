"""Regression test: crps_from_quantiles fused per-alpha pinball is bit-identical
to the prior per-column loop.

The per-alpha mean pinball vector inside ``crps_from_quantiles`` was built by a
Python loop calling ``pinball_loss(y, p[:, k], a[k])`` K times; it now uses the
fused ``_fast_pinball_per_alpha`` matrix kernel (2.2x-6.9x faster, N in {10k..1M}).
This locks in bit-identity so the perf change cannot silently alter CRPS values.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.quantile import (
    crps_from_quantiles,
    pinball_loss,
    _fast_pinball_per_alpha,
)


def _crps_loop_reference(y, p, a):
    """Explicit re-implementation of the ORIGINAL per-column loop path."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    per_alpha = np.empty(a.shape[0], dtype=np.float64)
    for k in range(a.shape[0]):
        per_alpha[k] = pinball_loss(y, p[:, k], float(a[k]))
    integral = float(np.sum((a[1:] - a[:-1]) * (per_alpha[1:] + per_alpha[:-1]) * 0.5))
    if a[0] > 0.0:
        pin_lo_edge = pinball_loss(y, p[:, 0], 0.0)
        integral += float(0.5 * a[0] * (pin_lo_edge + per_alpha[0]))
    if a[-1] < 1.0:
        pin_hi_edge = pinball_loss(y, p[:, -1], 1.0)
        integral += float(0.5 * (1.0 - a[-1]) * (per_alpha[-1] + pin_hi_edge))
    return 2.0 * integral


@pytest.mark.parametrize("n", [1, 100, 5_000])
@pytest.mark.parametrize("k", [2, 10, 19])
def test_crps_fused_bit_identical_to_loop(n, k):
    """Crps fused bit identical to loop."""
    rng = np.random.default_rng(n * 100 + k)
    y = rng.standard_normal(n)
    p = np.sort(rng.standard_normal((n, k)), axis=1)
    a = np.linspace(0.05, 0.95, k)
    got = crps_from_quantiles(y, p, a)
    ref = _crps_loop_reference(y, p, a)
    # BIT-identical: same per-element accumulation order per column.
    assert got == ref, f"CRPS drifted: fused={got!r} loop={ref!r}"


def test_fused_per_alpha_matches_pinball_loss_per_column():
    """Fused per alpha matches pinball loss per column."""
    rng = np.random.default_rng(7)
    n, k = 2_000, 9
    y = np.ascontiguousarray(rng.standard_normal(n))
    p = np.ascontiguousarray(np.sort(rng.standard_normal((n, k)), axis=1))
    a = np.ascontiguousarray(np.linspace(0.1, 0.9, k))
    fused = _fast_pinball_per_alpha(y, p, a)
    for j in range(k):
        assert fused[j] == pinball_loss(y, p[:, j], float(a[j]))


def test_crps_non_contiguous_inputs_still_identical():
    # F-ordered / strided inputs must be handled by the ascontiguousarray casts.
    """Crps non contiguous inputs still identical."""
    rng = np.random.default_rng(11)
    n, k = 500, 10
    y = rng.standard_normal(n * 2)[::2]  # strided
    p = np.asfortranarray(np.sort(rng.standard_normal((n, k)), axis=1))
    a = np.linspace(0.05, 0.95, k)
    got = crps_from_quantiles(y, p, a)
    ref = _crps_loop_reference(y, p, a)
    assert got == ref
