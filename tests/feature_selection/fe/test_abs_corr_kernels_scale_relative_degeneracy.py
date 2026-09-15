"""The |corr| kernels judge near-constancy relative to a column's own scale, not against an absolute variance floor.

``va <= 1e-24 * n`` declared any column with std <= 1e-12 constant, so a genuinely tiny-scale column (values ~1e-13: a normalised residual,
a difference of two nearly-equal engineered columns) reported |corr| 0.0 against its own perfect correlate. That 0.0 reads as "not redundant"
in the dedup gate and "no signal" in the y-gate.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core import _abs_corr_finite_njit, _abs_corr_zerofill_njit


def _pair(scale):
    """A column and a noisy linear correlate, the first scaled by ``scale``."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    y = x + 0.05 * rng.normal(size=400)
    return x * scale, y


def test_masked_kernel_tiny_scale_column_keeps_its_correlation():
    """A 1e-13-scaled column correlates with y exactly as the unscaled one does."""
    a_tiny, y = _pair(1e-13)
    a_unit, _ = _pair(1.0)
    fin = np.ones(y.shape[0], dtype=np.bool_)
    expected = _abs_corr_finite_njit(a_unit, y, fin, 8)
    assert expected > 0.99
    got = _abs_corr_finite_njit(a_tiny, y, fin, 8)
    assert abs(got - expected) < 1e-9, f"tiny-scale column reported |corr| {got} instead of {expected}"


def test_zerofill_kernel_tiny_scale_column_keeps_its_correlation():
    """Same contract for the zero-fill twin."""
    a_tiny, y = _pair(1e-13)
    a_unit, _ = _pair(1.0)
    expected = _abs_corr_zerofill_njit(a_unit, y)
    got = _abs_corr_zerofill_njit(a_tiny, y)
    assert abs(got - expected) < 1e-9, f"tiny-scale column reported |corr| {got} instead of {expected}"


def test_constant_column_still_reports_zero():
    """Control: a truly constant column (at any offset) is still degenerate in both kernels."""
    _, y = _pair(1.0)
    fin = np.ones(y.shape[0], dtype=np.bool_)
    for c in (np.zeros(400), np.full(400, 7.0), np.full(400, 1e9)):
        assert _abs_corr_finite_njit(c, y, fin, 8) == 0.0
        assert _abs_corr_zerofill_njit(c, y) == 0.0
