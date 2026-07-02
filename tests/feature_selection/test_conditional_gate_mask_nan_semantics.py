"""Pins the off-region NaN semantics of the conditional-gate mask feature block.

``_build_feats`` (mask mode) in ``_conditional_gate_fe`` builds ``feats[:, j] = av * (cv > tau)``.
This MUST propagate NaN from ``av`` in the gated-OFF region exactly like the prior
``(cv > tau).astype(float) * av`` (``0.0 * NaN == NaN``). A ``np.where(cv > tau, av, 0.0)``
"simplification" would silently zero those NaNs and change downstream MI binning, so the
sensor also asserts that wrong form diverges.
"""
import numpy as np


def _old(cv, av, taus):
    f = np.empty((cv.shape[0], len(taus)))
    for j, t in enumerate(taus):
        f[:, j] = (cv > t).astype(np.float64) * av
    return f


def _new(cv, av, taus):
    f = np.empty((cv.shape[0], len(taus)))
    for j, t in enumerate(taus):
        np.multiply(av, cv > t, out=f[:, j])
    return f


def _wrong_where(cv, av, taus):
    f = np.empty((cv.shape[0], len(taus)))
    for j, t in enumerate(taus):
        f[:, j] = np.where(cv > t, av, 0.0)
    return f


def test_mask_block_nan_identical_to_prior_astype_form():
    rng = np.random.default_rng(0)
    cv = rng.standard_normal(2000)
    av = rng.standard_normal(2000)
    av[::7] = np.nan
    cv[::11] = np.nan
    taus = np.round(np.linspace(0.1, 0.9, 17), 4)
    assert np.array_equal(_old(cv, av, taus), _new(cv, av, taus), equal_nan=True)


def test_np_where_form_would_drop_off_region_nan():
    rng = np.random.default_rng(1)
    cv = rng.standard_normal(2000)
    av = rng.standard_normal(2000)
    av[::7] = np.nan
    taus = np.round(np.linspace(0.1, 0.9, 17), 4)
    assert not np.array_equal(_old(cv, av, taus), _wrong_where(cv, av, taus), equal_nan=True)


def test_njit_grid_kernels_bit_identical_to_numpy_incl_nan():
    """The fused njit build kernels (gated ON above _GATE_BUILD_NJIT_MIN_N) must match the numpy per-tau forms
    exactly, including NaN operands -- the mask off-value ``a*0.0`` preserves the ``av*(cv>tau)`` off-region NaN."""
    from mlframe.feature_selection.filters._conditional_gate_fe import (
        _gate_mask_grid_njit,
        _gate_select_grid_njit,
    )
    rng = np.random.default_rng(2)
    n = 25000
    cv = rng.standard_normal(n)
    av = rng.standard_normal(n)
    bv = rng.standard_normal(n)
    cv[::11] = np.nan
    av[::13] = np.nan
    bv[::17] = np.nan
    taus = np.round(np.linspace(0.1, 0.9, 17), 4)

    mask_np = _old(cv, av, taus)
    mask_jit = _gate_mask_grid_njit(
        np.ascontiguousarray(cv), np.ascontiguousarray(av), np.ascontiguousarray(taus)
    )
    assert np.array_equal(mask_np, mask_jit, equal_nan=True)

    sel_np = np.empty((n, len(taus)))
    for j, t in enumerate(taus):
        sel_np[:, j] = np.where(cv > t, av, bv)
    sel_jit = _gate_select_grid_njit(
        np.ascontiguousarray(cv), np.ascontiguousarray(av), np.ascontiguousarray(bv), np.ascontiguousarray(taus)
    )
    assert np.array_equal(sel_np, sel_jit, equal_nan=True)
