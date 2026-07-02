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
