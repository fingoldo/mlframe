"""Core-selection coverage: sample_weight handling in MRMR.fit.

Contracts (``_maybe_resample_for_sample_weight``):
- None / uniform weights -> byte-identical to the no-weight path (same support_); the cache may reuse the fit.
- 1-D requirement, length must match n_rows, must be finite and non-negative, must not sum to zero -> ValueError.
- Non-uniform weights trigger a reproducible weighted resample: same seed + same weights -> same support_.
- Heavily up-weighting a subpopulation in which a DIFFERENT feature drives y shifts the selection toward that
  feature (the weighting is functionally effective, not a no-op).

These input-validation + functional-effect contracts had no test under mrmr_api/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


def _no_fe(**kw):
    base = dict(
        random_seed=0,
        verbose=0,
        fe_max_steps=0,
        interactions_max_order=1,
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        cat_fe_config=None,
        fe_hinge_enable=False,
        fe_modular_enable=False,
        fe_pairwise_modular_enable=False,
        fe_integer_lattice_enable=False,
        fe_row_argmax_enable=False,
        fe_conditional_gate_enable=False,
    )
    base.update(kw)
    return MRMR(**base)


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    X = pd.DataFrame({"x0": x0, "x1": x1, "noise": rng.normal(size=n)})
    y = (x0 + 0.5 * x1 > 0).astype(int)
    return X, y


def _sup(m):
    return np.sort(np.asarray(m.support_, dtype=np.intp))


def test_uniform_weight_equals_no_weight():
    """All-equal sample_weight is a no-op -> identical support_ to the unweighted fit."""
    X, y = _data()
    n = len(y)
    MRMR._FIT_CACHE.clear()
    base = _no_fe().fit(X, y)
    MRMR._FIT_CACHE.clear()
    weighted = _no_fe().fit(X, y, sample_weight=np.full(n, 3.7))
    np.testing.assert_array_equal(_sup(base), _sup(weighted))


def test_none_weight_equals_no_weight():
    X, y = _data()
    MRMR._FIT_CACHE.clear()
    a = _no_fe().fit(X, y)
    MRMR._FIT_CACHE.clear()
    b = _no_fe().fit(X, y, sample_weight=None)
    np.testing.assert_array_equal(_sup(a), _sup(b))


def test_nonuniform_weight_reproducible_same_seed():
    """Non-uniform weights + same seed -> identical support_ (deterministic resample)."""
    X, y = _data()
    rng = np.random.default_rng(123)
    w = rng.random(len(y)) + 0.1
    MRMR._FIT_CACHE.clear()
    a = _no_fe(random_seed=4).fit(X, y, sample_weight=w)
    MRMR._FIT_CACHE.clear()
    b = _no_fe(random_seed=4).fit(X, y, sample_weight=w.copy())
    np.testing.assert_array_equal(_sup(a), _sup(b))


def test_weight_length_mismatch_raises():
    X, y = _data()
    with pytest.raises(ValueError):
        _no_fe().fit(X, y, sample_weight=np.ones(len(y) + 1))


def test_weight_negative_raises():
    X, y = _data()
    w = np.ones(len(y))
    w[0] = -1.0
    with pytest.raises(ValueError):
        _no_fe().fit(X, y, sample_weight=w)


def test_weight_nonfinite_raises():
    X, y = _data()
    w = np.ones(len(y))
    w[0] = np.nan
    with pytest.raises(ValueError):
        _no_fe().fit(X, y, sample_weight=w)


def test_weight_sums_to_zero_raises():
    X, y = _data()
    with pytest.raises(ValueError):
        _no_fe().fit(X, y, sample_weight=np.zeros(len(y)))


def test_weight_2d_raises():
    X, y = _data()
    with pytest.raises(ValueError):
        _no_fe().fit(X, y, sample_weight=np.ones((len(y), 1)))


def test_subpopulation_weighting_shifts_selection():
    """Two disjoint subpopulations, each driven by a different feature; up-weighting one makes ITS driver win.

    Region A (first half): y = sign(xa). Region B (second half): y = sign(xb). With weight concentrated on
    region A, xa must be selected; concentrating on region B selects xb. Proves sample_weight is effective.
    """
    rng = np.random.default_rng(7)
    n = 1200
    half = n // 2
    xa = rng.normal(size=n)
    xb = rng.normal(size=n)
    region = np.zeros(n, dtype=int)
    region[half:] = 1
    y = np.where(region == 0, (xa > 0), (xb > 0)).astype(int)
    X = pd.DataFrame({"xa": xa, "xb": xb, "noise": rng.normal(size=n)})

    w_a = np.where(region == 0, 100.0, 1.0)
    w_b = np.where(region == 1, 100.0, 1.0)

    MRMR._FIT_CACHE.clear()
    ma = _no_fe(random_seed=1).fit(X, y, sample_weight=w_a)
    sel_a = {str(X.columns[i]) for i in _sup(ma)}
    MRMR._FIT_CACHE.clear()
    mb = _no_fe(random_seed=1).fit(X, y, sample_weight=w_b)
    sel_b = {str(X.columns[i]) for i in _sup(mb)}

    assert "xa" in sel_a, f"weighting region A should select its driver xa; got {sel_a}"
    assert "xb" in sel_b, f"weighting region B should select its driver xb; got {sel_b}"
    # The selections must differ -- otherwise the weighting had no effect.
    assert sel_a != sel_b
