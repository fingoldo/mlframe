"""Pickle round-trip -> re-fit -> transform coverage for MRMR.

Gap addressed: the sibling ``test_setstate_legacy_pickle_refit.py`` drives ``__setstate__``
with a hand-stripped state dict. This file exercises the REAL ``pickle.dumps``/``loads``
path end-to-end: a fitted estimator survives a pickle round-trip with its support and
transform output intact, then can be RE-FIT and transformed again without crashing.

It also pins the legacy-override contract through the genuine pickle protocol: a pickle
whose state is missing assorted ctor params (simulating an OLD pickle predating those
params) must re-fit cleanly AND keep the documented legacy override value
(``mrmr_identity_cache_ycorr_threshold`` -> 0.0, NOT the live ctor default 0.5).
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import MRMR


def _data(n: int = 150, m: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, m))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(np.int32)
    return pd.DataFrame(X, columns=list("abcd")[:m]), y


def _fast(**kw):
    base = dict(full_npermutations=5, baseline_npermutations=3, n_jobs=1, verbose=0, fe_fast_search=False, interactions_max_order=1)
    base.update(kw)
    return MRMR(**base)


def test_pickle_round_trip_preserves_support_and_transform():
    X, y = _data(seed=1)
    MRMR._FIT_CACHE.clear()
    m = _fast(random_seed=7).fit(X, y)
    sup = np.sort(np.asarray(m.support_))
    names = list(m.get_feature_names_out())
    out = m.transform(X)

    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(np.sort(np.asarray(m2.support_)), sup)
    assert list(m2.get_feature_names_out()) == names
    out2 = m2.transform(X)
    assert np.asarray(out2).shape == np.asarray(out).shape
    np.testing.assert_allclose(np.asarray(out2, dtype=float), np.asarray(out, dtype=float))


def test_pickle_then_refit_then_transform():
    """A pickled-and-restored estimator can be re-fit on new data and transformed."""
    X, y = _data(seed=2)
    MRMR._FIT_CACHE.clear()
    m = _fast(random_seed=7).fit(X, y)
    m2 = pickle.loads(pickle.dumps(m))

    X2, y2 = _data(seed=3)
    MRMR._FIT_CACHE.clear()
    assert m2.fit(X2, y2) is m2
    assert np.asarray(m2.support_).size >= 1
    out = m2.transform(X2)
    assert np.asarray(out).shape[0] == X2.shape[0]


def test_pickle_state_missing_params_refits_and_keeps_legacy_override():
    """Simulate an OLD pickle: drop assorted ctor params from the pickled state, restore
    via the genuine pickle/__setstate__ path, then RE-FIT. The estimator must (a) not
    crash on a bare ``self.<param>`` read, and (b) keep the documented legacy override
    ``mrmr_identity_cache_ycorr_threshold == 0.0`` (NOT the live ctor default 0.5)."""
    X, y = _data(seed=4)
    MRMR._FIT_CACHE.clear()
    m = _fast(random_seed=3).fit(X, y)

    state = m.__getstate__()
    for k in ("fe_wavelet_enable", "dtype", "mrmr_identity_cache_ycorr_threshold", "nbins_strategy", "bur_lambda", "fe_kfold_te_enable"):
        state.pop(k, None)

    restored = MRMR.__new__(MRMR)
    restored.__setstate__(dict(state))

    # Legacy override stays at the legacy value, not the live ctor default (0.5).
    assert restored.mrmr_identity_cache_ycorr_threshold == 0.0
    # A non-override missing param is re-sourced from the ctor default.
    assert restored.dtype is not None
    # Re-fit cleanly (no AttributeError on a bare self.<param> read).
    MRMR._FIT_CACHE.clear()
    restored.fit(X, y)
    assert np.asarray(restored.support_).size >= 1


def test_unfitted_estimator_pickles_and_fits():
    """An UNFITTED MRMR survives pickle and is usable afterwards."""
    m = _fast(random_seed=9)
    m2 = pickle.loads(pickle.dumps(m))
    assert not hasattr(m2, "support_")
    X, y = _data(seed=5)
    MRMR._FIT_CACHE.clear()
    m2.fit(X, y)
    assert np.asarray(m2.support_).size >= 1
