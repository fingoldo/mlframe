"""The cross-target identity cache keys on the sample weights and is bounded (AUD23-49).

A fit that selected every column is cached as an identity verdict; a later fit on the same X, y and params skips its work. Weights change the relevance of every column, so a fit under other weights must not reuse
the verdict of an unweighted one. The process-wide dict was also unbounded.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fingerprints import _MRMR_IDENTITY_FP_CACHE, _MRMR_IDENTITY_FP_CACHE_MAX, _mrmr_identity_cache_key, identity_cache_put
from mlframe.feature_selection.filters.mrmr import MRMR


def _frame(n: int = 3000):
    """Three informative columns: the unweighted fit selects all of them (an identity result)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((n, 3)), columns=list("abc"))
    y = pd.Series(((X["a"] + X["b"] + X["c"] + 0.5 * rng.standard_normal(n)) > 0).astype(int))
    return X, y


def test_key_differs_for_other_weights_and_for_no_weights():
    """Same X, y and params: no weights, weights A and weights B give three keys; equal weights give the same key."""
    X, y = _frame()
    m = MRMR(verbose=0, random_seed=0, fe_max_steps=0)
    wa = np.where(X["a"].to_numpy() > 0, 1.0, 0.0)
    wb = np.where(X["b"].to_numpy() > 0, 1.0, 0.0)
    keys = {_mrmr_identity_cache_key(m, X, y), _mrmr_identity_cache_key(m, X, y, wa), _mrmr_identity_cache_key(m, X, y, wb)}
    assert len(keys) == 3
    assert _mrmr_identity_cache_key(m, X, y, wa) == _mrmr_identity_cache_key(m, X, y, wa.copy())


def test_a_weighted_fit_does_not_take_the_identity_shortcut_of_an_unweighted_one():
    """The second fit runs for real: the shortcut would set ``_identity_cache_ycorr_`` to a number."""
    X, y = _frame()
    _MRMR_IDENTITY_FP_CACHE.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = MRMR(verbose=0, random_seed=0, fe_max_steps=0).fit(X, y)
        assert len(first.support_) == 3, "the fixture must give an identity result for the test to mean anything"
        weighted = MRMR(verbose=0, random_seed=0, fe_max_steps=0).fit(X, y, sample_weight=np.where(X["a"].to_numpy() > 0, 1.0, 0.0))
    assert getattr(weighted, "_identity_cache_ycorr_", None) is None


def test_the_cache_drops_the_oldest_entries_beyond_its_bound():
    """Contrast with the old unbounded dict: the size stays at the bound and the newest entries survive."""
    cache: dict = {}
    for i in range(_MRMR_IDENTITY_FP_CACHE_MAX + 50):
        identity_cache_put(cache, f"k{i}", (True, None))
    assert len(cache) == _MRMR_IDENTITY_FP_CACHE_MAX
    assert "k0" not in cache and f"k{_MRMR_IDENTITY_FP_CACHE_MAX + 49}" in cache


def test_weight_key_digest_is_the_full_content_digest():
    """The weights part of the key hashes the full weight buffer, byte for byte as tobytes would (no second copy is made)."""
    import hashlib

    X, y = _frame()
    m = MRMR(verbose=0, random_seed=0, fe_max_steps=0)
    w = np.where(X["a"].to_numpy() > 0, 1.0, 0.0)
    _w = np.ascontiguousarray(np.asarray(w, dtype=np.float64).reshape(-1))
    expected = hashlib.blake2b(_w.tobytes(), digest_size=10).hexdigest()
    assert f"_wfp_{expected}" in _mrmr_identity_cache_key(m, X, y, sample_weight=w)
