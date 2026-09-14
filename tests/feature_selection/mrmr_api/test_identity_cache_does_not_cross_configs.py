"""The cross-target identity cache must be keyed on the selector's configuration too (mrmr_audit_2026-09-14 PERIPHERY-3).

With ``mrmr_skip_when_prior_was_identity`` on, a fit whose selection was the identity stores an entry keyed by an X fingerprint, and a
later fit on the same X takes ``_fit_identity_shortcut`` instead of fitting. The key covered X (and optionally y) but no constructor
parameter, so a permissive config's identity result licensed a different config -- one that would have scored and dropped columns --
to skip its fit and "select" everything. The in-object signature and ``_FIT_CACHE`` layers already fold the params signature; this one did not.

The test stores the entry through a real fit, then marks that entry as identity so the cache is primed exactly the way a permissive fit
leaves it, independent of whether this small fixture happens to select every column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR

_SHORTCUT = "_mrmr_identity_shortcut"


def _xy(n=300, seed=0):
    """Four columns with signal on 'a'."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.normal(size=n) for c in "abcd"})
    y = ((X["a"].to_numpy() + 0.1 * rng.normal(size=n)) > 0).astype(int)
    return X, y


def _fit(cache: dict, X, y, **params):
    """Fit a fresh instance sharing ``cache`` as its identity cache, with the y-correlation gate off."""
    m = MRMR(mrmr_skip_when_prior_was_identity=True, mrmr_identity_cache_include_y=False, mrmr_identity_cache_ycorr_threshold=0.0, max_runtime_mins=1.0, **params)
    m._mlframe_identity_cache_override_ = cache
    m.fit(X, y)
    return m


def _prime_cache_as_identity(cache: dict, X, y, **params) -> None:
    """Store an entry through a real fit, then mark every stored entry as an identity result."""
    _fit(cache, X, y, **params)
    assert cache, "the priming fit stored no identity-cache entry; the assertions below would observe nothing"
    for key in list(cache):
        cache[key] = True


def test_identity_entry_from_one_config_does_not_shortcut_another():
    """A different ``quantization_nbins`` must not reuse the entry: it gets a real fit, not the shortcut."""
    X, y = _xy()
    cache: dict = {}
    _prime_cache_as_identity(cache, X, y, quantization_nbins=10)
    other = _fit(cache, X, y, quantization_nbins=4)
    assert not str(other.signature).startswith(_SHORTCUT), "an identity result cached under one config short-circuited a fit under a different config"


def test_identity_entry_still_shortcuts_the_same_config():
    """Control: a fresh instance with the SAME config still takes the shortcut, so the cache itself keeps working."""
    X, y = _xy()
    cache: dict = {}
    _prime_cache_as_identity(cache, X, y, quantization_nbins=10)
    same = _fit(cache, X, y, quantization_nbins=10)
    assert str(same.signature).startswith(_SHORTCUT), "the identity cache no longer hits for an identical config"
