"""Regression test: ``mi_correction='chao_shen'`` no-op must surface as a ``UserWarning``, not
only a ``logger.warning``.

The Chao-Shen estimator is accepted as a valid ``mi_correction`` value but is not yet wired
into the relevance/null path -- it silently degrades to plug-in ('none') MI for both observed
and null. Before the fix this was only ``logger.warning``-ed, which a caller with unconfigured
logging (the common case for a plain script) never sees -- so a user who explicitly requested
the bias correction got ordinary plug-in MI with zero visible indication.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlframe.feature_selection.filters import MRMR


def _fast(**kw):
    """Build a fast-fitting MRMR instance for these tests, overridable via kwargs."""
    base = dict(full_npermutations=5, baseline_npermutations=3, n_jobs=1, verbose=0, fe_fast_search=False, interactions_max_order=1, random_seed=9)
    base.update(kw)
    return MRMR(**base)


def _xy(seed: int = 5, n: int = 160):
    """Build a small synthetic classification fixture with signal on columns 0 and 2."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] + 0.4 * X[:, 2] > 0).astype(np.int32)
    return X, y


def test_mi_correction_chao_shen_emits_user_warning():
    """``mi_correction='chao_shen'`` must emit a UserWarning when it falls back to plug-in MI."""
    X, y = _xy()
    MRMR._FIT_CACHE.clear()
    m = _fast(mi_correction="chao_shen")
    with pytest.warns(UserWarning, match="chao_shen"):
        m.fit(X, y)


def test_mi_correction_none_emits_no_chao_shen_warning():
    """The default ``mi_correction='none'`` must not emit the chao_shen fallback warning."""
    X, y = _xy()
    MRMR._FIT_CACHE.clear()
    m = _fast(mi_correction="none")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.fit(X, y)
    assert not any("chao_shen" in str(w.message) for w in caught)
