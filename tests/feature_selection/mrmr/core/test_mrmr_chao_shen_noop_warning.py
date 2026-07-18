"""Regression test: ``mi_correction='chao_shen'`` must NOT emit the old no-op fallback warning.

``mi_correction='chao_shen'`` used to be accepted as a valid value but not wired into the
relevance/null path -- it silently degraded to plug-in ('none') MI for both observed and null,
surfaced via a UserWarning. It is now genuinely wired (05_concurrency_and_statistics.md finding #7,
see ``compute_relevance_score``/``mi_or_su_from_classes`` in ``info_theory``); this file re-frames the
original no-op-warning contract to its new, correct opposite: chao_shen must complete SILENTLY (no
fallback warning) since it is a real, active estimator now, not a no-op. See
``test_biz_val_filters_mrmr_chao_shen.py`` for the activation/behavioral tests.
"""

from __future__ import annotations

import warnings

import numpy as np

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


def test_mi_correction_chao_shen_no_longer_emits_noop_warning():
    """``mi_correction='chao_shen'`` must NOT emit the stale "not yet wired" fallback warning -- it is
    now a genuinely-active estimator, not a no-op."""
    X, y = _xy()
    MRMR._FIT_CACHE.clear()
    m = _fast(mi_correction="chao_shen")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.fit(X, y)
    stale = [w for w in caught if "chao_shen" in str(w.message).lower() and "not yet wired" in str(w.message).lower()]
    assert not stale, f"chao_shen must no longer emit the stale not-wired warning; got {[str(w.message) for w in stale]}"


def test_mi_correction_none_emits_no_chao_shen_warning():
    """The default ``mi_correction='none'`` must not emit any chao_shen-related warning."""
    X, y = _xy()
    MRMR._FIT_CACHE.clear()
    m = _fast(mi_correction="none")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.fit(X, y)
    assert not any("chao_shen" in str(w.message).lower() for w in caught)
