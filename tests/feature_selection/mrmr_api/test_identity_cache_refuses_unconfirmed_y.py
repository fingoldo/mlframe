"""Regression: the cross-target identity-cache shortcut must NOT emit a cached identity
selection for a target it cannot confirm. When the y-correlation gate is requested
(mrmr_identity_cache_ycorr_threshold > 0) but the cached entry is the legacy bool format
with no prior y-sample to check against, the pre-fix code left _ycorr_ok = True and fired
the shortcut anyway -- returning a selection that never saw the new y. The fix refuses and
runs a full fit in that case, while preserving the legacy threshold==0.0 opt-out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import mlframe.feature_selection.filters.mrmr._mrmr_class as _mc
from mlframe.feature_selection.filters.mrmr import MRMR, _mrmr_compute_x_fingerprint

_SHORTCUT = "_mrmr_identity_shortcut"


def _xy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n), "d": rng.normal(size=n)})
    y = ((X["a"].to_numpy() + 0.1 * rng.normal(size=n)) > 0).astype(int)
    return X, y


def _fit_with_cache(cache, thr, X, y):
    m = MRMR(mrmr_skip_when_prior_was_identity=True, mrmr_identity_cache_include_y=False,
             mrmr_identity_cache_ycorr_threshold=thr, max_runtime_mins=1.0)
    m._mlframe_identity_cache_override_ = cache
    m.fit(X, y)
    return m


def test_refuses_shortcut_when_threshold_set_but_no_prior_y_sample():
    """thr=0.5 + legacy bool cache entry (no y-sample) -> cannot confirm -> full fit, not shortcut."""
    X, y = _xy()
    x_fp = _mrmr_compute_x_fingerprint(X)
    m = _fit_with_cache({x_fp: True}, thr=0.5, X=X, y=y)
    assert not str(m.signature).startswith(_SHORTCUT), (
        "identity shortcut fired for an unconfirmable target (thr>0, no prior y-sample) -- should refuse"
    )


def test_fires_shortcut_when_prior_y_sample_correlates():
    """thr=0.5 + a prior y-sample correlated with the new y -> the shortcut legitimately fires."""
    X, y = _xy()
    x_fp = _mrmr_compute_x_fingerprint(X)
    prior_sample = _mc._mrmr_y_corr_sample(y)  # identical target -> corr 1.0 >= 0.5
    m = _fit_with_cache({x_fp: (True, prior_sample)}, thr=0.5, X=X, y=y)
    assert str(m.signature).startswith(_SHORTCUT), (
        "identity shortcut should fire when the prior y-sample confirms the new target correlates"
    )


def test_legacy_threshold_zero_opt_out_still_fires():
    """thr=0.0 is the documented legacy opt-out (gate off): the shortcut still fires on a bool entry."""
    X, y = _xy()
    x_fp = _mrmr_compute_x_fingerprint(X)
    m = _fit_with_cache({x_fp: True}, thr=0.0, X=X, y=y)
    assert str(m.signature).startswith(_SHORTCUT), "threshold==0.0 must preserve the legacy fire-anyway behaviour"
