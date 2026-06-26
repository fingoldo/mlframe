"""Regression (C4): the SERIAL Fleuret confirm path must compute confidence with the same
add-one (Monte-Carlo) p-value estimator the PARALLEL path uses. Pre-fix the serial branch did
``confidence = 1 - nfailed / nchecked`` (raw), which reports confidence exactly 1.0 on a clean
null (nfailed=0) -- the parallel path (get_fleuret_criteria_confidence_parallel -> _perm_pvalue)
never does, so n_workers=1 and n_workers>1 runs diverged in the confidence that feeds ranking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import mlframe.feature_selection.filters._confirm_predictor as _cp
from mlframe.feature_selection.filters.mrmr import MRMR


def _xy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(size=n), "b": rng.normal(size=n),
        "c": rng.normal(size=n), "d": rng.normal(size=n),
    })
    y = ((X["a"].to_numpy() + 0.5 * X["b"].to_numpy() + 0.1 * rng.normal(size=n)) > 0).astype(int)
    return X, y


def test_serial_fleuret_confidence_routes_through_addone(monkeypatch):
    seen_nchecked: list[int] = []
    real_pv = _cp._perm_pvalue

    def _spy_pv(nfailed, nchecked, **kw):
        seen_nchecked.append(int(nchecked))
        return real_pv(nfailed, nchecked, **kw)

    # Force the serial Fleuret core to report a clean null with a distinctive nchecked=64 so we can
    # detect whether the serial confirm routes that through _perm_pvalue (add-one) or computes it raw.
    monkeypatch.setattr(_cp, "_perm_pvalue", _spy_pv)
    monkeypatch.setattr(_cp, "get_fleuret_criteria_confidence", lambda **kw: (0, 64))

    X, y = _xy()
    MRMR(n_jobs=1, mrmr_relevance_algo="fleuret", max_runtime_mins=1.0).fit(X, y)

    assert 64 in seen_nchecked, (
        "serial Fleuret confirm did not route confidence through _perm_pvalue -- still raw nfailed/nchecked "
        "(pre-fix), so serial and parallel confidence diverge on a clean null"
    )
