"""The additional-RFECV rescue only ever considers raw input columns, and is skipped when none is eligible.

Its pool was ``X.columns`` minus the selection and two FE rosters (hybrid-orth, mi-greedy). Engineered columns from every other FE family
(here an unary/binary pair column) stayed in the pool, so RFECV ran on them; had it selected one, mapping it back through
``feature_names_in_`` would have raised KeyError. Its guard also counted ``X.shape[1] - len(selected_vars)`` across two index spaces, so an
all-raw-selected fit still built and fitted RFECV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import mrmr as mrmr_pkg
from mlframe.feature_selection.filters.mrmr import MRMR


class _RFECVSpy:
    """Stands in for RFECV: records construction and the pool it is fitted on, and selects nothing."""

    constructed = 0
    pools: list = []

    def __init__(self, *args, **kwargs):
        """Count the construction."""
        type(self).constructed += 1

    def fit(self, X, y):
        """Record the pool columns and select nothing."""
        type(self).pools.append(list(X.columns))
        self.n_features_ = 0
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        return self


def _fit(monkeypatch, X, y):
    """Fit with FE on and the rescue enabled, RFECV replaced by the spy."""
    _RFECVSpy.constructed = 0
    _RFECVSpy.pools = []
    monkeypatch.setattr(mrmr_pkg, "RFECV", _RFECVSpy)
    MRMR._FIT_CACHE.clear()
    m = MRMR(random_seed=0, n_jobs=1, verbose=0, fe_max_steps=1, full_npermutations=3, baseline_npermutations=2, run_additional_rfecv_minutes=1)
    return m.fit(X, y)


def test_rfecv_rescue_skipped_when_pool_is_empty(monkeypatch):
    """Every raw column carries signal and is selected; FE adds engineered columns; the rescue must not construct RFECV."""
    rng = np.random.default_rng(0)
    n = 800
    a, b = rng.normal(size=n), rng.normal(size=n)
    X = pd.DataFrame({"a": a, "b": b})
    y = ((a + b + 0.5 * a * b) > 0).astype(np.int64)
    m = _fit(monkeypatch, X, y)
    assert set(np.asarray(m.feature_names_in_)[np.asarray(m.support_)]) == {"a", "b"}, "fixture precondition: both raw columns selected"
    assert _RFECVSpy.constructed == 0, "RFECV was constructed although no discarded raw column was eligible for the rescue"


def test_rfecv_rescue_pool_contains_only_raw_columns(monkeypatch):
    """With a discarded raw noise column and engineered columns present, every column handed to RFECV is a raw input column."""
    rng = np.random.default_rng(1)
    n = 800
    a, b, noise = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    X = pd.DataFrame({"a": a, "b": b, "noise": noise})
    y = ((a + b + 0.5 * a * b) > 0).astype(np.int64)
    m = _fit(monkeypatch, X, y)
    raw = set(map(str, m.feature_names_in_))
    assert _RFECVSpy.pools, "fixture precondition: the rescue must run on the discarded raw column"
    for pool in _RFECVSpy.pools:
        leaked = [c for c in pool if str(c) not in raw]
        assert not leaked, f"engineered columns reached the raw-only rescue pool: {leaked}"
