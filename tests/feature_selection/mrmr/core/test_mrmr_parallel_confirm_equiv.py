"""Regression: each threading worker in the parallel candidate-scoring round must receive its OWN
copy of the dicts it writes in-place (cached_MIs, partial_gains). Sharing one object let workers
__setitem__ it concurrently -> a "dict changed size during iteration" crash at scale. Pre-fix every
worker got the same shared object (one id); the fix passes a per-worker sanitize() copy (distinct ids).

(A serial-vs-parallel selection-equivalence assertion would be confounded by a separate, pre-existing
tie-break order divergence between the serial and parallel incumbent paths, so this pins the copy
mechanism directly instead.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mlframe.feature_selection.filters._confirm_predictor as _cp
from mlframe.feature_selection.filters.mrmr import MRMR


def _wide_xy(n=1200, p=50, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = ((X["f0"].to_numpy() + X["f1"].to_numpy() - X["f2"].to_numpy() + 0.1 * rng.normal(size=n)) > 0).astype(int)
    return X, y


def test_parallel_workers_receive_distinct_dict_copies(monkeypatch):
    cached_ids: list[int] = []
    partial_ids: list[int] = []
    real = _cp.evaluate_candidates

    def _spy(*args, **kwargs):
        cached_ids.append(id(kwargs.get("cached_MIs")))
        partial_ids.append(id(kwargs.get("partial_gains")))
        return real(*args, **kwargs)

    monkeypatch.setattr(_cp, "evaluate_candidates", _spy)
    X, y = _wide_xy()
    MRMR(n_workers=4, max_runtime_mins=2.0).fit(X, y)

    assert cached_ids, "evaluate_candidates was never called (fixture too small to reach the parallel path)"
    assert len(set(cached_ids)) >= 2, (
        f"workers shared one cached_MIs object ({len(set(cached_ids))} distinct of {len(cached_ids)} calls) "
        "-- per-worker copy missing; concurrent __setitem__ can crash"
    )
    assert len(set(partial_ids)) >= 2, f"workers shared one partial_gains object ({len(set(partial_ids))} distinct of {len(partial_ids)} calls)"
