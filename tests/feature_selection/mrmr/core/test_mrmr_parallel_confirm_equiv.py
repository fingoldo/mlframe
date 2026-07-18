"""Regression: each threading worker in the parallel candidate-scoring round must receive its OWN
copy of the dicts it writes in-place (cached_MIs, partial_gains). Sharing one object let workers
__setitem__ it concurrently -> a "dict changed size during iteration" crash at scale. Pre-fix every
worker got the same shared object (one id); the fix passes a per-worker sanitize() copy (distinct ids).

(A serial-vs-parallel selection-equivalence assertion would be confounded by a separate, pre-existing
tie-break order divergence between the serial and parallel incumbent paths, so this pins the copy
mechanism directly instead.)

RE-FRAMED (2026-07-19, 7-site joblib.Parallel audit, site 3): the parallel ``evaluate_candidates`` dispatch
this test exercises is no longer reachable through a normal ``MRMR(n_workers=...).fit(...)`` call --
isolated/warmed/best-of-3+ measurement found the pool never wins over serial at any tested scale (m=10 ->
0.03x, m=320 wellbore-scale -> 0.72-0.73x, m=820/n_workers=8 -> 0.81x), so ``score_candidates`` now gates
the branch behind ``_EVALUATE_CANDIDATES_POOL_ENABLED = False`` (see ``_confirm_predictor.py``) and
``_screen_predictors.py`` never builds a non-``None`` ``workers_pool`` for it. Per this repo's
REJECTED-!=-DELETED convention the branch/mechanism is kept, not deleted, so this test now force-re-enables
the flag via monkeypatch to prove the retired per-worker-copy mechanism is still correct -- it would fail
identically to before if that mechanism ever regressed, it is just no longer reachable by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mlframe.feature_selection.filters._confirm_predictor as _cp
from mlframe.feature_selection.filters.mrmr import MRMR


def _wide_xy(n=1200, p=50, seed=0):
    """Wide xy."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = ((X["f0"].to_numpy() + X["f1"].to_numpy() - X["f2"].to_numpy() + 0.1 * rng.normal(size=n)) > 0).astype(int)
    return X, y


def test_parallel_workers_receive_distinct_dict_copies(monkeypatch):
    """Parallel workers receive distinct dict copies.

    Force-re-enables the retired ``_EVALUATE_CANDIDATES_POOL_ENABLED`` gate (default ``False`` since the
    2026-07-19 fix) and rebuilds a real ``workers_pool`` via the same construction the (now-dead-by-default)
    ``_screen_predictors.py`` code used, so the parallel branch is actually reached.
    """
    monkeypatch.setattr(_cp, "_EVALUATE_CANDIDATES_POOL_ENABLED", True)

    import mlframe.feature_selection.filters._screen_predictors as _sp

    _real_score_candidates_setup_done = {"n": 0}
    _real_ctor = _cp.ScreenContext.__init__

    def _patched_init(self, *args, **kwargs):
        """Build a real threading Parallel pool in place of the retired (always-None) default, so the
        parallel evaluate_candidates branch under test is actually reached."""
        _real_ctor(self, *args, **kwargs)
        if getattr(self, "n_workers", 0) and self.n_workers > 1 and self.workers_pool is None:
            self.workers_pool = _sp.Parallel(n_jobs=self.n_workers, backend="threading", max_nbytes=None)
            _real_score_candidates_setup_done["n"] += 1

    monkeypatch.setattr(_cp.ScreenContext, "__init__", _patched_init)

    cached_ids: list[int] = []
    partial_ids: list[int] = []
    real = _cp.evaluate_candidates

    def _spy(*args, **kwargs):
        """Helper that spy."""
        cached_ids.append(id(kwargs.get("cached_MIs")))
        partial_ids.append(id(kwargs.get("partial_gains")))
        return real(*args, **kwargs)

    monkeypatch.setattr(_cp, "evaluate_candidates", _spy)
    X, y = _wide_xy()
    MRMR(n_workers=4, max_runtime_mins=2.0).fit(X, y)

    assert _real_score_candidates_setup_done["n"] > 0, "ScreenContext was never built with n_workers>1 (fixture too small to reach the parallel path)"
    assert cached_ids, "evaluate_candidates was never called (fixture too small to reach the parallel path)"
    assert len(set(cached_ids)) >= 2, (
        f"workers shared one cached_MIs object ({len(set(cached_ids))} distinct of {len(cached_ids)} calls) "
        "-- per-worker copy missing; concurrent __setitem__ can crash"
    )
    assert len(set(partial_ids)) >= 2, f"workers shared one partial_gains object ({len(set(partial_ids))} distinct of {len(partial_ids)} calls)"
