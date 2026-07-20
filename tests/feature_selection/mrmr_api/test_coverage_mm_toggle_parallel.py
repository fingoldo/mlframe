"""Regression (A2, re-framed 2026-07-20): the Miller-Madow mi_correction toggle must be active
during candidate evaluation regardless of ``n_workers``.

Originally this pinned that the toggle reaches the joblib threading workers spawned by
``score_candidates``'s ``evaluate_candidates`` dispatch (threading.local does not cross into
worker threads, so the worker had to re-publish the six Wave-8 toggles -- mi_miller_madow was
once omitted, leaving MM a silent no-op there). That worker pool was permanently retired
2026-07-19 (``_confirm_predictor.py``'s ``_EVALUATE_CANDIDATES_POOL_ENABLED = False`` -- an
isolated/warmed A/B found it never wins over the serial fallback at realistic candidate counts),
so ``_evaluate_candidates_inner`` is no longer reachable via any worker thread; every candidate
now evaluates on the main thread through ``evaluate_candidate`` (singular), where
threading.local naturally applies with no forwarding needed. The regression this test guards
against is therefore now structurally impossible to reintroduce via that path, but the toggle
correctness itself is still worth pinning against a real ``n_workers>1`` fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mlframe.feature_selection.filters._confirm_predictor as _cp
from mlframe.feature_selection.filters.info_theory import use_mi_miller_madow
from mlframe.feature_selection.filters.mrmr import MRMR


def _wide_xy(n=1200, p=50, seed=0):
    """Wide xy."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = ((X["f0"].to_numpy() + X["f1"].to_numpy() - X["f2"].to_numpy() + 0.1 * rng.normal(size=n)) > 0).astype(int)
    return X, y


def _toggle_seen_during_fit(mi_correction, monkeypatch):
    """Spy on the serial per-candidate scorer (the only path evaluate_candidates now reaches, since the joblib worker pool was retired) and record the toggle's value at each call."""
    seen: list[bool] = []
    real = _cp.evaluate_candidate

    def _spy(*args, **kwargs):
        """Helper that spy."""
        seen.append(use_mi_miller_madow())
        return real(*args, **kwargs)

    monkeypatch.setattr(_cp, "evaluate_candidate", _spy)
    X, y = _wide_xy()
    MRMR(mi_correction=mi_correction, n_workers=4, max_runtime_mins=2.0).fit(X, y)
    return seen


def test_miller_madow_toggle_active_in_parallel_workers(monkeypatch):
    """Miller madow toggle active in parallel workers."""
    seen = _toggle_seen_during_fit("miller_madow", monkeypatch)
    assert seen, "candidate scoring path was not exercised (fixture too small)"
    assert all(seen), f"mi_correction='miller_madow' did not stay active during scoring: {seen} (pre-fix: all False)"


def test_no_correction_leaves_toggle_off_in_workers(monkeypatch):
    """No correction leaves toggle off in workers."""
    seen = _toggle_seen_during_fit("none", monkeypatch)
    assert seen, "candidate scoring path was not exercised"
    assert not any(seen), f"MM wrongly active during scoring with mi_correction='none': {seen}"
