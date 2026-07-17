"""Regression (A2): the Miller-Madow mi_correction toggle must reach the joblib threading
workers. threading.local does not cross into worker threads, so the worker re-publishes the
six Wave-8 toggles -- but mi_miller_madow was omitted, leaving MM a silent no-op in the parallel
greedy loop (mi_or_su / the class-MI kernels consult use_mi_miller_madow()). The fix forwards it
alongside the others.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mlframe.feature_selection.filters._evaluation_driver as _ed
from mlframe.feature_selection.filters.info_theory import use_mi_miller_madow
from mlframe.feature_selection.filters.mrmr import MRMR


def _wide_xy(n=1200, p=50, seed=0):
    """Wide xy."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = ((X["f0"].to_numpy() + X["f1"].to_numpy() - X["f2"].to_numpy() + 0.1 * rng.normal(size=n)) > 0).astype(int)
    return X, y


def _toggle_seen_in_workers(mi_correction, monkeypatch):
    """Toggle seen in workers."""
    seen: list[bool] = []
    real = _ed._evaluate_candidates_inner

    def _spy(*args, **kwargs):
        """Helper that spy."""
        seen.append(use_mi_miller_madow())  # read the worker thread's toggle, set just before this call
        return real(*args, **kwargs)

    monkeypatch.setattr(_ed, "_evaluate_candidates_inner", _spy)
    X, y = _wide_xy()
    MRMR(mi_correction=mi_correction, n_workers=4, max_runtime_mins=2.0).fit(X, y)
    return seen


def test_miller_madow_toggle_active_in_parallel_workers(monkeypatch):
    """Miller madow toggle active in parallel workers."""
    seen = _toggle_seen_in_workers("miller_madow", monkeypatch)
    assert seen, "parallel worker path was not exercised (fixture too small)"
    assert all(seen), f"mi_correction='miller_madow' did not reach the workers: {seen} (pre-fix: all False)"


def test_no_correction_leaves_toggle_off_in_workers(monkeypatch):
    """No correction leaves toggle off in workers."""
    seen = _toggle_seen_in_workers("none", monkeypatch)
    assert seen, "parallel worker path was not exercised"
    assert not any(seen), f"MM wrongly active in workers with mi_correction='none': {seen}"
