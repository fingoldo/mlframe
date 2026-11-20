"""Correctness regressions of the usability-aware selection helpers and the unified FE gate."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_usability_lists_drop_rows_with_nonfinite_target(monkeypatch):
    """A NaN target row is dropped before the selection instead of being scrubbed to a fabricated 0.0."""
    from mlframe.feature_selection.filters import _usability_aware_selection as aware, _usability_lists as lists

    seen: dict = {}

    def _pool(df, y, base_names, **kw):
        """Record the frame and target the pool is built from."""
        seen["pool_rows"], seen["pool_y"] = len(df), np.asarray(y)
        return []

    def _greedy(pool, y, **kw):
        """Record the target the greedy sees."""
        seen["greedy_y"] = np.asarray(y)
        return []

    monkeypatch.setattr(aware, "build_usability_candidate_pool", _pool)
    monkeypatch.setattr(aware, "usability_greedy", _greedy)
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.standard_normal(40), "b": rng.standard_normal(40)})
    y = rng.standard_normal(40)
    y[[3, 17]] = np.nan
    mrmr = SimpleNamespace(feature_names_in_=np.array(["a", "b"]), support_=None)
    lists.build_usability_lists(mrmr, X, y)
    assert seen["pool_rows"] == 38
    assert np.isfinite(seen["pool_y"]).all() and np.isfinite(seen["greedy_y"]).all()
    assert seen["greedy_y"].size == 38


def test_multiclass_cv_logloss_survives_a_train_fold_missing_a_class():
    """A fold whose train rows miss a class still scores: the probabilities are widened to the full class set."""
    from mlframe.feature_selection.filters._usability_aware_selection import _usability_scorers
    from mlframe.feature_selection.filters._usability_greedy_steps import _usability_greedy_step2_def_cv_fold

    rng = np.random.default_rng(1)
    n = 60
    y = rng.integers(0, 2, n)
    y[[0, 3]] = 2  # both class-2 rows sit in fold 0
    folds = np.arange(n) % 3
    X = rng.standard_normal((n, 1)) + y[:, None]
    mk, logloss, y_enc, n_classes = _usability_scorers(True, y)
    cv = _usability_greedy_step2_def_cv_fold(True, 3, folds, y_enc, n_classes, logloss, lambda i: X[:, i], mk, y)
    errs = cv([0])
    assert errs.shape == (3,) and np.isfinite(errs).all()
    assert errs[0] > errs[1]  # the unseen class costs the fold that holds it


class _NumpyCp:
    """numpy as the array namespace of the resident twins, with cupy's non-raising ``linalg.solve`` on a singular matrix."""

    class linalg:
        """cupy's solve returns inf / nan on a singular matrix instead of raising."""

        @staticmethod
        def solve(a, b):
            """numpy solve, nan-filled where numpy would raise."""
            try:
                return np.linalg.solve(a, b)
            except np.linalg.LinAlgError:
                return np.full(np.shape(b), np.nan)

    asnumpy = staticmethod(np.asarray)

    def __getattr__(self, name):
        """Everything else is numpy's."""
        return getattr(np, name)


def _duplicate_column_problem():
    """A design whose columns 0 and 1 are identical (a singular border), with fold index arrays."""
    rng = np.random.default_rng(2)
    n = 40
    x = rng.standard_normal(n)
    V = np.column_stack([x, x, rng.standard_normal(n)])
    y = x + 0.1 * rng.standard_normal(n)
    folds = np.arange(n) % 2
    tr = [np.where(folds != f)[0] for f in range(2)]
    va = [np.where(folds == f)[0] for f in range(2)]
    return V, y, tr, va


def test_resident_bordered_solve_nonfinite_triggers_cpu_fallback():
    """A singular bordered Gram (cupy returns nan) raises the fallback signal instead of silently rejecting the candidate."""
    from mlframe.feature_selection.filters import _usability_greedy_gpu_resident as mod

    V, y, tr, va = _duplicate_column_problem()
    cv = mod._usability_greedy_gpu_step3_def_cv_candidates(2, tr, va, y, V, _NumpyCp())
    with pytest.raises(mod._ResidentFallbackError):
        cv([0], [1])


def test_resident_selected_fit_nonfinite_triggers_cpu_fallback():
    """A singular selected design (cupy returns nan) raises the fallback signal instead of a nan residual that zeroes the shortlist scores."""
    from mlframe.feature_selection.filters import _usability_greedy_gpu_resident as mod

    V, y, tr, va = _duplicate_column_problem()
    cp = _NumpyCp()
    abscorr = mod._usability_greedy_gpu_step1_known_output_size(V, cp, V.shape[1])
    _, shortlist = mod._usability_greedy_gpu_step2_def_shortlist_sel(va, tr, V, y, abscorr, cp, np.full(3, 0.1), 3, 0.5, 0.1, 0.97, 3, 2)
    with pytest.raises(mod._ResidentFallbackError):
        shortlist([0, 1])


def test_unified_second_pass_gate_seed_obeys_fragmentation_cap():
    """Four raw seed columns that jointly name every row do not zero the CMI of a genuinely informative engineered column."""
    from mlframe.feature_selection.filters._unified_fe_gate import unified_second_pass_gate

    rng = np.random.default_rng(3)
    n = 150
    y = rng.integers(0, 2, n)
    frame = pd.DataFrame({f"r{i}": rng.standard_normal(n) for i in range(4)})
    frame["eng"] = 0.4 * y + rng.standard_normal(n)
    kept = unified_second_pass_gate(frame, y, [f"r{i}" for i in range(4)], ["eng"], seed_raw_cols_count=4)
    assert kept == ["eng"]
