"""Regression: the final p>=n raw cap must hand its capped list back to ``_assign_support_tail``.

Before the fix ``_assign_support_tai_adds_more_raw_columns`` re-capped the raw support into a local
``selected_vars`` and returned nothing, so the caller kept the uncapped list. Every later stage that
recomputes ``n_features_`` from that list could push the count back over the ``max(20, p//3)`` ceiling.
"""

from __future__ import annotations

import numpy as np

import mlframe.feature_selection.filters._mrmr_fit_impl._assign_support_tail as tail_mod
from mlframe.feature_selection.filters._mrmr_fit_impl._assign_support_tail import (
    _assign_support_tai_adds_more_raw_columns,
    _assign_support_tail,
)

_MOD = "mlframe.feature_selection.filters._mrmr_fit_impl"


class _FitState:
    """Minimal stand-in for the MRMR instance: only the attributes the cap helper reads and writes."""

    def __init__(self, names, relevance_by_index, n_in):
        self.feature_names_in_ = list(names)
        self.n_features_in_ = n_in
        self.cached_MIs = {(i,): float(relevance_by_index[i]) for i in range(len(names))}
        self._engineered_recipes_ = []
        self.support_ = None
        self.n_features_ = None


def test_pgn_cap_helper_returns_capped_selection_for_caller():
    """Pre-fix: the helper returned None, so the caller could not adopt the capped list (TypeError on len).

    p=60, n=30 gives ceiling max(20, 60//3)=20 with no engineered survivors, so 40 raw selections must be
    trimmed to the 20 most relevant ones. The returned list must be that trimmed set, and the instance's
    support_/n_features_ must agree with it.
    """
    p, n = 60, 30
    names = [f"x{i}" for i in range(p)]
    # Relevance strictly decreasing in the column index, so the top-20 by relevance is exactly 0..19.
    relevance = [float(p - i) for i in range(p)]
    state = _FitState(names, relevance, n_in=p)
    selected = list(range(40))

    capped = _assign_support_tai_adds_more_raw_columns(state, p, n, selected, names, 0, False)

    assert capped is not None, "helper must return the capped selection so the caller can adopt it"
    assert len(capped) == 20
    assert sorted(int(v) for v in capped) == list(range(20))
    assert np.array_equal(np.asarray(state.support_), np.asarray(capped))
    assert state.n_features_ == len(capped)


def test_pgn_cap_helper_returns_selection_unchanged_under_budget():
    """Below the ceiling the helper must hand back the same selection, not a re-ordered or trimmed one."""
    p, n = 60, 30
    names = [f"x{i}" for i in range(p)]
    state = _FitState(names, [float(p - i) for i in range(p)], n_in=p)
    selected = [3, 7, 11]

    returned = _assign_support_tai_adds_more_raw_columns(state, p, n, selected, names, 0, False)

    assert list(returned) == [3, 7, 11]


def test_assign_support_tail_passes_capped_selection_to_downstream_stages(monkeypatch):
    """Caller path: the list handed to the stage after the cap must be the capped list, not the uncapped one.

    Pre-fix the caller discarded the helper's (absent) return, so ``run_residual_rescue`` received all 40 raw
    selections and any later ``n_features_`` recount exceeded the ceiling of 20. The downstream stages are
    stubbed to pass-through so the only thing under test is what the caller threads through the cap.
    """
    p, n = 60, 30
    names = [f"x{i}" for i in range(p)]
    state = _FitState(names, [float(p - i) for i in range(p)], n_in=p)
    state._lean_second_pass_ = True  # skips the fe_max_steps retention block, which is not under test
    state.min_features_fallback = 0

    seen = {}

    def _record_rescue(mrmr, X, selected_vars, cols, recipes, added, verbose, prune_only=False):
        """Stand-in for the residual rescue stage that records the selection it was handed."""
        seen["selected_vars"] = list(selected_vars)
        return False

    monkeypatch.setattr(tail_mod, "_assign_support_tai_drop_sweep_disabled_shares", lambda self, sel, *a, **k: sel)
    monkeypatch.setattr(tail_mod, "_assign_support_tai_drop_exact_linear_dependence", lambda self, X, cols, sel, verbose: (False, sel))
    monkeypatch.setattr(f"{_MOD}._residual_rescue_stage.run_residual_rescue", _record_rescue)
    monkeypatch.setattr(f"{_MOD}._final_subsumption_stage.run_final_subsumption_prune", lambda self, X, sel, cy, verbose: (False, sel))
    monkeypatch.setattr(f"{_MOD}._selection_compaction.run_selection_compaction", lambda self, X, sel, *a, **k: (False, sel))
    monkeypatch.setattr(f"{_MOD}._main_effect_restore.run_main_effect_restore", lambda self, X, sel, *a, **k: (False, sel))

    data = np.zeros((n, p), dtype=np.float32)
    _assign_support_tail(
        state,
        X=None,
        classes_y=None,
        cols=names,
        data=data,
        nbins=10,
        target_indices=[],
        y=None,
        verbose=0,
        fe_max_steps=0,
        _eng_continuous_snapshot=None,
        selected_vars=list(range(40)),
        _allowed_raw_idx=None,
        _retention_added_eng_names=set(),
    )

    assert len(seen["selected_vars"]) == 20, "downstream stage must receive the capped list, got the uncapped one"
    assert sorted(seen["selected_vars"]) == list(range(20))
    assert state.n_features_ == 20
