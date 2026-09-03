"""Split off ``mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core`` for the sub-split
that brings ``_fit_impl_core.py`` below the project's 1k-LOC module-size gate.

Dispatches ``_friend_graph_and_redundancy_passes``: the "Friend-graph post-analysis" section of
``MRMR._fit_impl`` -- friend-graph construction/pruning, adaptive-fourier/hybrid-orth/missingness
re-add passes, hinge/orth-basis protection, usability-aware raw-signal re-add, post-DCD cluster
pruning, pseudo-remix-aware post-selection redundancy drop, and the monotone-twin drop -- every
pass that runs on ``selected_vars`` in cols-space AFTER the main greedy screen and BEFORE the
cols-to-original-frame-index remap that follows this section in ``_fit_impl``. Further split
(2026-08-15) into four sibling group modules once this package itself crossed the 1k-LOC gate --
each group is a verbatim contiguous slice of the original single-function body; a systematic
cross-group name-usage audit (every local import's name checked against every OTHER group's
usage sites) confirmed no pass silently relies on an earlier pass's local import the way a
sibling split of ``_hybrid_orth_family_variants`` was caught doing (Python's function-level,
not block-level, scoping let 14 of 19 family blocks there implicitly share one early block's
import in the original monolith -- verified absent here before this split shipped).

Threads ``self`` plus every fit-body local this section reads as explicit keyword arguments
(mirrors the ``_finalise_fs_results`` / ``_assign_support`` carve-outs' own pattern), derived via
``pyutilz.dev.freevar_analysis`` rather than by eyeballing 1550 lines by hand. Unlike
``_assign_support`` (whose ``selected_vars`` is consumed entirely via ``self.*`` attributes it
sets), THIS section's ``selected_vars``/``cols``/``data``/``nbins`` mutations feed directly into
``_fit_impl``'s own next step (the cols-to-original-frame-index remap) -- so all four are BOTH
incoming parameters AND part of the return value, threaded through all four group siblings in
the SAME sequential order the original single function ran its passes in.

BUG FIX (caught post-extraction, before this module's first commit landed): the initial cut only
threaded ``selected_vars`` and ``X`` back out, on the assumption ``cols``/``data``/``nbins`` were
read-only in this section (true for ``_assign_support``, false here). In fact the hinge-recipe
re-add path (``cols = [*cols, _hn]``) and the post-DCD-swap path (``cols = _dref.get("cols",
cols)`` / same for ``data``/``nbins``) genuinely GROW/replace these locally inside the function --
a plain list/array reassignment inside a callee never propagates back to the caller. The symptom
was an ``IndexError`` at ``_fit_impl_core.py``'s very next line (``np.array(cols)[np.array
(selected_vars, ...)]``): ``selected_vars`` correctly reflected the grown cols-space (returned),
but the caller's own ``cols`` was silently still the PRE-growth object, one or more columns short.
Root-caused by comparing ``id(cols)`` immediately before this function's return vs immediately
after the call site in ``_fit_impl_core.py`` -- different objects, despite no re-entrant call.
"""

from __future__ import annotations

from ._group1 import _friend_graph_and_redundancy_passes_group1
from ._group2 import _friend_graph_and_redundancy_passes_group2
from ._group3 import _friend_graph_and_redundancy_passes_group3
from ._group4 import _friend_graph_and_redundancy_passes_group4


def _friend_graph_and_redundancy_passes(
    self,
    *,
    X,
    classes_y,
    cols,
    data,
    nbins,
    target_indices,
    y,
    verbose,
    cached_MIs,
    engineered_recipes,
    _eng_continuous_snapshot,
    selected_vars,
    _effective_min_relevance_gain,
    _hinge_deferred_recipes,
    _hinge_deferred_values,
    _hybrid_orth_pre_recipes,
    _miss_ind_pre_recipes,
    _persisted_dcd_state,
    _y_np,
    fe_to_pandas,
    _fe_family_on,
):
    """Run every post-screen, pre-remap cols-space pass and return ``(selected_vars, cols, data, nbins)``.

    See the module docstring for the full section this carves out and why all four are returned.
    Dispatches through the four group siblings in the SAME order the original single function ran
    its passes in -- each group receives the (possibly already-mutated) state the prior group
    returned.
    """
    selected_vars, cols, data, nbins = _friend_graph_and_redundancy_passes_group1(
        self=self,
        X=X,
        classes_y=classes_y,
        cols=cols,
        data=data,
        nbins=nbins,
        target_indices=target_indices,
        y=y,
        verbose=verbose,
        cached_MIs=cached_MIs,
        engineered_recipes=engineered_recipes,
        _eng_continuous_snapshot=_eng_continuous_snapshot,
        selected_vars=selected_vars,
        _effective_min_relevance_gain=_effective_min_relevance_gain,
        _hinge_deferred_recipes=_hinge_deferred_recipes,
        _hinge_deferred_values=_hinge_deferred_values,
        _hybrid_orth_pre_recipes=_hybrid_orth_pre_recipes,
        _miss_ind_pre_recipes=_miss_ind_pre_recipes,
        _persisted_dcd_state=_persisted_dcd_state,
        _y_np=_y_np,
        fe_to_pandas=fe_to_pandas,
        _fe_family_on=_fe_family_on,
    )
    selected_vars, cols, data, nbins = _friend_graph_and_redundancy_passes_group2(
        self=self,
        X=X,
        classes_y=classes_y,
        cols=cols,
        data=data,
        nbins=nbins,
        target_indices=target_indices,
        y=y,
        verbose=verbose,
        cached_MIs=cached_MIs,
        engineered_recipes=engineered_recipes,
        _eng_continuous_snapshot=_eng_continuous_snapshot,
        selected_vars=selected_vars,
        _effective_min_relevance_gain=_effective_min_relevance_gain,
        _hinge_deferred_recipes=_hinge_deferred_recipes,
        _hinge_deferred_values=_hinge_deferred_values,
        _hybrid_orth_pre_recipes=_hybrid_orth_pre_recipes,
        _miss_ind_pre_recipes=_miss_ind_pre_recipes,
        _persisted_dcd_state=_persisted_dcd_state,
        _y_np=_y_np,
        fe_to_pandas=fe_to_pandas,
        _fe_family_on=_fe_family_on,
    )
    selected_vars, cols, data, nbins = _friend_graph_and_redundancy_passes_group3(
        self=self,
        X=X,
        classes_y=classes_y,
        cols=cols,
        data=data,
        nbins=nbins,
        target_indices=target_indices,
        y=y,
        verbose=verbose,
        cached_MIs=cached_MIs,
        engineered_recipes=engineered_recipes,
        _eng_continuous_snapshot=_eng_continuous_snapshot,
        selected_vars=selected_vars,
        _effective_min_relevance_gain=_effective_min_relevance_gain,
        _hinge_deferred_recipes=_hinge_deferred_recipes,
        _hinge_deferred_values=_hinge_deferred_values,
        _hybrid_orth_pre_recipes=_hybrid_orth_pre_recipes,
        _miss_ind_pre_recipes=_miss_ind_pre_recipes,
        _persisted_dcd_state=_persisted_dcd_state,
        _y_np=_y_np,
        fe_to_pandas=fe_to_pandas,
        _fe_family_on=_fe_family_on,
    )
    selected_vars, cols, data, nbins = _friend_graph_and_redundancy_passes_group4(
        self=self,
        X=X,
        classes_y=classes_y,
        cols=cols,
        data=data,
        nbins=nbins,
        target_indices=target_indices,
        y=y,
        verbose=verbose,
        cached_MIs=cached_MIs,
        engineered_recipes=engineered_recipes,
        _eng_continuous_snapshot=_eng_continuous_snapshot,
        selected_vars=selected_vars,
        _effective_min_relevance_gain=_effective_min_relevance_gain,
        _hinge_deferred_recipes=_hinge_deferred_recipes,
        _hinge_deferred_values=_hinge_deferred_values,
        _hybrid_orth_pre_recipes=_hybrid_orth_pre_recipes,
        _miss_ind_pre_recipes=_miss_ind_pre_recipes,
        _persisted_dcd_state=_persisted_dcd_state,
        _y_np=_y_np,
        fe_to_pandas=fe_to_pandas,
        _fe_family_on=_fe_family_on,
    )
    return selected_vars, cols, data, nbins
