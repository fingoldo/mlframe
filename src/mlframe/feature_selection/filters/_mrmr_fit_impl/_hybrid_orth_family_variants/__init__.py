"""Split off ``mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core`` for the sub-split
that brings ``_fit_impl_core.py`` below the project's 1k-LOC module-size gate.

Dispatches ``_hybrid_orth_family_variants``: the block of ~19 ``fe_hybrid_orth_*_enable``-gated FE
family stages inside ``MRMR._fit_impl`` (triplet, quadruplet, adaptive-arity, adaptive-degree,
conditional-routing, diff-basis, cluster-basis, bootstrap, three-gate, KSG, copula, dCor, HSIC,
JMIM, TC, CMIM, auto-scorer, ensemble, meta) -- one contiguous run of near-identical-shaped blocks
between the initial ``_gbm_seeded_triplet_names`` setup and the (separately-gated) MI-greedy FE
stage that follows. Further split (2026-08-15) into four sibling group modules (``_group1``
through ``_group4``, ~5 families each) once this package itself crossed the 1k-LOC gate -- each
group is a verbatim contiguous slice of the original single-function body, so no per-block
free-variable re-analysis was needed: the whole original function only ever closed over the params
below, so every slice's free variables are a subset by construction.

Threads ``self`` plus every fit-body local this section reads as explicit keyword arguments
(mirrors the other sub-split carve-outs' own pattern), derived via ``pyutilz.dev.freevar_analysis``.
Unlike the cols-space sections (``_assign_support``, ``_friend_graph_and_redundancy``), this
section operates entirely on ``X`` (the raw/engineered pandas-or-polars frame, format-agnostic via
the matrix-native FE seam) -- each family stage appends its own winning columns onto ``X`` via
``fe_append_columns``/``fe_extract_columns`` and records recipes into ``_hybrid_orth_pre_recipes``
(a dict, mutated in place) plus ``self.hybrid_orth_features_`` (a list, reassigned in place via
``self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + [...]`` at each site). Like
``_friend_graph_and_redundancy``'s ``selected_vars``, ``X`` is passed in AND returned: confirmed by
grepping every ``X = fe_append_columns(X, ...)`` reassignment in range (19 sites, one per family)
and that the code immediately following this section keeps reading/mutating ``X``.
"""

from __future__ import annotations

from ._group1 import _hybrid_orth_family_variants_group1
from ._group2 import _hybrid_orth_family_variants_group2
from ._group3 import _hybrid_orth_family_variants_group3
from ._group4 import _hybrid_orth_family_variants_group4


def _hybrid_orth_family_variants(
    self,
    *,
    X,
    y,
    verbose,
    _y_np,
    _hybrid_orth_pre_recipes,
    _gbm_seeded_triplet_names,
    _fe_family_on,
):
    """Run every ``fe_hybrid_orth_*_enable``-gated FE family stage and return the (possibly
    column-augmented) ``X``.

    See the module docstring for the full section this carves out. Dispatches through the four
    group siblings in the SAME order the original single function ran its 19 families in -- each
    group receives the (possibly already-augmented) ``X`` the prior group returned, exactly
    mirroring the original function's sequential in-place ``X = fe_append_columns(X, ...)`` flow.
    """
    kwargs = dict(
        self=self,
        y=y,
        verbose=verbose,
        _y_np=_y_np,
        _hybrid_orth_pre_recipes=_hybrid_orth_pre_recipes,
        _gbm_seeded_triplet_names=_gbm_seeded_triplet_names,
        _fe_family_on=_fe_family_on,
    )
    X = _hybrid_orth_family_variants_group1(X=X, **kwargs)
    X = _hybrid_orth_family_variants_group2(X=X, **kwargs)
    X = _hybrid_orth_family_variants_group3(X=X, **kwargs)
    X = _hybrid_orth_family_variants_group4(X=X, **kwargs)
    return X
