"""Stage helpers of ``evaluate_swap_candidate`` (``_dcd_swap``): aggregate fit, relevance estimation, member scoring, member branch and aggregate branch.

Each stage reads/writes the shared ``st`` namespace and returns the finished ``SwapDecision`` on an early exit, else ``None`` (the last stage always
returns one). The helpers that tests call or patch (``_select_swap_method_auto``, ``_evaluate_swap_cand_*``) are looked up on ``_dcd_swap`` at call time.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from . import _dcd_swap as _sw
from ._dcd_metrics import _binarize_aggregate
from ._dcd_swap_matrix import _resolve_member_matrix

if TYPE_CHECKING:
    from . import SwapDecision

logger = logging.getLogger("mlframe.feature_selection.filters._dynamic_cluster_discovery._dcd_swap")


def _run_member_null(state: Any, anchor: int, target: Any, S_minus_anchor: list, member_idx: int, member_rel: float, B_: int) -> float:
    """Delegate to ``_dcd_swap_null.run_member_null`` for a p-value on the member candidate's relevance."""
    # The B-permutation null is parallelized across cores in ``_dcd_swap_null.run_member_null``:
    # it pre-generates all B shuffles of ONLY the member column (SAME rng sequence -> bit-identical
    # permutation multiset / p-value), then ``prange``s the per-draw conditional MI over a thread-local
    # shuffled column against the precomputed (Z)/(Y,Z) class labelings - NO frame copy, NO mutate-restore
    # on the parallel path (Z and Y are read read-only from state.factors_data, the shuffled X is passed
    # directly). Tiny B falls back to the exact serial mutate-and-restore path. H(Z)/H(Y,Z) are hoisted once
    # (permutation-invariant); see _dcd_swap_null docstring.
    from ._dcd_swap_null import run_member_null
    return run_member_null(
        state=state, member_idx=int(member_idx), member_rel=float(member_rel), B_=int(B_),
        anchor=int(anchor), target=target, S_minus_anchor=S_minus_anchor, logger=logger,
    )


def stage_fit_aggregate(st: Any, state: Any, anchor: int, X_raw: Any, target_y: Optional[np.ndarray]) -> Optional[SwapDecision]:
    """Resolve the cluster members, standardise them and fit the aggregate representative; a ``SwapDecision`` when the swap is already ruled out."""
    from . import SwapDecision

    st.cluster = state.cluster_anchors.get(anchor, set())
    if len(st.cluster) < max(int(state.min_cluster_size), int(state.cluster_size_threshold)):
        return SwapDecision(accept=False)
    members = [anchor, *sorted(st.cluster)]
    if X_raw is None:
        X_raw = state.X_raw_ref
    if X_raw is None:
        return SwapDecision(accept=False)
    cols = state.cols
    if cols is None or len(cols) <= max(members):
        return SwapDecision(accept=False)
    try:
        from .._cluster_aggregate import _standardize_align, _derive_weights
    except Exception as exc:
        logger.warning("DCD swap: failed to import cluster_aggregate helpers: %r", exc)
        return SwapDecision(accept=False)
    try:
        member_names = [cols[m] for m in members]
        resolved = _resolve_member_matrix(X_raw, members, member_names)
        if resolved is None:
            return SwapDecision(accept=False)
        M, members, member_names = resolved
        Z, st.mean, st.std, st.signs = _standardize_align(M, ref_col=0)
        # Layer 43 (PART B): ``auto`` swap method.
        # Run a K-fold (n_folds=5) OOF MI bake-off over the three linear-
        # combiner methods and pick the per-cluster winner. The chosen method
        # is recorded in the recipe + swap_log so replay uses it bit-identically
        # (no y at transform time). K-fold scores are cached on
        # state._auto_method_cache keyed by tuple(member_names) so successive
        # re-evaluations of the same cluster reuse the bake-off.
        if str(state.swap_method) == "auto":
            st.chosen_method, st.kfold_scores = _sw._select_swap_method_auto(
                state=state, Z=Z, target_y=target_y,
                member_names=tuple(member_names),
            )
        else:
            st.chosen_method = str(state.swap_method)
            st.kfold_scores = None
        # Layer 44: route all non-linear / row-reduction methods through the
        # shared ``_apply_method_nonlinear`` (median / median_z / signed_max_abs
        # / signed_l2_sum). Linear methods stay on the ``Z @ weights`` fast path.
        from .._cluster_aggregate import (
            _apply_method_nonlinear, _NONLINEAR_METHODS,
        )
        if st.chosen_method in _NONLINEAR_METHODS:
            st.weights = None
            st.rep_continuous = _apply_method_nonlinear(Z, st.chosen_method).astype(float)
        else:
            st.weights = _derive_weights(Z, st.chosen_method)
            if st.weights is None:
                return SwapDecision(accept=False)
            st.rep_continuous = (Z @ st.weights).astype(float)
        # Bin the rep via quantile/uniform to integer codes.
        st.rep_binned = _binarize_aggregate(
            st.rep_continuous, method=state.quantization_method,
            n_bins=state.quantization_nbins, dtype=state.quantization_dtype,
        )
    except Exception as exc:
        logger.warning("DCD swap: PC1 fit failed: %r", exc)
        return SwapDecision(accept=False)
    st.members, st.cols = members, cols
    return None


def stage_relevance(st: Any, state: Any, anchor: int, selected_vars: list, target_y: Optional[np.ndarray], entropy_cache: Optional[dict]) -> Optional[SwapDecision]:
    """Build the candidate matrix with the representative appended and estimate the representative's / anchor's conditional relevance."""
    from . import SwapDecision

    # Build a candidate matrix with rep appended.
    assert state.factors_data is not None, "DCD swap candidate scoring requires a populated DCDState.factors_data"
    st.new_col_idx = int(state.factors_data.shape[1])
    st.data_with_rep = np.column_stack([state.factors_data, st.rep_binned])
    st.nbins_with_rep = np.concatenate([
        np.asarray(state.factors_nbins, dtype=np.int64),
        [int(st.rep_binned.max()) + 1 if st.rep_binned.size else int(state.quantization_nbins)],
    ])
    # Relevance comparison: conditional MI against Selected − {anchor}.
    target = state.target_indices if state.target_indices is not None and state.target_indices.size > 0 else target_y
    if target is None:
        return SwapDecision(accept=False)
    # Build conditioning set Selected − {anchor}.
    S_minus_anchor = [int(s) for s in selected_vars if int(s) != int(anchor)]
    try:
        from ..info_theory import mi, conditional_mi
        if S_minus_anchor:
            st.rep_relevance = float(conditional_mi(
                factors_data=st.data_with_rep,
                x=np.array([st.new_col_idx], dtype=np.int64),
                y=np.asarray(target, dtype=np.int64),
                z=np.array(S_minus_anchor, dtype=np.int64),
                var_is_nominal=None,
                factors_nbins=st.nbins_with_rep,
                entropy_cache=entropy_cache,
                can_use_x_cache=False, can_use_y_cache=True,
            ))
            st.anchor_rel = float(conditional_mi(
                factors_data=state.factors_data,
                x=np.array([int(anchor)], dtype=np.int64),
                y=np.asarray(target, dtype=np.int64),
                z=np.array(S_minus_anchor, dtype=np.int64),
                var_is_nominal=None,
                factors_nbins=state.factors_nbins,
                entropy_cache=entropy_cache,
                can_use_x_cache=False, can_use_y_cache=True,
            ))
        else:
            # First-selected case: use unconditional MI.
            st.rep_relevance = float(mi(
                st.data_with_rep, np.array([st.new_col_idx], dtype=np.int64),
                np.asarray(target, dtype=np.int64), st.nbins_with_rep,
            ))
            st.anchor_rel = float(mi(
                state.factors_data, np.array([int(anchor)], dtype=np.int64),
                np.asarray(target, dtype=np.int64), state.factors_nbins,
            ))
    except Exception as exc:
        logger.warning("DCD swap: relevance estimation failed: %r", exc)
        return SwapDecision(accept=False)
    st.target, st.S_minus_anchor, st.conditional_mi, st.mi = target, S_minus_anchor, conditional_mi, mi
    return None


def stage_member_scoring(st: Any, state: Any, entropy_cache: Optional[dict], full_npermutations: int) -> Optional[SwapDecision]:
    """Score every cluster member (anchor refinement) and apply the pre-filter gates; branch A (no swap) is returned here."""
    from . import SwapDecision

    target, S_minus_anchor = st.target, st.S_minus_anchor
    # Layer 45: ANCHOR REFINEMENT. Pre-fix the decision was a
    # binary gate (aggregate vs anchor). When the FIRST-picked feature
    # was a high-MI noisy spike rather than the cluster's center, the
    # aggregate ended up sign-aligned to a sub-optimal reference and the
    # swap either fired weakly or never fired even though a sibling
    # member carried strictly more information about y.
    #
    # The refinement: score each cluster member with the SAME conditional
    # MI(member; y | Selected − anchor) used for the anchor. Pick the
    # best member; if it dominates the anchor by the swap gain, it's a
    # viable swap target on its own. The final branch is whichever of
    # ``aggregate`` / ``best_member`` has the higher CMI - both have to
    # individually beat the anchor by ``swap_gain_threshold``.
    st.member_relevances = {}
    st.best_member_idx = -1
    st.best_member_rel = float("-inf")
    st.target_arr = np.asarray(target, dtype=np.int64)
    st._sorted_cluster = sorted(st.cluster)
    # BATCHED: one parallel call scoring every cluster member (bounded by max_cluster_size,
    # default 12) against the SAME (y, Selected-anchor) instead of a per-member Python loop - see
    # ``_dcd_member_rank_batch.py`` for why the public ``conditional_mi_batched_dispatch``/``_cpu_cmi_loop``
    # dispatchers are NOT safe here (their default hoisted fast path silently mis-handles a multi-column Z)
    # and why the one real call site's ``entropy_cache`` was never actually live to lose. Any failure
    # (shape mismatch, degenerate cardinality, ...) falls back to the exact original per-member loop with
    # its own per-member fail-to-0.0 semantics, unchanged.
    st.best_member_idx, st.best_member_rel = _sw._evaluate_swap_cand_its_own_per_member(state, st._sorted_cluster, st.target_arr, S_minus_anchor, st.member_relevances, st.best_member_rel, entropy_cache, st.best_member_idx)
    st.gain_factor = 1.0 + float(state.swap_gain_threshold)
    # Cheap pre-filter ahead of the (potentially expensive) permutation null below. When a null
    # will actually run (``full_npermutations > 0``), soften this to plain dominance - the
    # candidate merely needs to beat the anchor's point estimate - and let the null's p-value be
    # the SOLE significance arbiter: the fixed ``swap_gain_threshold`` margin was rejecting real,
    # null-eligible swaps whose point-estimate gain fell just under the bar (measured case:
    # aggregate beat anchor by 3.3%, short of the 5% default margin, despite the underlying
    # improvement being real). When no null is requested (``full_npermutations <= 0``, so no
    # statistical backstop is available), keep the stricter margin as the sole heuristic gate.
    st._null_will_run = int(full_npermutations or 0) > 0
    if st._null_will_run:
        st.aggregate_gate = st.rep_relevance > st.anchor_rel
        st.member_gate = st.best_member_idx >= 0 and st.best_member_rel > st.anchor_rel
    else:
        st.aggregate_gate = st.rep_relevance > st.anchor_rel * st.gain_factor
        st.member_gate = st.best_member_idx >= 0 and st.best_member_rel > st.anchor_rel * st.gain_factor
    if not st.aggregate_gate and not st.member_gate:
        # Branch A: no swap candidate beats the anchor.
        return SwapDecision(
            accept=False,
            rep_relevance=st.rep_relevance,
            anchor_relevance_in_ctx=st.anchor_rel,
            branch="none",
            member_col_idx=st.best_member_idx,
            member_relevance=(st.best_member_rel if st.best_member_idx >= 0 else 0.0),
        )
    return None


def stage_member_branch(st: Any, state: Any, anchor: int, full_npermutations: int) -> Optional[SwapDecision]:
    """Pick the higher-CMI branch and run branch B (member swap, with its permutation null) when it wins."""
    from . import SwapDecision

    target, S_minus_anchor = st.target, st.S_minus_anchor
    # When the caller requested a permutation null (``full_npermutations > 0``), apply the SAME null to the member
    # candidate too. The point-CMI gate is upward-biased on small/noisy data; if the swap is firing on pure noise
    # the null catches it for the aggregate path - the member branch must not be a side door that bypasses it.
    # Both gates active -> pick the higher CMI as the candidate branch.
    # When only one is active, that one wins by definition.
    st.prefer_aggregate = st.aggregate_gate and (not st.member_gate or st.rep_relevance >= st.best_member_rel)
    # resolve the effective
    # permutation-null draw count. ``full_npermutations`` (the screening
    # confidence, default 3) acts ONLY as the on/off switch - 0 means the
    # caller opted out of every null. When a null is requested we source the
    # actual count from ``state.swap_npermutations`` and auto-raise it to
    # ceil(1/swap_alpha) so 1/(B+1) < swap_alpha holds; otherwise the gate is
    # arithmetically un-passable (B=3 -> min-p 0.25 >> 0.05) and every swap is
    # silently rejected. Both the aggregate and member nulls use this B_eff.
    st.B_eff = _sw._evaluate_swap_cand_silently_rejected_both_aggregate(full_npermutations, state)
    if not st.prefer_aggregate:
        # Branch B: member swap. Apply permutation null when requested
        # (B>0) - otherwise this branch silently bypasses the check the
        # caller asked for on the swap as a whole.
        member_p = _run_member_null(state, int(anchor), target, S_minus_anchor, int(st.best_member_idx), float(st.best_member_rel), st.B_eff)
        if st.B_eff > 0 and member_p >= float(state.swap_alpha):
            return SwapDecision(
                accept=False,
                rep_relevance=st.rep_relevance,
                anchor_relevance_in_ctx=st.anchor_rel,
                perm_p_value=member_p,
                branch="none",
                member_col_idx=int(st.best_member_idx),
                member_relevance=float(st.best_member_rel),
            )
        return SwapDecision(
            accept=True,
            new_col_idx=int(st.best_member_idx),
            aggregate_name="",
            binned_rep=None,
            new_nbins=0,
            recipe_obj=None,
            rep_relevance=st.rep_relevance,
            anchor_relevance_in_ctx=st.anchor_rel,
            perm_p_value=member_p,
            branch="member",
            member_col_idx=int(st.best_member_idx),
            member_relevance=float(st.best_member_rel),
        )
    return None


def stage_aggregate_branch(st: Any, state: Any, anchor: int) -> SwapDecision:
    """Branch C: the aggregate must still pass the permutation null, with the member swap as the fallback; always returns the final ``SwapDecision``."""
    from . import SwapDecision

    target, S_minus_anchor, conditional_mi, mi = st.target, st.S_minus_anchor, st.conditional_mi, st.mi
    members, cols = st.members, st.cols
    # Branch C continues - aggregate must still pass the permutation null.
    st.deterministic_gate = st.aggregate_gate
    # Permutation null on rep.
    # The deterministic point-MI gate is upward-biased on small / noisy data
    # because rep (a continuous PC1 projection re-binned with quantization_nbins
    # bins, often > anchor's bin count) gets more degrees of freedom than the
    # raw anchor. Without a null check, swap accepts spurious aggregates on
    # pure noise. Plan v2 B-2 mandated this null but it was never implemented.
    #
    # Null hypothesis: rep_binned has no real conditional dependence on y
    # given Selected\{anchor}. Reject when observed rep_relevance lies in
    # the upper tail of the shuffled-rep distribution.
    st.perm_p_value = 0.0
    st.B = st.B_eff
    st.perm_p_value = _sw._evaluate_swap_cand_upper_tail_shuffled_rep(st.B, state, anchor, target, S_minus_anchor, st.data_with_rep, st.nbins_with_rep, st.new_col_idx, st.rep_binned, st.rep_relevance, conditional_mi, mi, st.perm_p_value)
    st.accept = st.deterministic_gate and (st.B <= 0 or st.perm_p_value < float(state.swap_alpha))
    if not st.accept:
        # Layer 45: aggregate failed its permutation null. If the
        # best-member gate also held (member_gate True), fall through to
        # the member-swap branch - but apply the SAME permutation null
        # the caller requested via ``full_npermutations``. The earlier
        # bypass made the member branch a side door
        # past the null on pure-noise candidates.
        if st.member_gate and st.best_member_idx >= 0:
            member_p2 = _run_member_null(state, int(anchor), target, S_minus_anchor, int(st.best_member_idx), float(st.best_member_rel), st.B_eff)
            if st.B_eff > 0 and member_p2 >= float(state.swap_alpha):
                return SwapDecision(
                    accept=False,
                    rep_relevance=st.rep_relevance,
                    anchor_relevance_in_ctx=st.anchor_rel,
                    perm_p_value=max(st.perm_p_value, member_p2),
                    branch="none",
                    member_col_idx=int(st.best_member_idx),
                    member_relevance=float(st.best_member_rel),
                )
            return SwapDecision(
                accept=True,
                new_col_idx=int(st.best_member_idx),
                aggregate_name="",
                binned_rep=None,
                new_nbins=0,
                recipe_obj=None,
                rep_relevance=st.rep_relevance,
                anchor_relevance_in_ctx=st.anchor_rel,
                perm_p_value=member_p2,
                branch="member",
                member_col_idx=int(st.best_member_idx),
                member_relevance=float(st.best_member_rel),
            )
        return SwapDecision(
            accept=False,
            rep_relevance=st.rep_relevance,
            anchor_relevance_in_ctx=st.anchor_rel,
            perm_p_value=st.perm_p_value,
            branch="none",
            member_col_idx=st.best_member_idx,
            member_relevance=(st.best_member_rel if st.best_member_idx >= 0 else 0.0),
        )
    aggregate_name = f"_dcd_pc1_{'_'.join(str(cols[m])[:6] for m in members[:3])}" f"_n{len(members)}_a{anchor}"
    # Layer 43 (PART B): record the chosen method (the actual
    # combiner used to build ``rep_continuous``) in the recipe, NOT the user-
    # facing ``state.swap_method`` string. When ``auto`` was active, the
    # chosen method is the K-fold OOF winner; when a specific method was
    # pinned, chosen_method == state.swap_method. Replay reads recipe.method
    # so the transform-time aggregate is bit-identical with fit.
    st.recipe_obj = {
        "method": st.chosen_method, "members": members,
        "mean": st.mean.tolist(), "std": st.std.tolist(),
        "signs": st.signs.tolist(),
    }
    if st.weights is not None:
        st.recipe_obj["weights"] = st.weights.tolist()
    if st.kfold_scores is not None:
        st.recipe_obj["kfold_scores"] = {k: float(v) for k, v in st.kfold_scores.items()}
        st.recipe_obj["auto_winner"] = st.chosen_method
    return SwapDecision(
        accept=True,
        new_col_idx=st.new_col_idx,
        aggregate_name=aggregate_name,
        binned_rep=st.rep_binned,
        new_nbins=int(st.nbins_with_rep[-1]),
        recipe_obj=st.recipe_obj,
        rep_relevance=st.rep_relevance,
        anchor_relevance_in_ctx=st.anchor_rel,
        perm_p_value=st.perm_p_value,
        branch="aggregate",
        member_col_idx=st.best_member_idx,
        member_relevance=(st.best_member_rel if st.best_member_idx >= 0 else 0.0),
        rep_continuous=st.rep_continuous,
    )
