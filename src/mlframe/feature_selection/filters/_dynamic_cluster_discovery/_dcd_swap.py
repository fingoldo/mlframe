"""DCD anchor->aggregate swap machinery (carved from ``_dynamic_cluster_discovery.py``).

Holds the K-fold OOF method bake-off (``_select_swap_method_auto``), the swap-acceptance gate
(``evaluate_swap_candidate``), and ``commit_swap`` which threads the engineered-recipes host dict.
The aggregate / info-theory / recipe helpers are lazy-imported in-body as in the original;
``_binarize_aggregate`` comes from the metrics sibling and ``SwapDecision`` is lazy-imported from
the parent inside ``evaluate_swap_candidate`` to avoid an import cycle.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from ._dcd_metrics import _binarize_aggregate

if TYPE_CHECKING:
    from . import DCDState, SwapDecision

logger = logging.getLogger(__name__)


# combiner shapes to pick from. The original 3 cover homogeneous linear
# averaging; the 4 additions extend the menu with:
#   - ``pca_pc2``: secondary principal component (correlated multi-latent
#     clusters where PC1 leaves shared structure on the table)
#   - ``median_z``: row-robust to outlier members
#   - ``signed_max_abs``: surfaces the loudest single member's signal
#   - ``signed_l2_sum``: signed quadratic combiner
_AUTO_METHOD_CANDIDATES = (
    "mean_z", "mean_inv_var", "pca_pc1",
    "pca_pc2", "median_z", "signed_max_abs", "signed_l2_sum",
)


def _select_swap_method_auto(
    *,
    state: DCDState,
    Z: np.ndarray,
    target_y,
    member_names: tuple,
    n_folds: int = 5,
) -> tuple:
    """K-fold OOF MI bake-off over the three linear-combiner methods.

    For each method in ``_AUTO_METHOD_CANDIDATES``:
      1. Split rows into ``n_folds`` folds (deterministic by hash of member_names).
      2. For each fold: derive weights on the train rows of Z, project the
         held-out rows to a scalar aggregate, bin via the quantization recipe.
      3. Compute MI(aggregate_oof; y_held_out) on the held-out rows.
      4. Mean the per-fold MIs.
    Returns ``(winner_method, scores_dict)`` where ``scores_dict`` maps every
    method to its mean OOF MI. Caches under ``state._auto_method_cache`` keyed
    by ``member_names`` for cheap re-evaluation.
    """
    from .._cluster_aggregate import _derive_weights
    # Cache lookup -- same cluster, same bake-off result.
    cache = getattr(state, "_auto_method_cache", None)
    if cache is None:
        cache = {}
        state._auto_method_cache = cache  # type: ignore[attr-defined]
    cache_key = tuple(member_names)
    cached = cache.get(cache_key)
    if cached is not None:
        return tuple(cached)
    # Resolve target -- prefer state's target index column.
    y_arr = None
    if state.target_indices is not None and state.target_indices.size > 0 and state.factors_data is not None:
        try:
            y_arr = np.asarray(state.factors_data[:, int(state.target_indices[0])], dtype=np.int64)
        except Exception:
            y_arr = None
    if y_arr is None and target_y is not None:
        y_arr = np.asarray(target_y, dtype=np.int64).ravel()
    if y_arr is None or y_arr.size != Z.shape[0]:
        # Cannot K-fold without a per-row target -> fall back to pca_pc1.
        result: tuple = ("pca_pc1", {})
        cache[cache_key] = result
        return result

    n_samples = Z.shape[0]
    n_folds_eff = max(2, min(int(n_folds), n_samples))
    # Deterministic shuffle seeded by member_names so repeat runs are stable.
    seed_material = abs(hash(cache_key)) & 0xFFFFFFFF
    rng = np.random.default_rng(int(seed_material))
    perm = rng.permutation(n_samples)
    fold_sizes = np.full(n_folds_eff, n_samples // n_folds_eff, dtype=np.int64)
    fold_sizes[: n_samples % n_folds_eff] += 1
    fold_bounds = np.concatenate([[0], np.cumsum(fold_sizes)])

    # Layer 50 (2026-05-31): loop folds-outer / methods-inner with a per-fold
    # SVD cache. Pre-fix the loop was methods-outer; 4 of the 7 candidates
    # (``mean_inv_var``, ``pca_pc1``, ``pca_pc2``, ``factor_score``) each
    # re-SVD'd the SAME Z_train independently -- 4x redundant SVD work per
    # fold per cluster. Profile on p=200/n=5000/10 latents attributed
    # 0.444s tottime / 150 calls to ``np.linalg.svd`` alone; with the cache
    # the 4 SVDs collapse to 1 (~4x reduction on the SVD line item, ~2x
    # reduction on the auto-bake-off cumtime). Bit-equivalent: every method
    # consumes the SAME vt[0] / vt[1] / Zc / communalities arrays it would
    # have computed independently; the cache just hands back the precomputed
    # result instead of redoing it.
    scores: dict[str, list] = {m: [] for m in _AUTO_METHOD_CANDIDATES}
    # Per-fold Z_train cache slot (reset on every new fold). The dict is
    # reused as a sentinel: cleared at the top of each fold so methods
    # within the fold see the same SVD; methods across folds get a fresh
    # cache (Z_train differs by row partition).
    svd_cache: dict = {}
    for f in range(n_folds_eff):
        test_idx = perm[fold_bounds[f] : fold_bounds[f + 1]]
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_idx] = False
        if train_mask.sum() < 3 or test_idx.size < 2:
            continue
        Z_train = Z[train_mask]
        Z_test = Z[test_idx]
        svd_cache.clear()  # fresh per-fold SVD slot.
        for method in _AUTO_METHOD_CANDIDATES:
            try:
                w = _derive_weights(Z_train, method, svd_cache=svd_cache)
                if w is None:
                    # Layer 44: non-linear / row-reduction combiners (median /
                    # median_z / signed_max_abs / signed_l2_sum) have no weight
                    # vector. Apply the same row-reducer the recipe will use at
                    # replay so the K-fold MI estimate matches the production
                    # path.
                    from .._cluster_aggregate import (
                        _apply_method_nonlinear, _NONLINEAR_METHODS,
                    )
                    if method not in _NONLINEAR_METHODS:
                        continue
                    rep_test = _apply_method_nonlinear(Z_test, method)
                else:
                    rep_test = Z_test @ np.asarray(w, dtype=np.float64)
                rep_test = np.nan_to_num(rep_test, nan=0.0, posinf=0.0, neginf=0.0)
                # Bin rep_test with the recipe's quantization (uses test-fold
                # edges -- cheap, fold-local).
                rep_binned = _binarize_aggregate(
                    rep_test, method=state.quantization_method,
                    n_bins=state.quantization_nbins, dtype=state.quantization_dtype,
                )
                y_test = y_arr[test_idx]
                # MI(rep_binned; y_test). Use the mlframe info_theory.mi with
                # a 2-col data block (rep_binned, y_test).
                from ..info_theory import mi as _mi_func
                _data = np.column_stack([
                    rep_binned.astype(np.int64), y_test.astype(np.int64),
                ])
                _nb_rep = int(rep_binned.max()) + 1 if rep_binned.size else int(state.quantization_nbins)
                _nb_y = int(y_test.max()) + 1 if y_test.size else 2
                _nbins_arr = np.array([_nb_rep, _nb_y], dtype=np.int64)
                _mi_val = float(_mi_func(
                    _data, np.array([0], dtype=np.int64),
                    np.array([1], dtype=np.int64), _nbins_arr,
                ))
                scores[method].append(_mi_val)
            except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
                logger.debug("suppressed in _dcd_swap.py:158: %s", e)
                continue

    # Reduce per-method per-fold lists to mean OOF MI; tie-break by candidate
    # order (mean_z first). Bit-equivalent to the pre-Layer-50 flow: same
    # MI values, same averaging, same tie-break -- the only change was the
    # loop nesting order to enable per-fold SVD caching.
    mean_scores = {m: (float(np.mean(v)) if v else 0.0) for m, v in scores.items()}
    winner = max(_AUTO_METHOD_CANDIDATES, key=lambda m: (mean_scores.get(m, 0.0), -_AUTO_METHOD_CANDIDATES.index(m)))
    result = (winner, mean_scores)
    cache[cache_key] = result
    return result

def evaluate_swap_candidate(
    state: DCDState,
    anchor: int,
    selected_vars: list,
    *,
    X_raw=None,
    target_y=None,
    factors_data=None,
    factors_nbins=None,
    cached_MIs: Optional[dict] = None,
    entropy_cache: Optional[dict] = None,
    full_npermutations: int = 0,
) -> SwapDecision:
    """Decide whether (and how) to swap ``anchor`` for a better cluster
    representative.

    Returns a SwapDecision with ``accept=False`` if either:
      - the cluster is below ``min_cluster_size`` members
      - aggregate fit fails (degenerate / NaN variance)
      - neither the candidate aggregate nor the best cluster member's
        conditional relevance ``I(. ; y | Selected − anchor)`` exceeds
        anchor's conditional relevance × ``(1 + swap_gain_threshold)``

    Layer 45 (2026-05-31): three exclusive branches are evaluated:

      A. No swap — anchor's CMI already dominates the cluster; ``branch="none"``.
      B. Member swap — a cluster member's CMI dominates the anchor's AND
         the candidate aggregate's. The anchor index in ``selected_vars`` is
         simply replaced by the member index (no new aggregate column);
         ``branch="member"``.
      C. Aggregate swap — the aggregate's CMI dominates both the anchor's
         and every member's. Existing behaviour; ``branch="aggregate"``.

    The branch with the highest CMI wins (tie-break order: aggregate >
    member > none, because aggregate already passed the explicit
    permutation null when present). Both the aggregate AND the member
    branch run the permutation null when one is requested (B > 0): the
    Wave 9.1 iter-3 follow-up closed the member-branch side door that
    previously bypassed it. The null's draw count is sourced from
    ``state.swap_npermutations`` (NOT the screening-confidence
    ``full_npermutations``, which only acts as the on/off switch); see the
    ``swap_npermutations`` field comment for why decoupling is mandatory.

    No mutation occurs until ``commit_swap`` is called with the returned
    decision (Critic1/B-2 pre-confirmation guarantee).
    """
    # ``SwapDecision`` lives in the DCD parent; lazy-import to avoid the parent<->swap import cycle.
    from . import SwapDecision
    cluster = state.cluster_anchors.get(anchor, set())
    if len(cluster) < max(int(state.min_cluster_size), int(state.cluster_size_threshold)):
        return SwapDecision(accept=False)
    members = [anchor, *sorted(cluster)]
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
        # Resolve raw columns; X_raw may be a DataFrame or ndarray.
        if hasattr(X_raw, "columns"):
            present = [c for c in member_names if c in X_raw.columns]
            if len(present) < 2:
                return SwapDecision(accept=False)
            M = X_raw[present].to_numpy(dtype=np.float64, copy=True)
        else:
            arr = np.asarray(X_raw)
            if arr.ndim != 2 or arr.shape[1] < max(members) + 1:
                return SwapDecision(accept=False)
            M = arr[:, members].astype(np.float64, copy=True)
        # Drop columns containing NaN / Inf to keep PC1 stable.
        finite_mask = np.isfinite(M).all(axis=0)
        if finite_mask.sum() < 2:
            return SwapDecision(accept=False)
        M = M[:, finite_mask]
        Z, mean, std, signs = _standardize_align(M, ref_col=0)
        # 2026-05-31 Layer 43 (PART B): ``auto`` swap method.
        # Run a K-fold (n_folds=5) OOF MI bake-off over the three linear-
        # combiner methods and pick the per-cluster winner. The chosen method
        # is recorded in the recipe + swap_log so replay uses it bit-identically
        # (no y at transform time). K-fold scores are cached on
        # state._auto_method_cache keyed by tuple(member_names) so successive
        # re-evaluations of the same cluster reuse the bake-off.
        if str(state.swap_method) == "auto":
            chosen_method, kfold_scores = _select_swap_method_auto(
                state=state, Z=Z, target_y=target_y,
                member_names=tuple(member_names),
            )
        else:
            chosen_method = str(state.swap_method)
            kfold_scores = None
        # Layer 44: route all non-linear / row-reduction methods through the
        # shared ``_apply_method_nonlinear`` (median / median_z / signed_max_abs
        # / signed_l2_sum). Linear methods stay on the ``Z @ weights`` fast path.
        from .._cluster_aggregate import (
            _apply_method_nonlinear, _NONLINEAR_METHODS,
        )
        if chosen_method in _NONLINEAR_METHODS:
            weights = None
            rep_continuous = _apply_method_nonlinear(Z, chosen_method).astype(np.float64)
        else:
            weights = _derive_weights(Z, chosen_method)
            if weights is None:
                return SwapDecision(accept=False)
            rep_continuous = (Z @ weights).astype(np.float64)
        # Bin the rep via quantile/uniform to integer codes.
        rep_binned = _binarize_aggregate(
            rep_continuous, method=state.quantization_method,
            n_bins=state.quantization_nbins, dtype=state.quantization_dtype,
        )
    except Exception as exc:
        logger.warning("DCD swap: PC1 fit failed: %r", exc)
        return SwapDecision(accept=False)
    # Build a candidate matrix with rep appended.
    assert state.factors_data is not None, "DCD swap candidate scoring requires a populated DCDState.factors_data"
    new_col_idx = int(state.factors_data.shape[1])
    data_with_rep = np.column_stack([state.factors_data, rep_binned])
    nbins_with_rep = np.concatenate([
        np.asarray(state.factors_nbins, dtype=np.int64),
        [int(rep_binned.max()) + 1 if rep_binned.size else int(state.quantization_nbins)],
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
            rep_relevance = float(conditional_mi(
                factors_data=data_with_rep,
                x=np.array([new_col_idx], dtype=np.int64),
                y=np.asarray(target, dtype=np.int64),
                z=np.array(S_minus_anchor, dtype=np.int64),
                var_is_nominal=None,
                factors_nbins=nbins_with_rep,
                entropy_cache=entropy_cache,
                can_use_x_cache=False, can_use_y_cache=True,
            ))
            anchor_rel = float(conditional_mi(
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
            rep_relevance = float(mi(
                data_with_rep, np.array([new_col_idx], dtype=np.int64),
                np.asarray(target, dtype=np.int64), nbins_with_rep,
            ))
            anchor_rel = float(mi(
                state.factors_data, np.array([int(anchor)], dtype=np.int64),
                np.asarray(target, dtype=np.int64), state.factors_nbins,
            ))
    except Exception as exc:
        logger.warning("DCD swap: relevance estimation failed: %r", exc)
        return SwapDecision(accept=False)
    # 2026-05-31 Layer 45: ANCHOR REFINEMENT. Pre-fix the decision was a
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
    # ``aggregate`` / ``best_member`` has the higher CMI -- both have to
    # individually beat the anchor by ``swap_gain_threshold``.
    from ..info_theory import mi as _mi_func, conditional_mi as _cmi_func
    member_relevances: dict = {}
    best_member_idx = -1
    best_member_rel = float("-inf")
    target_arr = np.asarray(target, dtype=np.int64)
    _sorted_cluster = sorted(cluster)
    # BATCHED (2026-07-13): one parallel call scoring every cluster member (bounded by max_cluster_size,
    # default 12) against the SAME (y, Selected-anchor) instead of a per-member Python loop -- see
    # ``_dcd_member_rank_batch.py`` for why the public ``conditional_mi_batched_dispatch``/``_cpu_cmi_loop``
    # dispatchers are NOT safe here (their default hoisted fast path silently mis-handles a multi-column Z)
    # and why the one real call site's ``entropy_cache`` was never actually live to lose. Any failure
    # (shape mismatch, degenerate cardinality, ...) falls back to the exact original per-member loop with
    # its own per-member fail-to-0.0 semantics, unchanged.
    try:
        from ._dcd_member_rank_batch import batched_member_relevance
        _rels = batched_member_relevance(
            state.factors_data, np.asarray(_sorted_cluster, dtype=np.int64), target_arr,
            list(S_minus_anchor), np.asarray(state.factors_nbins, dtype=np.int64),
        )
        for _k, m_idx in enumerate(_sorted_cluster):
            m_rel = float(_rels[_k])
            member_relevances[int(m_idx)] = m_rel
            if m_rel > best_member_rel:
                best_member_rel = m_rel
                best_member_idx = int(m_idx)
    except Exception:
        member_relevances = {}
        best_member_idx = -1
        best_member_rel = float("-inf")
        for m_idx in _sorted_cluster:
            try:
                if S_minus_anchor:
                    m_rel = float(_cmi_func(
                        factors_data=state.factors_data,
                        x=np.array([int(m_idx)], dtype=np.int64),
                        y=target_arr,
                        z=np.array(S_minus_anchor, dtype=np.int64),
                        var_is_nominal=None,
                        factors_nbins=state.factors_nbins,
                        entropy_cache=entropy_cache,
                        can_use_x_cache=False, can_use_y_cache=True,
                    ))
                else:
                    m_rel = float(_mi_func(
                        state.factors_data,
                        np.array([int(m_idx)], dtype=np.int64),
                        target_arr, state.factors_nbins,
                    ))
            except Exception:
                m_rel = 0.0
            member_relevances[int(m_idx)] = m_rel
            if m_rel > best_member_rel:
                best_member_rel = m_rel
                best_member_idx = int(m_idx)
    gain_factor = 1.0 + float(state.swap_gain_threshold)
    aggregate_gate = rep_relevance > anchor_rel * gain_factor
    member_gate = best_member_idx >= 0 and best_member_rel > anchor_rel * gain_factor
    if not aggregate_gate and not member_gate:
        # Branch A: no swap candidate beats the anchor.
        return SwapDecision(
            accept=False,
            rep_relevance=rep_relevance,
            anchor_relevance_in_ctx=anchor_rel,
            branch="none",
            member_col_idx=best_member_idx,
            member_relevance=(best_member_rel if best_member_idx >= 0 else 0.0),
        )
    # Both gates active -> pick the higher CMI as the candidate branch.
    # When only one is active, that one wins by definition.
    prefer_aggregate = aggregate_gate and (not member_gate or rep_relevance >= best_member_rel)
    # 2026-06-03 (audit dcd-core-1 / dcd-swap-null-1/2): resolve the effective
    # permutation-null draw count. ``full_npermutations`` (the screening
    # confidence, default 3) acts ONLY as the on/off switch -- 0 means the
    # caller opted out of every null. When a null is requested we source the
    # actual count from ``state.swap_npermutations`` and auto-raise it to
    # ceil(1/swap_alpha) so 1/(B+1) < swap_alpha holds; otherwise the gate is
    # arithmetically un-passable (B=3 -> min-p 0.25 >> 0.05) and every swap is
    # silently rejected. Both the aggregate and member nulls use this B_eff.
    if int(full_npermutations or 0) <= 0:
        B_eff = 0
    else:
        B_eff = int(getattr(state, "swap_npermutations", 199) or 0)
        if B_eff <= 0:
            B_eff = int(full_npermutations)
        _swap_alpha = float(state.swap_alpha)
        if _swap_alpha > 0.0:
            _min_B = int(np.ceil(1.0 / _swap_alpha))  # 1/(B_eff+1) < swap_alpha
            if B_eff < _min_B:
                B_eff = _min_B
    # Wave 9.1 iter-3 follow-up: when the caller requested a permutation
    # null (``full_npermutations > 0``), apply the SAME null to the member
    # candidate too. The point-CMI gate is upward-biased on small/noisy
    # data; if the swap is firing on pure noise the null catches it for
    # the aggregate path -- the member branch must not be a side door
    # that bypasses the same check.
    def _run_member_null(member_idx: int, member_rel: float, B_: int) -> float:
        """Delegate to ``_dcd_swap_null.run_member_null`` for a p-value on the member candidate's relevance, closing over ``state``/``anchor``/``target``/``S_minus_anchor``/``logger`` so both call sites below need only pass the member-specific args."""
        # The B-permutation null is parallelized across cores in ``_dcd_swap_null.run_member_null``:
        # it pre-generates all B shuffles of ONLY the member column (SAME rng sequence -> bit-identical
        # permutation multiset / p-value), then ``prange``s the per-draw conditional MI over a thread-local
        # shuffled column against the precomputed (Z)/(Y,Z) class labelings -- NO frame copy, NO mutate-restore
        # on the parallel path (Z and Y are read read-only from state.factors_data, the shuffled X is passed
        # directly). Tiny B falls back to the exact serial mutate-and-restore path. H(Z)/H(Y,Z) are hoisted once
        # (permutation-invariant); see _dcd_swap_null docstring.
        from ._dcd_swap_null import run_member_null
        return run_member_null(
            state=state, member_idx=int(member_idx), member_rel=float(member_rel), B_=int(B_),
            anchor=int(anchor), target=target, S_minus_anchor=S_minus_anchor, logger=logger,
        )
    if not prefer_aggregate:
        # Branch B: member swap. Apply permutation null when requested
        # (B>0) -- otherwise this branch silently bypasses the check the
        # caller asked for on the swap as a whole.
        member_p = _run_member_null(int(best_member_idx), float(best_member_rel), B_eff)
        if B_eff > 0 and member_p >= float(state.swap_alpha):
            return SwapDecision(
                accept=False,
                rep_relevance=rep_relevance,
                anchor_relevance_in_ctx=anchor_rel,
                perm_p_value=member_p,
                branch="none",
                member_col_idx=int(best_member_idx),
                member_relevance=float(best_member_rel),
            )
        return SwapDecision(
            accept=True,
            new_col_idx=int(best_member_idx),
            aggregate_name="",
            binned_rep=None,
            new_nbins=0,
            recipe_obj=None,
            rep_relevance=rep_relevance,
            anchor_relevance_in_ctx=anchor_rel,
            perm_p_value=member_p,
            branch="member",
            member_col_idx=int(best_member_idx),
            member_relevance=float(best_member_rel),
        )
    # Branch C continues -- aggregate must still pass the permutation null.
    deterministic_gate = aggregate_gate
    # 2026-05-30 Wave 9.1 fix (loop iter 3): permutation null on rep.
    # The deterministic point-MI gate is upward-biased on small / noisy data
    # because rep (a continuous PC1 projection re-binned with quantization_nbins
    # bins, often > anchor's bin count) gets more degrees of freedom than the
    # raw anchor. Without a null check, swap accepts spurious aggregates on
    # pure noise. Plan v2 B-2 mandated this null but it was never implemented.
    #
    # Null hypothesis: rep_binned has no real conditional dependence on y
    # given Selected\{anchor}. Reject when observed rep_relevance lies in
    # the upper tail of the shuffled-rep distribution.
    perm_p_value = 0.0
    B = B_eff
    if B > 0:
        try:
            rng = np.random.default_rng(int(getattr(state, "_perm_seed", 0)) + int(anchor))
            # Persist rolling seed so successive swaps don't reuse the same null draws.
            state._perm_seed = int(getattr(state, "_perm_seed", 0)) + B + 1
            target_arr = np.asarray(target, dtype=np.int64)
            n_exceed = 0
            # Hoist permutation-invariant H(Z) + H(Y,Z) (only the appended rep
            # column is shuffled, so y/z are fixed across draws). Bit-identical
            # by construction; see _run_member_null + bench_dcd_swap_null_entropy_hoist.py.
            # Also keeps the dense Z/YZ (or Y) CLASS LABEL arrays -- not just their entropies -- the
            # batched-prange path below needs (previously discarded via ``_, _fz_a, _ = ...``).
            _h_z_a = -1.0
            _h_yz_a = -1.0
            _h_x_a = -1.0
            _h_y_a = -1.0
            _z_classes_a = _z_nclasses_a = None
            _yz_classes_a = _yz_nclasses_a = None
            _y_classes_a = _y_nclasses_a = None
            from ..info_theory import entropy as _entropy_a, merge_vars as _merge_a
            if S_minus_anchor:
                _z_arr_a = np.sort(np.array(S_minus_anchor, dtype=np.int64))
                _z_classes_a, _fz_a, _z_nclasses_a = _merge_a(data_with_rep, _z_arr_a, None, np.asarray(nbins_with_rep, dtype=np.int64))
                _h_z_a = float(_entropy_a(_fz_a))
                _yz_arr_a = np.sort(np.concatenate([target_arr, _z_arr_a]))
                _yz_classes_a, _fyz_a, _yz_nclasses_a = _merge_a(data_with_rep, _yz_arr_a, None, np.asarray(nbins_with_rep, dtype=np.int64))
                _h_yz_a = float(_entropy_a(_fyz_a))
            else:
                # No-Z: I(rep; y). H(rep) is permutation-invariant (shuffle preserves the marginal) -> hoist it too.
                _, _fx_a, _ = _merge_a(data_with_rep, np.array([new_col_idx], dtype=np.int64), None, np.asarray(nbins_with_rep, dtype=np.int64))
                _h_x_a = float(_entropy_a(_fx_a))
                _y_classes_a, _fy_a, _y_nclasses_a = _merge_a(data_with_rep, np.sort(target_arr), None, np.asarray(nbins_with_rep, dtype=np.int64))
                _h_y_a = float(_entropy_a(_fy_a))

            # BATCHED (2026-07-13): the MEMBER branch's null (``_run_member_null`` ->
            # ``_dcd_swap_null.run_member_null``) was already parallelised via ``prange`` over B
            # pre-generated shuffles against precomputed Z/YZ class labelings; this AGGREGATE branch
            # ran the equivalent B-loop serially, one ``conditional_mi``/``mi`` call per draw, because it
            # discarded the class arrays those kernels need (see the hoist above). Reuse the SAME
            # ``_dcd_swap_null`` kernels here instead of writing a new one. Tiny B keeps the exact serial
            # mutate-free loop (prange spawn not worth it below ``_PARALLEL_MIN_B``, mirrors the member path).
            from ._dcd_swap_null import _PARALLEL_MIN_B, _member_null_cmi_prange, _member_null_mi_prange
            if B >= _PARALLEL_MIN_B:
                # Pre-generate all B shuffles of the rep column SERIALLY with the SAME rng sequence the
                # (still-present, tiny-B) serial path below uses -- bit-identical permutation multiset --
                # then ``prange`` the per-draw (C)MI over a thread-local shuffled column. No frame copy: Y/Z
                # are read read-only from ``data_with_rep`` (already built once above), only the shuffled
                # rep column is passed directly into the kernel.
                n_rows = rep_binned.shape[0]
                base_col = np.asarray(rep_binned, dtype=np.int64)
                shuffles = np.empty((B, n_rows), dtype=np.int64)
                for _b in range(B):
                    s = base_col.copy()
                    rng.shuffle(s)
                    shuffles[_b] = s
                nb_x = int(nbins_with_rep[new_col_idx])
                if S_minus_anchor:
                    # Statically guaranteed non-None here: the ``if S_minus_anchor:`` hoist branch above
                    # (mirroring this same condition) always populates these before this point is reached.
                    assert _z_classes_a is not None and _z_nclasses_a is not None
                    assert _yz_classes_a is not None and _yz_nclasses_a is not None
                    n_exceed = int(_member_null_cmi_prange(
                        shuffles, nb_x, _z_classes_a.astype(np.int64), int(_z_nclasses_a),
                        _yz_classes_a.astype(np.int64), int(_yz_nclasses_a), _h_z_a, _h_yz_a, float(rep_relevance),
                    ))
                else:
                    assert _y_classes_a is not None and _y_nclasses_a is not None
                    n_exceed = int(_member_null_mi_prange(
                        shuffles, nb_x, _y_classes_a.astype(np.int64), int(_y_nclasses_a),
                        _h_x_a, _h_y_a, float(rep_relevance),
                    ))
            else:
                data_with_rep_perm = data_with_rep.copy()
                for _ in range(B):
                    rep_shuffled = rep_binned.copy()
                    rng.shuffle(rep_shuffled)
                    data_with_rep_perm[:, new_col_idx] = rep_shuffled
                    if S_minus_anchor:
                        null_rel = float(conditional_mi(
                            factors_data=data_with_rep_perm,
                            x=np.array([new_col_idx], dtype=np.int64),
                            y=target_arr,
                            z=np.array(S_minus_anchor, dtype=np.int64),
                            var_is_nominal=None,
                            factors_nbins=nbins_with_rep,
                            entropy_z=_h_z_a, entropy_yz=_h_yz_a,
                            entropy_cache=None,
                            can_use_x_cache=False, can_use_y_cache=False,
                        ))
                    else:
                        null_rel = float(mi(
                            data_with_rep_perm, np.array([new_col_idx], dtype=np.int64),
                            target_arr, nbins_with_rep,
                        ))
                    if null_rel >= rep_relevance:
                        n_exceed += 1
            perm_p_value = (n_exceed + 1) / (B + 1)
        except Exception as exc:
            logger.warning("DCD swap: permutation null failed (B=%s): %r", B, exc)
            perm_p_value = 1.0  # conservative: fail closed
    accept = deterministic_gate and (B <= 0 or perm_p_value < float(state.swap_alpha))
    if not accept:
        # Layer 45: aggregate failed its permutation null. If the
        # best-member gate also held (member_gate True), fall through to
        # the member-swap branch -- but apply the SAME permutation null
        # the caller requested via ``full_npermutations``. Wave 9.1 iter-3
        # follow-up: the prior bypass made the member branch a side door
        # past the null on pure-noise candidates.
        if member_gate and best_member_idx >= 0:
            member_p2 = _run_member_null(int(best_member_idx), float(best_member_rel), B_eff)
            if B_eff > 0 and member_p2 >= float(state.swap_alpha):
                return SwapDecision(
                    accept=False,
                    rep_relevance=rep_relevance,
                    anchor_relevance_in_ctx=anchor_rel,
                    perm_p_value=max(perm_p_value, member_p2),
                    branch="none",
                    member_col_idx=int(best_member_idx),
                    member_relevance=float(best_member_rel),
                )
            return SwapDecision(
                accept=True,
                new_col_idx=int(best_member_idx),
                aggregate_name="",
                binned_rep=None,
                new_nbins=0,
                recipe_obj=None,
                rep_relevance=rep_relevance,
                anchor_relevance_in_ctx=anchor_rel,
                perm_p_value=member_p2,
                branch="member",
                member_col_idx=int(best_member_idx),
                member_relevance=float(best_member_rel),
            )
        return SwapDecision(
            accept=False,
            rep_relevance=rep_relevance,
            anchor_relevance_in_ctx=anchor_rel,
            perm_p_value=perm_p_value,
            branch="none",
            member_col_idx=best_member_idx,
            member_relevance=(best_member_rel if best_member_idx >= 0 else 0.0),
        )
    aggregate_name = f"_dcd_pc1_{'_'.join(str(cols[m])[:6] for m in members[:3])}" f"_n{len(members)}_a{anchor}"
    # 2026-05-31 Layer 43 (PART B): record the chosen method (the actual
    # combiner used to build ``rep_continuous``) in the recipe, NOT the user-
    # facing ``state.swap_method`` string. When ``auto`` was active, the
    # chosen method is the K-fold OOF winner; when a specific method was
    # pinned, chosen_method == state.swap_method. Replay reads recipe.method
    # so the transform-time aggregate is bit-identical with fit.
    recipe_obj = {
        "method": chosen_method, "members": members,
        "mean": mean.tolist(), "std": std.tolist(),
        "signs": signs.tolist(),
    }
    if weights is not None:
        recipe_obj["weights"] = weights.tolist()
    if kfold_scores is not None:
        recipe_obj["kfold_scores"] = {k: float(v) for k, v in kfold_scores.items()}
        recipe_obj["auto_winner"] = chosen_method
    return SwapDecision(
        accept=True,
        new_col_idx=new_col_idx,
        aggregate_name=aggregate_name,
        binned_rep=rep_binned,
        new_nbins=int(nbins_with_rep[-1]),
        recipe_obj=recipe_obj,
        rep_relevance=rep_relevance,
        anchor_relevance_in_ctx=anchor_rel,
        perm_p_value=perm_p_value,
        branch="aggregate",
        member_col_idx=best_member_idx,
        member_relevance=(best_member_rel if best_member_idx >= 0 else 0.0),
        rep_continuous=rep_continuous,
    )

def commit_swap(
    state: DCDState,
    anchor: int,
    decision: SwapDecision,
    *,
    selected_vars: list,
    data_ref: dict,
    engineered_recipes: Optional[dict] = None,
    predictors_log: Optional[list] = None,
) -> int:
    """Atomic mutation of state, the host MRMR data structures, and the
    cluster bookkeeping. ``data_ref`` is a dict containing references the
    caller wants updated: keys ``data``, ``cols``, ``nbins`` map to the
    np.ndarray / list / np.ndarray objects to be replaced in-place via
    re-assignment to the SAME dict (caller reads them back).
    """
    if not decision.accept:
        return -1
    new_idx = int(decision.new_col_idx)
    # 2026-05-31 Layer 45: member-swap branch. The decision points at an
    # existing factors_data column (a cluster member), not a new aggregate.
    # No matrix extension, no recipe, no pool_pruned_mask resize. We
    # simply unprune the chosen member (discover_cluster_members had
    # pruned every member when it grew the cluster), replace the anchor
    # index in selected_vars, and reseat the cluster bookkeeping under
    # the member as the new anchor. The rest of the pipeline -- bin counts,
    # transform replay, support_ resolution -- already trusts the column.
    is_member_swap = getattr(decision, "branch", "aggregate") == "member" and not decision.aggregate_name and decision.binned_rep is None
    if is_member_swap:
        member_idx = new_idx
        # Replace anchor in selected_vars with the chosen member.
        try:
            pos = selected_vars.index(int(anchor))
            selected_vars[pos] = member_idx
        except ValueError:
            selected_vars.append(member_idx)
        # Move cluster ownership to the new anchor; drop the chosen
        # member from its own membership set (it IS the anchor now).
        cluster_members = state.cluster_anchors.pop(int(anchor), set())
        new_member_set = {int(m) for m in cluster_members if int(m) != member_idx}
        # Add old anchor as a member of the new cluster (it's still
        # SU-redundant with the new anchor and should stay pruned).
        new_member_set.add(int(anchor))
        state.cluster_anchors[member_idx] = new_member_set
        for m in new_member_set:
            state.member_to_anchor[int(m)] = member_idx
        # Mark old anchor as pruned (it's now a member); unprune the new
        # anchor (it must be eligible for downstream confirm/select).
        assert state.pool_pruned_mask is not None, "DCD swap application requires a populated DCDState.pool_pruned_mask"
        if int(anchor) < state.pool_pruned_mask.shape[0]:
            state.pool_pruned_mask[int(anchor)] = True
        if 0 <= member_idx < state.pool_pruned_mask.shape[0]:
            state.pool_pruned_mask[member_idx] = False
        # Remove the new anchor from member_to_anchor since it is itself
        # the anchor now (anchors aren't tracked there).
        state.member_to_anchor.pop(member_idx, None)
        if predictors_log is not None:
            predictors_log.append(
                {
                    "dcd_swap": True,
                    "dcd_swap_branch": "member",
                    "anchor": int(anchor),
                    "new_col_idx": member_idx,
                    "aggregate_name": "",
                    "n_members": len(new_member_set),
                }
            )
        state.swap_log.append(
            {
                "anchor": int(anchor),
                "new_col_idx": member_idx,
                "aggregate_name": "",
                "n_members": len(new_member_set),
                "rep_relevance": float(decision.rep_relevance),
                "anchor_relevance_in_ctx": float(decision.anchor_relevance_in_ctx),
                "branch": "member",
                "member_relevance": float(decision.member_relevance),
            }
        )
        return member_idx
    # Branch C below: aggregate swap (existing behaviour).
    # 1. Extend matrix.
    assert state.factors_data is not None, "DCD swap application requires a populated DCDState.factors_data"
    assert decision.binned_rep is not None, "aggregate branch requires the accepted decision to carry its binned representative"
    new_data = np.column_stack([state.factors_data, decision.binned_rep])
    assert state.cols is not None, "DCD swap application requires a populated DCDState.cols"
    new_cols = [*list(state.cols), str(decision.aggregate_name)]
    new_nbins = np.concatenate([
        np.asarray(state.factors_nbins, dtype=np.int64),
        [int(decision.new_nbins)],
    ])
    state.factors_data = new_data
    state.cols = new_cols
    state.factors_nbins = new_nbins
    # 2026-06-03 (audit dcd-core-2): invalidate the int64 view of factors_nbins
    # cached by pair_su / pair_su_batch / pair_vi. It is built lazily at the
    # PRE-swap length and then reused while IGNORING the passed factors_nbins
    # (lines 451-454 / 511-514 / 559-562 / 654-657); without this reset, any
    # pair_su on the new aggregate column index would index the stale, too-short
    # array and raise IndexError (re-raised as a fit-aborting RuntimeError at
    # _screen_predictors.py). Invalidating the derived cache where the source
    # mutates is the runtime-cache contract. The next call rebuilds it at the
    # new length.
    state._fn_arr_cached = None
    # Expand pool_pruned_mask to match (new column is implicitly NOT pruned).
    state.pool_pruned_mask = np.concatenate([
        state.pool_pruned_mask, np.zeros(1, dtype=bool),
    ])
    # 2. Update caller's data references.
    if data_ref is not None:
        data_ref["data"] = new_data
        data_ref["cols"] = new_cols
        data_ref["nbins"] = new_nbins
    # 3. Replace anchor in selected_vars with new aggregate col idx.
    try:
        pos = selected_vars.index(int(anchor))
        selected_vars[pos] = new_idx
    except ValueError:
        # Anchor not in selected_vars (interaction-tuple case); append.
        selected_vars.append(new_idx)
    # 4. Move cluster_anchors[anchor] under new key; mark anchor pruned.
    cluster_members = state.cluster_anchors.pop(int(anchor), set())
    state.cluster_anchors[new_idx] = cluster_members
    for m in cluster_members:
        state.member_to_anchor[int(m)] = new_idx
    if int(anchor) < state.pool_pruned_mask.shape[0]:
        state.pool_pruned_mask[int(anchor)] = True
    # 5. Persist recipe + log.
    # 2026-05-31 Layer 43 (PART A): when an ``engineered_recipes`` dict is
    # supplied (host MRMR's name->EngineeredRecipe map), upgrade the swap
    # ``recipe_obj`` (a plain dict carrying method/members/mean/std/signs/
    # weights) to a frozen ``EngineeredRecipe`` of kind ``cluster_aggregate``
    # so the ``_mrmr_fit_impl`` remap routes it into ``self._engineered_recipes_``
    # and ``get_feature_names_out`` / ``transform`` replay the PC1 aggregate
    # column. Pre-fix this stored a raw dict and the remap dropped the
    # aggregate from ``support_``/output silently.
    if engineered_recipes is not None and decision.aggregate_name:
        _recipe = decision.recipe_obj
        if isinstance(_recipe, dict):
            try:
                from ..engineered_recipes import build_cluster_aggregate_recipe

                _src_names = tuple(str(state.cols[m]) if 0 <= m < len(state.cols) else f"col_{m}" for m in _recipe.get("members", []))
                _quant = {
                    "nbins": int(state.quantization_nbins),
                    "method": str(state.quantization_method),
                    "dtype": np.dtype(state.quantization_dtype).str,
                }
                # Persist fit-time bin edges so ``_apply_cluster_aggregate``
                # uses identical edges at transform-time (no re-quantile drift).
                _bin = np.asarray(decision.binned_rep)
                if _bin.size > 0:
                    n_bins_eff = int(_bin.max()) + 1
                    _q_arr = np.linspace(0.0, 100.0, n_bins_eff + 1)
                    # Derive edges from the continuous rep consistent with the fit-time binning.
                    try:
                        if decision.rep_continuous is not None:
                            # evaluate_swap_candidate already built the standardized + sign-aligned
                            # aggregate (Z / rep_continuous) that these edges must match -- reuse it
                            # directly instead of re-deriving M/Z from X_raw + the recipe's stored
                            # mean/std/signs (own comment below).
                            _cont = np.nan_to_num(
                                np.asarray(decision.rep_continuous, dtype=np.float64),
                                copy=False, nan=0.0, posinf=0.0, neginf=0.0,
                            )
                            _edges = np.nanpercentile(_cont, _q_arr)
                            _quant["edges"] = _edges.tolist()
                        elif state.X_raw_ref is not None and hasattr(state.X_raw_ref, "columns"):
                            # Fallback: reconstruct the standardized + sign-aligned matrix the swap
                            # was evaluated against. X_raw_ref + src_names are the canonical sources.
                            _M = state.X_raw_ref[list(_src_names)].to_numpy(
                                dtype=np.float64, copy=True,
                            )
                            _mean = np.asarray(_recipe["mean"], dtype=np.float64)
                            _std_raw = np.asarray(_recipe["std"], dtype=np.float64)
                            _std = np.where(_std_raw > 0.0, _std_raw, 1.0)
                            _signs = np.asarray(_recipe["signs"], dtype=np.float64)
                            _Z = ((_M - _mean) / _std) * _signs
                            # Layer 44: non-linear / row-reduction methods
                            # (median / median_z / signed_max_abs / signed_l2_sum)
                            # rebuild via the shared reducer; linear methods
                            # stay on ``Z @ weights``.
                            from .._cluster_aggregate import (
                                _apply_method_nonlinear, _NONLINEAR_METHODS,
                            )
                            _m = _recipe.get("method")
                            if _m in _NONLINEAR_METHODS:
                                _cont = _apply_method_nonlinear(_Z, _m)
                            else:
                                _cont = _Z @ np.asarray(_recipe["weights"], dtype=np.float64)
                            _cont = np.nan_to_num(
                                _cont, copy=False, nan=0.0, posinf=0.0, neginf=0.0,
                            )
                            _edges = np.nanpercentile(_cont, _q_arr)
                            _quant["edges"] = _edges.tolist()
                    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                        logger.debug("suppressed in _dcd_swap.py:785: %s", e)
                        pass
                engineered_recipe = build_cluster_aggregate_recipe(
                    name=str(decision.aggregate_name),
                    src_names=_src_names,
                    method=str(_recipe.get("method", "pca_pc1")),
                    member_mean=np.asarray(_recipe["mean"], dtype=np.float64),
                    member_std=np.asarray(_recipe["std"], dtype=np.float64),
                    signs=np.asarray(_recipe["signs"], dtype=np.float64),
                    weights=(np.asarray(_recipe["weights"], dtype=np.float64) if "weights" in _recipe else None),
                    quantization=_quant,
                )
                engineered_recipes[decision.aggregate_name] = engineered_recipe
            except Exception as _build_exc:
                logger.warning(
                    "DCD commit_swap: failed to build EngineeredRecipe " "(falling back to dict): %r",
                    _build_exc,
                )
                engineered_recipes[decision.aggregate_name] = _recipe
        else:
            engineered_recipes[decision.aggregate_name] = _recipe
    if predictors_log is not None:
        predictors_log.append({
            "dcd_swap": True,
            "anchor": int(anchor),
            "new_col_idx": new_idx,
            "aggregate_name": decision.aggregate_name,
            "n_members": len(cluster_members),
        })
    # 2026-05-31 Layer 43 (PART B): record the chosen swap method (and any
    # K-fold OOF bake-off scores when ``auto`` ran) in the swap_log entry so
    # downstream callers can audit per-cluster method selection.
    _swap_log_entry = {
        "anchor": int(anchor),
        "new_col_idx": new_idx,
        "aggregate_name": decision.aggregate_name,
        "n_members": len(cluster_members),
        "rep_relevance": float(decision.rep_relevance),
        "anchor_relevance_in_ctx": float(decision.anchor_relevance_in_ctx),
        # Layer 45: branch discriminator on every swap_log entry. Aggregate
        # path here; member-swap path emits "branch":"member" above.
        "branch": "aggregate",
        "member_relevance": float(decision.member_relevance),
    }
    if isinstance(decision.recipe_obj, dict):
        _method = decision.recipe_obj.get("method")
        if _method is not None:
            _swap_log_entry["method"] = str(_method)
        _kfold = decision.recipe_obj.get("kfold_scores")
        if _kfold is not None:
            _swap_log_entry["kfold_scores"] = {k: float(v) for k, v in _kfold.items()}
        _winner = decision.recipe_obj.get("auto_winner")
        if _winner is not None:
            _swap_log_entry["auto_winner"] = str(_winner)
    state.swap_log.append(_swap_log_entry)
    return new_idx
