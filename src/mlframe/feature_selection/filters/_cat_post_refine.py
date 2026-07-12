"""Post-screen refinement of cat-interaction pairs for ``cat_interactions``.

Split out of ``cat_interactions.py`` to keep the parent below the 1k-line
monolith threshold. Behaviour preserved bit-for-bit; the parent re-exports
the public-looking entries so the orchestrator in
``run_cat_interaction_step`` continues to call them via the same names.

What lives here:
  - ``_bootstrap_ii_cis`` (per-pair bootstrap CIs on the II statistic)
  - ``_anti_redundancy_rerank`` (drop pairs whose II is already explained
    by an already-kept feature)
  - ``_kfold_stability_filter`` (II stability across K folds)
  - ``_refine_kway_coordinate_ascent`` (one-pass coordinate refinement
    of k-way memberships)
"""
from __future__ import annotations

import bisect
import logging

import numpy as np

from .cat_fe_state import CatFEConfig
from .info_theory import compute_mi_from_classes, merge_vars
# ``_materialize_pairs`` / ``_select_top_k_pairs`` /
# ``resolve_min_interaction_information`` live in ``cat_interactions`` itself
# (via the kway-materialize sibling re-export); imported lazily inside the
# function bodies that need them so the
# ``cat_interactions -> _cat_post_refine -> cat_interactions`` import cycle
# stays broken.

logger = logging.getLogger(__name__)


def _build_merge_prefix_states(
    factors_data: np.ndarray,
    sorted_members: list,
    factors_nbins: np.ndarray,
    dtype,
    final_state: tuple | None = None,
) -> list:
    """Incremental ``merge_vars`` states after merging ``sorted_members[:i]`` (ascending order), for every ``i`` in ``0..len(sorted_members)``.

    ``merge_vars``'s dense renumbering is ORDER-SENSITIVE: merging the same variable set in a different order yields a bijective but numerically DIFFERENT
    ``final_classes`` encoding (verified empirically -- only the count of distinct classes is order-invariant, not the labels). A candidate variable inserted
    at some position ``pos`` among ``sorted_members`` therefore needs the merge to walk ``sorted_members[:pos] + [cand] + sorted_members[pos:]`` to stay
    bit-identical to a fresh ``merge_vars`` over the fully re-sorted tuple -- these prefix states let ``_merge_vars_sorted_insert`` splice a candidate in at
    its correct sorted position without re-scanning the members before it, for every candidate sharing that same insertion point.

    ``final_state`` lets the caller hand in an already-computed ``(classes, nclasses)`` for the FULL ``sorted_members`` merge (e.g. the parent state carried
    from the previous round) instead of re-deriving it here.
    """
    n_rows = factors_data.shape[0]
    states = [(np.zeros(n_rows, dtype=dtype), 1)]
    upto = len(sorted_members) - 1 if final_state is not None else len(sorted_members)
    for i in range(1, upto + 1):
        prev_classes, prev_nclasses = states[-1]
        classes_i, _freqs_i, nclasses_i = merge_vars(
            factors_data=factors_data,
            vars_indices=np.array([sorted_members[i - 1]], dtype=np.int64),
            var_is_nominal=None, factors_nbins=factors_nbins,
            current_nclasses=prev_nclasses, final_classes=prev_classes.copy(), dtype=dtype,
        )
        states.append((classes_i, nclasses_i))
    if final_state is not None:
        states.append(final_state)
    return states


def _merge_vars_sorted_insert(
    factors_data: np.ndarray,
    prefix_states: list,
    sorted_members: list,
    cand_int: int,
    factors_nbins: np.ndarray,
    dtype,
) -> tuple:
    """``merge_vars`` over ``sorted(sorted_members + [cand_int])``, splicing ``cand_int`` into its correct sorted position via ``prefix_states`` instead of
    re-scanning the members before it. Bit-identical to a fresh full-tuple ``merge_vars`` call (verified end-to-end against the pre-fix algorithm across
    randomized trials incl. min/mid/max insertion positions and varying arities/cardinalities/row-counts -- see
    ``_benchmarks/bench_kway_coord_ascent_frozen_prefix.py``)."""
    ins = bisect.bisect_left(sorted_members, cand_int)
    prefix_classes, prefix_nclasses = prefix_states[ins]
    classes1, freqs1, nclasses1 = merge_vars(
        factors_data=factors_data, vars_indices=np.array([cand_int], dtype=np.int64),
        var_is_nominal=None, factors_nbins=factors_nbins,
        current_nclasses=prefix_nclasses, final_classes=prefix_classes.copy(), dtype=dtype,
    )
    suffix = sorted_members[ins:]
    if not suffix:
        return classes1, freqs1, nclasses1
    classes2, freqs2, nclasses2 = merge_vars(
        factors_data=factors_data, vars_indices=np.array(suffix, dtype=np.int64),
        var_is_nominal=None, factors_nbins=factors_nbins,
        current_nclasses=nclasses1, final_classes=classes1, dtype=dtype,
    )
    return classes2, freqs2, nclasses2


# ============================================================================
# Bootstrap CIs on II
#
# For each top-K survivor, draw ``n_replicates`` subsamples (size ``sample_frac * n``), recompute II per replicate, return (lower, median, upper) CI. Complements
# permutation tests: perm checks significance vs null; bootstrap checks STABILITY under sample variation.
# ============================================================================


def _bootstrap_ii_cis(
    factors_data: np.ndarray,
    pairs_a: np.ndarray,
    pairs_b: np.ndarray,
    selected_idx: np.ndarray,
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    cfg: CatFEConfig,
    dtype,
    verbose: int,
) -> dict:
    """For each pair in ``selected_idx``, compute bootstrap CI on II. Returns ``{(i, j): (lower, median, upper)}`` per ``cfg.bootstrap_ci_alpha``.

    Cost: ``n_replicates * top_k * O(n)`` -- at n_replicates=20, top_k=32, n=10000 that's ~6.4M merge_vars-equivalents.
    Heavy; gated by user opt-in (``bootstrap_ci_n_replicates > 0``).
    """
    if cfg.bootstrap_ci_n_replicates <= 0 or len(selected_idx) == 0:
        return {}
    n_samples = factors_data.shape[0]
    n_rep = int(cfg.bootstrap_ci_n_replicates)
    sub_size = max(int(n_samples * cfg.bootstrap_sample_frac), cfg.min_n_samples)
    alpha = float(cfg.bootstrap_ci_alpha)
    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0

    if verbose:
        logger.info(
            "cat-FE bootstrap CIs: %d replicates x %d survivors "
            "(subsample size %d)",
            n_rep, len(selected_idx), sub_size,
        )

    ci_dict: dict = {}
    for k in selected_idx:
        i = int(pairs_a[k])
        jj = int(pairs_b[k])
        ii_replicates = np.empty(n_rep, dtype=np.float64)
        for r in range(n_rep):
            rng = np.random.default_rng(seed=int(jj) * 65537 + r)
            idx = rng.choice(n_samples, size=sub_size, replace=True)
            sub_data = factors_data[idx]
            sub_cls_y = classes_y[idx]
            # Recompute marginals + joint MI on the subsample.
            cls_a_s, fq_a_s, _ = merge_vars(
                factors_data=sub_data,
                vars_indices=np.array([i], dtype=np.int64),
                var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
            )
            cls_b_s, fq_b_s, _ = merge_vars(
                factors_data=sub_data,
                vars_indices=np.array([jj], dtype=np.int64),
                var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
            )
            cls_pair_s, fq_pair_s, _ = merge_vars(
                factors_data=sub_data,
                vars_indices=np.array([i, jj], dtype=np.int64),
                var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
            )
            # Need fq_y on the subsample; rebuild from sub_cls_y
            sub_nbins_y = int(sub_cls_y.max()) + 1 if sub_cls_y.size else 1
            sub_fq_y = np.bincount(
                sub_cls_y.astype(np.int64), minlength=sub_nbins_y,
            ).astype(np.float64) / max(sub_size, 1)
            i_a = compute_mi_from_classes(cls_a_s, fq_a_s, sub_cls_y, sub_fq_y, dtype)
            i_b = compute_mi_from_classes(cls_b_s, fq_b_s, sub_cls_y, sub_fq_y, dtype)
            i_pair = compute_mi_from_classes(cls_pair_s, fq_pair_s, sub_cls_y, sub_fq_y, dtype)
            ii_replicates[r] = i_pair - i_a - i_b
        ci_dict[(i, jj)] = (
            float(np.quantile(ii_replicates, lower_q)),
            float(np.median(ii_replicates)),
            float(np.quantile(ii_replicates, upper_q)),
        )
    return ci_dict


# ============================================================================
# Anti-redundancy with selected features
#
# Pure-II ranking treats a pair (X1, X2) as relevant if it has high synergy with Y. But MRMR's overall objective is `relevance - β*redundancy`: a pair whose merged
# column is HIGHLY CORRELATED with an already-selected feature Z adds little new information regardless of its II. We down-weight II by `β * max_z I(merged; Z)`
# over already-selected Z.
#
# Two-stage decoupled design:
# 1. II floor gates "is this engineered col worth materializing?" (already done by ``_select_top_k_pairs``).
# 2. mRMR-style score = II - β * mean_z I(merged; Z) re-ranks survivors here. β=0 disables (default), preserving pure II.
#
# Cost: per survivor + per selected Z, one merge_vars + one compute_mi_from_classes. At top_k=64 survivors and |Z|=20 selected, that's 1280 merge_vars calls; linear in both.
# ============================================================================


def _anti_redundancy_rerank(
    factors_data: np.ndarray,
    pairs_a: np.ndarray,
    pairs_b: np.ndarray,
    selected_idx: np.ndarray,
    ii_arr: np.ndarray,
    nbins: np.ndarray,
    selected_so_far: list,  # column indices (in ``data``) of already-selected features
    classes_y: np.ndarray,  # unused -- the redundancy MI is against Z, not Y
    cfg: CatFEConfig,
    dtype,
    verbose: int,
) -> tuple:
    """Re-rank top-K survivors by ``score = II - β * mean_z I(merged; Z)``.

    When ``cfg.anti_redundancy_beta == 0`` or ``selected_so_far`` is empty, this is a no-op. Returns ``(scored_arr, selected_idx_reordered)``.
    """
    if cfg.anti_redundancy_beta <= 0 or not selected_so_far or len(selected_idx) == 0:
        return ii_arr, selected_idx

    beta = float(cfg.anti_redundancy_beta)
    if verbose:
        logger.info(
            "cat-FE: anti-redundancy re-rank with beta=%.3f over %d survivor(s) "
            "vs %d already-selected feature(s)",
            beta, len(selected_idx), len(selected_so_far),
        )

    scored = ii_arr.copy()
    selected_so_far_arr = np.asarray(selected_so_far, dtype=np.int64)
    # Each already-selected feature ``z``'s single-column merge depends only on ``z``, not on the outer pair
    # ``k`` -- precompute once and reuse across every survivor pair instead of recomputing per (k, z).
    z_merge_cache = {
        int(z): merge_vars(
            factors_data=factors_data,
            vars_indices=np.array([int(z)], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )[:2]
        for z in selected_so_far_arr
    }
    for k in selected_idx:
        i = int(pairs_a[k])
        j = int(pairs_b[k])
        # Materialise the merged class for this pair
        cls_merged, freqs_merged, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=np.array([i, j], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        red_terms = []
        for z in selected_so_far_arr:
            cls_z, freqs_z = z_merge_cache[int(z)]
            mi_to_z = compute_mi_from_classes(
                classes_x=cls_merged, freqs_x=freqs_merged,
                classes_y=cls_z, freqs_y=freqs_z, dtype=dtype,
            )
            red_terms.append(mi_to_z)
        # mRMR-style: subtract β * mean redundancy (mean over selected Z).
        # ``mean`` per Peng-Ding-Long 2005; max would also work but mean
        # is the canonical mRMR formulation.
        mean_red = sum(red_terms) / len(red_terms)
        scored[k] = ii_arr[k] - beta * mean_red

    # Re-sort selected_idx by the corrected scores
    if cfg.select_on == "synergy":
        order = np.argsort(-scored[selected_idx])
    elif cfg.select_on == "redundancy":
        order = np.argsort(scored[selected_idx])
    else:
        order = np.argsort(-np.abs(scored[selected_idx]))
    return scored, selected_idx[order]


# ============================================================================
# K-fold II stability filter
#
# A pair with II=0.3 on one fold and II=-0.1 on another is noise, not signal. Split the training data into K folds, recompute II on each fold's slice, and keep only
# pairs prevalent in >= floor·K folds. Guards against "II was driven by a few outlier rows" failures.
#
# Cost: K-1 extra pair searches (each fold uses ~n*(K-1)/K rows). Runs BEFORE the heavy permutation phase so the per-fold II costs don't multiply against npermutations.
# ============================================================================


def _kfold_stability_filter(
    factors_data: np.ndarray,
    pairs_a: np.ndarray,
    pairs_b: np.ndarray,
    selected_idx: np.ndarray,
    nbins: np.ndarray,
    target_indices: np.ndarray,
    cfg: CatFEConfig,
    dtype,
    verbose: int,
) -> tuple:
    """For each top-K survivor, recompute II on K disjoint folds; keep pairs whose II clears the floor on >= ``min_fold_prevalence * K`` folds. Returns
    ``(kept_selected_idx, per_fold_ii_dict)``. No-op (returns inputs unchanged) when ``cfg.n_folds_stability <= 0``.

    Determinism: folds are derived from ``np.arange(n) % K`` -- no shuffling, no RNG. Reproducible across runs.
    """
    if cfg.n_folds_stability <= 0 or len(selected_idx) == 0:
        return selected_idx, {}

    from .cat_interactions import resolve_min_interaction_information  # lazy: import-cycle, see module top

    n_samples = factors_data.shape[0]
    K = int(cfg.n_folds_stability)
    floor = resolve_min_interaction_information(cfg, n_samples)

    fold_ids = np.arange(n_samples) % K
    per_fold_ii: dict = {}
    kept = []

    if verbose:
        logger.info("cat-FE: K-fold stability check (K=%d) over %d top-K pair(s)", K, len(selected_idx))

    # ``slice_data`` and the target's fold-local merge depend only on ``f``, not on the outer survivor ``k`` --
    # precompute once per fold and reuse across every survivor instead of re-merging Y on the same fold slice
    # once per (k, f) pair. ``None`` marks a too-small fold (mirrors the original per-(k,f) "-inf" skip).
    fold_cache: dict = {}
    for f in range(K):
        mask = fold_ids == f
        n_fold = int(mask.sum())
        if n_fold < cfg.min_n_samples // 2:
            fold_cache[f] = None
            continue
        slice_data = factors_data[mask]
        cls_y_f, fq_y_f, _ = merge_vars(
            factors_data=slice_data, vars_indices=target_indices,
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        fold_cache[f] = (slice_data, cls_y_f, fq_y_f)

    for k in selected_idx:
        i = int(pairs_a[k])
        j = int(pairs_b[k])
        fold_ii_vals: list = []

        for f in range(K):
            cached = fold_cache[f]
            if cached is None:
                # Fold too small to estimate MI reliably; mark as failed.
                fold_ii_vals.append(float("-inf"))
                continue
            slice_data, cls_y_f, fq_y_f = cached
            cls_pair_f, fq_pair_f, _ = merge_vars(
                factors_data=slice_data,
                vars_indices=np.array([i, j], dtype=np.int64),
                var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
            )
            cls_x1_f, fq_x1_f, _ = merge_vars(
                factors_data=slice_data,
                vars_indices=np.array([i], dtype=np.int64),
                var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
            )
            cls_x2_f, fq_x2_f, _ = merge_vars(
                factors_data=slice_data,
                vars_indices=np.array([j], dtype=np.int64),
                var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
            )
            i_pair = compute_mi_from_classes(
                classes_x=cls_pair_f, freqs_x=fq_pair_f,
                classes_y=cls_y_f, freqs_y=fq_y_f, dtype=dtype,
            )
            i_x1 = compute_mi_from_classes(
                classes_x=cls_x1_f, freqs_x=fq_x1_f,
                classes_y=cls_y_f, freqs_y=fq_y_f, dtype=dtype,
            )
            i_x2 = compute_mi_from_classes(
                classes_x=cls_x2_f, freqs_x=fq_x2_f,
                classes_y=cls_y_f, freqs_y=fq_y_f, dtype=dtype,
            )
            fold_ii_vals.append(i_pair - i_x1 - i_x2)

        per_fold_ii[(i, j)] = fold_ii_vals
        n_clear = sum(v > floor for v in fold_ii_vals)
        prevalence = n_clear / K
        if prevalence >= cfg.min_fold_prevalence:
            kept.append(k)
        elif verbose:
            logger.info(
                "cat-FE: pair (%d, %d) dropped by K-fold stability "
                "(%d/%d folds cleared floor, need >= %.2f)",
                i, j, n_clear, K, cfg.min_fold_prevalence,
            )

    return np.asarray(kept, dtype=selected_idx.dtype), per_fold_ii


# ============================================================================
# Coordinate-ascent refinement after greedy k-way
#
# Greedy k-way picks a triplet (A, B, C). Coordinate-ascent then tries replacing each member with each non-member; if the swap improves joint MI, keep it. Catches cases
# where the greedy seed missed a better neighbor. Refines local optima, doesn't break global structure.
# ============================================================================


def _refine_kway_coordinate_ascent(
    factors_data: np.ndarray,
    kway_results: list,
    candidate_pool: np.ndarray,
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    max_combined_nbins: int,
    n_passes: int,
    dtype,
    verbose: int,
) -> list:
    """For each k-way result, run ``n_passes`` of coordinate-ascent: try swapping each member with each non-member; keep if joint MI improves. Returns refined kway_results."""
    if n_passes <= 0 or not kway_results:
        return kway_results
    refined = []
    for orig_tuple, orig_classes, orig_nuniq, orig_mi in kway_results:
        current = list(orig_tuple)
        current_mi = orig_mi
        current_classes = orig_classes
        current_nuniq = orig_nuniq
        for _ in range(n_passes):
            improved = False
            for pos in range(len(current)):
                # The other k-1 members are frozen for this position's whole candidate sweep -- pre-merge them
                # once (prefix states keyed by sorted insertion point) instead of re-scanning all k raw columns
                # for every candidate.
                frozen_sorted = sorted(current[:pos] + current[pos + 1:])
                prefix_states = _build_merge_prefix_states(factors_data, frozen_sorted, nbins, dtype)
                for cand in candidate_pool:
                    cand_int = int(cand)
                    if cand_int in current:
                        continue
                    new_tuple_sorted = tuple(sorted([*frozen_sorted, cand_int]))
                    # Card budget check
                    card = 1
                    for k in new_tuple_sorted:
                        card *= int(nbins[k])
                    if card > max_combined_nbins or card >= 2**31:
                        continue
                    new_classes, new_freqs, new_nuniq = _merge_vars_sorted_insert(
                        factors_data, prefix_states, frozen_sorted, cand_int, nbins, dtype,
                    )
                    new_mi = compute_mi_from_classes(
                        classes_x=new_classes, freqs_x=new_freqs,
                        classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
                    )
                    if new_mi > current_mi:
                        current = list(new_tuple_sorted)
                        current_mi = new_mi
                        current_classes = new_classes
                        current_nuniq = new_nuniq
                        improved = True
                        # An accepted swap mid-sweep changes what "frozen" (current minus position pos)
                        # means for the REMAINING candidates at this same pos -- rebuild immediately so
                        # they splice against the just-updated members, not a stale pre-accept snapshot.
                        frozen_sorted = sorted(current[:pos] + current[pos + 1:])
                        prefix_states = _build_merge_prefix_states(factors_data, frozen_sorted, nbins, dtype)
                        if verbose >= 2:
                            logger.info(
                                "cat-FE coord-ascent: %s -> %s (MI %.4f -> %.4f)",
                                orig_tuple, tuple(current), orig_mi, current_mi,
                            )
            if not improved:
                break
        refined.append((tuple(sorted(current)), current_classes, current_nuniq, current_mi))
    return refined
