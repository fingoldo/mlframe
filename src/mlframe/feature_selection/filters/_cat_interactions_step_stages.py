"""Stage helpers of ``run_cat_interaction_step`` (``_cat_interactions_step``): candidate screen, pair search, survivor selection, permutation
confirmation and materialisation. Each stage reads/writes the shared ``st`` namespace and returns the skipped-result tuple on an early exit, else ``None``
(the last stage returns the final result).

``.cat_interactions`` names are imported lazily inside each stage: ``.predict`` re-imports ``_cat_interactions_step`` at its bottom, so a top-level import
would create a hard cycle.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ._cat_interactions_step import (
    _drop_bootstrap_ci_unstable,
    _run_cat_interaction__step1_kernel_costs_extra,
    _run_cat_interaction__step2_signal_detectable_top,
    _run_cat_interactio_bandit_ucb1_budget_allocation,
    _run_cat_interactio_cache_active,
    _run_cat_interactio_diagnostics_always_cheap_gated,
    _run_cat_interactio_each_pair_recipe_additionally,
    _run_cat_interactio_kernel_costs_extra_ops,
    _run_cat_interactio_materialise_way_survivors_alongside,
    _run_cat_interactio_memmap_detection,
    _run_cat_interactio_out_name_enumerate_new,
    _run_cat_interactio_transform_reproduces_identical_bin,
    _run_cat_interactio_usability_gpu_py_batch,
    _run_cat_interactio_use_gpu,
    _run_cat_interactio_use_weights,
    _run_cat_interactio_value_instead_binning_through,
    _selection_keep_mask,
    _skipped,
    _warn_weighted_bandit_fallback,
    enumerate_candidate_pairs,
)

logger = logging.getLogger("mlframe.feature_selection.filters._cat_interactions_step")


def _stage_candidates(
    st: Any, data: np.ndarray, cols: list, nbins: np.ndarray, target_indices: np.ndarray, classes_y: np.ndarray, freqs_y: np.ndarray,
    categorical_vars: list, cfg: Any, streaming_cache: dict | None, numeric_raw_values: dict | None, dtype: type, verbose: int,
) -> tuple | None:
    """Gates, memmap detection, include_numeric working pool, candidate validation and marginal-MI screen; ``None`` to continue, else the skipped result."""
    from .cat_interactions import _select_candidate_indices

    # ---- Pathological-input gates ----
    if target_indices.size == 0:
        raise ValueError("cat-FE: empty target_indices; cannot compute MI(X;Y).")
    if st.n_samples < cfg.min_n_samples:
        return _skipped(st, verbose, "cat-FE skipped: n_samples=%d < cfg.min_n_samples=%d", st.n_samples, cfg.min_n_samples)

    # ---- Memmap detection ----
    _run_cat_interactio_memmap_detection(data)

    # ---- include_numeric: quantile-bin eligible numeric columns into a transient working pool ----
    # Numeric columns are appended (quantile-coded, edges captured) to working copies of data / cols / nbins
    # and their positions joined to the candidate pool, so the existing pair / k-way machinery treats them
    # exactly like categoricals. The ORIGINAL data / cols / nbins are restored before the final concat, so the
    # numeric columns' own (MDLP) codes that flow to downstream screening are UNCHANGED - only the engineered
    # cross columns are appended. The per-column quantile edges are stamped into each recipe (below) so
    # ``transform`` reproduces identical bin codes from raw test values (leak-free, no train/serve skew).
    st.numeric_candidate_idxs = []
    st.numeric_edges_by_name = {}
    cols, data, nbins = _run_cat_interactio_transform_reproduces_identical_bin(cfg, numeric_raw_values, st.n_samples, cols, st.state, st.numeric_candidate_idxs, data, st.numeric_edges_by_name, nbins, verbose)

    # ---- Column-level validation ----
    st.candidate_idxs = _select_candidate_indices(
        nbins=nbins,
        categorical_vars=list(categorical_vars) + st.numeric_candidate_idxs,
        cfg=cfg, state=st.state,
        n_samples=st.n_samples,
    )
    if len(st.candidate_idxs) < 2:
        return _skipped(st, verbose, "cat-FE skipped: only %d eligible candidate columns after validation", len(st.candidate_idxs))

    # ---- Marginal MI screen ----
    st.candidate_idxs_arr = np.asarray(st.candidate_idxs, dtype=np.int64)
    if verbose:
        logger.info("cat-FE: marginal MI screen over %d candidate columns", len(st.candidate_idxs))

    # Streaming cache check. If enabled AND cache provided, reuse cached marginal MIs for columns whose distribution hasn't drifted (KL < threshold).
    st.cache_active = getattr(cfg, "enable_streaming_cache", False) and streaming_cache is not None and streaming_cache  # non-empty
    # Content signature of the (discretized) target - gates cache reuse so a changed Y invalidates the
    # cached MI(X;Y) even when X's distribution is unchanged.
    from .cat_interactions import _target_signature
    st.target_sig = _target_signature(data[:, target_indices])
    st.new_signatures = {}
    st.candidate_mi, st.new_signatures = _run_cat_interactio_cache_active(st.cache_active, streaming_cache, data, st.candidate_idxs_arr, nbins, cfg, st.target_sig, verbose, classes_y, freqs_y, dtype, st.new_signatures)

    # Persist updated cache for next fit() call
    if getattr(cfg, "enable_streaming_cache", False):
        st.state.streaming_cache_out = {
            "target_sig": st.target_sig,
            "col_signatures": st.new_signatures,
            "marginal_mis": {int(st.candidate_idxs_arr[k]): float(st.candidate_mi[k]) for k in range(len(st.candidate_idxs_arr))},
        }
    # marginal_floor prune
    if cfg.marginal_floor > 0:
        keep_mask = st.candidate_mi > cfg.marginal_floor
        st.candidate_idxs_arr = st.candidate_idxs_arr[keep_mask]
        st.candidate_mi = st.candidate_mi[keep_mask]
    if len(st.candidate_idxs_arr) < 2:
        return _skipped(st, verbose, "cat-FE skipped: %d cols cleared marginal_floor", len(st.candidate_idxs_arr))

    # Build a marginal-MI lookup keyed by COLUMN INDEX (into data), so the pair kernel can look up by index without re-running the screen.
    marginal_mi_full = np.full(data.shape[1], np.nan, dtype=float)
    for k, idx in enumerate(st.candidate_idxs_arr):
        marginal_mi_full[int(idx)] = st.candidate_mi[k]
    st.data, st.cols, st.nbins, st.marginal_mi_full = data, cols, nbins, marginal_mi_full
    return None


def _stage_pair_search(
    st: Any, data: np.ndarray, nbins: np.ndarray, classes_y: np.ndarray, freqs_y: np.ndarray, cfg: Any, weights: np.ndarray | None,
    dtype: type, verbose: int, marginal_mi_full: np.ndarray,
) -> tuple | None:
    """Pair enumeration under the cardinality budget plus the CPU / GPU pair search; stores the pair arrays and weighting state on ``st``."""
    from .cat_interactions import resolve_max_combined_nbins

    # ---- Pair enumeration with cardinality budget ----
    max_combined = resolve_max_combined_nbins(cfg, st.n_samples)
    # VECTORIZED: ``np.triu_indices`` enumerates every unordered (ii, jj) pair in one call
    # instead of a pure-Python nested loop (~125K iterations at p~500 per the audit) - the cardinality-
    # budget filter is then a single boolean mask over the whole pair set rather than an if/continue per
    # iteration. Bit-identical selection: the SAME two conditions (``nb_prod <= max_combined`` and
    # ``nb_prod < 2**31``) gate the SAME candidate pairs, in the SAME (ii, jj) enumeration order
    # ``triu_indices`` produces (row-major upper triangle, matching the nested loop's iteration order).
    # ``nb_prod`` is computed in int64 (not numpy's default-width product) so a wide-cardinality pair
    # cannot silently wrap before the ``>= 2**31`` overflow check gets to see it - the whole point of
    # that guard.
    pairs_a, pairs_b = enumerate_candidate_pairs(st.candidate_idxs_arr, nbins, max_combined)
    if pairs_a.size == 0:
        return _skipped(st, verbose, "cat-FE skipped: 0 pairs cleared cardinality budget %d", max_combined)
    if verbose:
        logger.info(
            "cat-FE: pair search over %d candidate pairs (cardinality budget %d)",
            len(pairs_a), max_combined,
        )

    # ---- Pair search: CPU (njit prange) or GPU dispatch ----
    # Backend selection per cfg.backend:
    # - "cpu": always njit prange
    # - "gpu": always GPU dispatch (raises if CuPy missing)
    # - "auto": GPU only at large-N regime (N>=200 cols AND n>=500k rows)
    st.use_gpu = False
    # The pair-search GPU gate used to consult only cupy
    # importability + is_gpu_available()'s own probe, never the project's canonical gpu_globally_disabled()
    # (MLFRAME_DISABLE_GPU=1 / CUDA_VISIBLE_DEVICES="") - a user forcing CPU-only via that env convention
    # still got GPU dispatch for backend="auto"/"gpu" cat-FE fits whenever cupy was importable. The global
    # off-switch now wins even over an explicit backend="gpu" request (raises a clear error instead of
    # silently using GPU), matching the pattern already used elsewhere in this package (_fe_gpu_strict.py,
    # _usability_gpu.py, batch_pair_mi_gpu.py/friend_graph_gpu.py).
    from ._gpu_policy import gpu_globally_disabled
    st._gpu_disabled = gpu_globally_disabled()
    st.use_gpu = _run_cat_interactio_usability_gpu_py_batch(cfg, st._gpu_disabled, st.candidate_idxs_arr, st.n_samples, verbose, st.use_gpu)

    # Choose weighted vs unweighted kernel. Use weighted only when weights are actually non-uniform; uniform weights are equivalent to unweighted and the weighted
    # kernel costs extra ops, so skip in that case.
    use_weights = False
    use_weights, weights = _run_cat_interactio_kernel_costs_extra_ops(weights, st.n_samples, use_weights)
    _run_cat_interactio_use_weights(use_weights, weights, data, st.candidate_idxs_arr, nbins, classes_y, dtype, marginal_mi_full)
    st.ii_arr = st.joint_mi_arr = st.n_uniq_arr = None  # set by the GPU path below; the CPU helper fills them when the GPU path did not run
    _run_cat_interaction__step1_kernel_costs_extra(st, use_weights, verbose, pairs_a, cfg, data, pairs_b, nbins, classes_y, freqs_y, dtype, marginal_mi_full)
    st.ii_arr, st.joint_mi_arr, st.n_uniq_arr = _run_cat_interactio_use_gpu(st.use_gpu, use_weights, verbose, data, pairs_a, pairs_b, marginal_mi_full, nbins, classes_y, weights, dtype, freqs_y, st.ii_arr, st.joint_mi_arr, st.n_uniq_arr)
    st.pairs_a, st.pairs_b, st.max_combined, st.use_weights, st.weights = pairs_a, pairs_b, max_combined, use_weights, weights
    return None


def _stage_select_survivors(
    st: Any, selected_so_far: list | None, data: np.ndarray, nbins: np.ndarray, target_indices: np.ndarray, classes_y: np.ndarray,
    freqs_y: np.ndarray, cfg: Any, dtype: type, verbose: int, use_weights: bool, weights: np.ndarray | None, pairs_a: np.ndarray, pairs_b: np.ndarray,
) -> tuple | None:
    """Top-K selection, Miller-Madow re-rank, anti-redundancy re-rank and the K-fold stability filter."""
    from .cat_interactions import _anti_redundancy_rerank, _kfold_stability_filter, _maybe_rerank_with_mm, _select_top_k_pairs, resolve_min_interaction_information

    # ---- Top-K selection ----
    st.selected_idx = _select_top_k_pairs(
        ii_arr=st.ii_arr,
        pairs_a=pairs_a, pairs_b=pairs_b,
        cfg=cfg, n_samples=st.n_samples,
    )
    if len(st.selected_idx) == 0:
        return _skipped(st, verbose, "cat-FE: 0 pairs cleared min_interaction_information; no engineered cols")
    if verbose:
        logger.info("cat-FE: %d pair(s) selected for materialisation", len(st.selected_idx))

    # ---- Miller-Madow re-rank on top-K survivors ----
    # Only top-K pay the 5x merge_vars cost; saves N^2-scale compute while catching high-cardinality bias-driven false positives. The re-rank may shuffle the heap;
    # ``selected_idx`` order matters for downstream ``_materialize_pairs`` because it determines which engineered cols are added first (relevant when ``top_k_pairs``
    # cap binds).
    st.ii_arr, st.selected_idx = _maybe_rerank_with_mm(
        factors_data=data,
        pairs_a=pairs_a, pairs_b=pairs_b,
        selected_idx=st.selected_idx, ii_arr=st.ii_arr,
        nbins=nbins, target_indices=target_indices,
        classes_y=classes_y, freqs_y=freqs_y,
        cfg=cfg, dtype=dtype, verbose=verbose,
        weights=weights if use_weights else None,
    )
    # After MM re-rank, some pairs may drop below the floor; re-filter.
    st.floor = resolve_min_interaction_information(cfg, st.n_samples)
    st.keep = _selection_keep_mask(cfg, st.ii_arr, st.selected_idx, st.floor)
    st.selected_idx = st.selected_idx[st.keep]
    if len(st.selected_idx) == 0:
        return _skipped(st, verbose, "cat-FE: 0 pairs survived MM re-rank floor; no engineered cols")

    # ---- Anti-redundancy re-rank (opt-in via anti_redundancy_beta>0) ----
    # Adjusts each survivor's score by ``beta * mean_z I(merged; Z)`` where Z ranges over already-selected features in ``selected_so_far``. No-op when beta=0 or selected_so_far is empty.
    if selected_so_far:
        st.ii_arr, st.selected_idx = _anti_redundancy_rerank(
            factors_data=data, pairs_a=pairs_a, pairs_b=pairs_b,
            selected_idx=st.selected_idx, ii_arr=st.ii_arr,
            nbins=nbins, selected_so_far=selected_so_far,
            classes_y=classes_y, cfg=cfg, dtype=dtype, verbose=verbose,
            weights=weights if use_weights else None,
        )
        # Re-apply the floor after anti-redundancy correction
        st.floor = resolve_min_interaction_information(cfg, st.n_samples)
        st.keep = _selection_keep_mask(cfg, st.ii_arr, st.selected_idx, st.floor)
        st.selected_idx = st.selected_idx[st.keep]
        if len(st.selected_idx) == 0:
            return _skipped(st, verbose, "cat-FE: 0 pairs survived anti-redundancy floor")

    # ---- K-fold II stability filter (opt-in via n_folds_stability>0) ----
    # Drops pairs whose II is unstable across K folds (signal driven by outlier rows). Runs BEFORE permutation so we don't pay perm budget on pairs that fail stability.
    st.selected_idx, st.per_fold_ii_dict = _kfold_stability_filter(
        factors_data=data, pairs_a=pairs_a, pairs_b=pairs_b,
        selected_idx=st.selected_idx, nbins=nbins,
        target_indices=target_indices,
        cfg=cfg, dtype=dtype, verbose=verbose,
        weights=weights if use_weights else None,
    )
    # Surface fold IIs into the state UNCONDITIONALLY - even when 0 pairs survived, the per-fold values are useful diagnostics (user can see WHY the pair was rejected).
    # Must happen BEFORE the early return.
    if st.per_fold_ii_dict:
        st.state.ii_stability.update(st.per_fold_ii_dict)
    if len(st.selected_idx) == 0:
        return _skipped(st, verbose, "cat-FE: 0 pairs survived K-fold stability filter")
    return None


def _stage_confirm_pairs(
    st: Any, cfg: Any, use_weights: bool, data: np.ndarray, pairs_a: np.ndarray, pairs_b: np.ndarray, nbins: np.ndarray, classes_y: np.ndarray,
    freqs_y: np.ndarray, dtype: type, verbose: int, weights: np.ndarray | None, marginal_mi_full: np.ndarray,
) -> tuple | None:
    """Permutation confirmation with FWER correction, then the bootstrap-CI stability drop."""
    from .cat_interactions import _bootstrap_ii_cis

    # ---- Permutation confirmation + FWER correction ----
    # Runs only when ``cfg.full_npermutations > 0`` (default 100). Tests joint-independence null; failed pairs are dropped from ``selected_idx``. The resulting
    # ``confidence_dict`` is surfaced via diagnostics for user inspection. ``n_search_pairs`` is the family size for FWER correction - the count of pairs CONSIDERED
    # in the search phase, NOT the top-K count.
    # Full Westfall-Young requires the per-shuffle max-II across ALL search pairs, materially more expensive than per-survivor permutation. To get the full WY
    # behaviour, we substitute the per-pair p-values from the joint-independence test with the WY-corrected versions BEFORE the orchestrator applies the floor.
    # Memory budget: full WY pre-merges m * n int32 cells; if that exceeds e.g. 500 MB we fall back to Bonferroni-on-survivors via the _apply_fwer_correction path.
    st.use_full_wy = cfg.fwer_correction == "westfall_young" and cfg.full_npermutations > 0 and len(pairs_a) * st.n_samples * 4 < 500 * 1024 * 1024

    # perm_budget_strategy defaults to "bandit_ucb1" (not
    # "fixed"), and the bandit allocator's bulk-shuffle kernels have no weighted variant - so the DEFAULT
    # confirmation path for any weighted cat-FE fit silently tested a weighted II_obs against an unweighted
    # permutation null (the one warning that existed was gated behind `verbose`, invisible at the library's
    # normal verbose=0). Auto-fall-back to the fixed path (which IS correctly weighted) whenever weighting
    # is active, rather than silently dropping the weighting.
    _warn_weighted_bandit_fallback(cfg, use_weights)
    # Bandit UCB1 budget allocation overrides the fixed path when cfg.perm_budget_strategy='bandit_ucb1' (and full_npermutations>0 AND not using full WY which has its own coordination).
    st.confidence_dict, st.selected_idx = _run_cat_interactio_bandit_ucb1_budget_allocation(cfg, st.use_full_wy, use_weights, data, pairs_a, pairs_b, st.selected_idx, st.ii_arr, nbins, classes_y, freqs_y, dtype, verbose, weights, marginal_mi_full)
    if len(st.selected_idx) == 0:
        return _skipped(st, verbose, "cat-FE: 0 pairs cleared permutation confirmation")

    # ---- Bootstrap CIs on II (opt-in via bootstrap_ci_n_replicates>0) ----
    st.bootstrap_ci_dict = _bootstrap_ii_cis(
        factors_data=data, pairs_a=pairs_a, pairs_b=pairs_b,
        selected_idx=st.selected_idx,
        nbins=nbins, classes_y=classes_y, freqs_y=freqs_y,
        cfg=cfg, dtype=dtype, verbose=verbose,
        weights=weights if use_weights else None,
    )
    _ci_skipped = _drop_bootstrap_ci_unstable(cfg, st, pairs_a, pairs_b, verbose)
    if _ci_skipped is not None:
        return _ci_skipped
    return None


def _stage_materialise(
    st: Any, cfg: Any, data: np.ndarray, cols: list, nbins: np.ndarray, target_indices: np.ndarray, classes_y: np.ndarray, freqs_y: np.ndarray, dtype: type,
    pairs_a: np.ndarray, pairs_b: np.ndarray, marginal_mi_full: np.ndarray, max_combined: int, use_weights: bool, weights: np.ndarray | None, verbose: int,
) -> tuple:
    """K-way expansion, materialisation of the engineered columns, recipe/lineage bookkeeping and the final concat."""
    from .cat_interactions import _materialize_pairs

    # ---- K-way greedy expansion (opt-in via max_kway_order > 2) ----
    # HYBRID seeding - first try only the top-K confirmed pairs, which is O(top_k * N) = ~6400 merge_vars at top_k=64, N=100. If that produces ZERO k-way results
    # (the pure-k-way-XOR case where all 2-way IIs are noise around 0 and top-K is random), fall back to seeding from ALL pairs. This avoids the quadratic cost
    # when the signal is detectable from top-K, but preserves the all-pairs path for pathological niches.
    kway_results: list = []
    kway_results = _run_cat_interaction__step2_signal_detectable_top(cfg, st, pairs_a, pairs_b, data, nbins, classes_y, freqs_y, marginal_mi_full, max_combined, dtype, use_weights, weights, kway_results, verbose)

    # ---- Materialise pair survivors (single concat) ----
    st.new_data_block, st.new_names, st.new_nbins, st.new_recipes = _materialize_pairs(
        factors_data=data,
        pairs_a=pairs_a, pairs_b=pairs_b,
        selected_idx=st.selected_idx,
        nbins=nbins,
        cols=cols,
        dtype=dtype,
        unknown_strategy=cfg.unknown_strategy,
    )

    # ---- Materialise k-way survivors (alongside pairs) ----
    st.new_data_block = _run_cat_interactio_materialise_way_survivors_alongside(kway_results, data, nbins, cols, dtype, cfg, st.new_data_block, st.new_names, st.new_nbins, st.new_recipes, st.n_samples, st.state)
    # Diagnostics (always cheap, gated by cfg).
    _run_cat_interactio_diagnostics_always_cheap_gated(cfg, st.selected_idx, pairs_a, pairs_b, st.ii_arr, st.joint_mi_arr, marginal_mi_full, st.n_uniq_arr, cols, st.n_samples, st.confidence_dict, st.bootstrap_ci_dict, st.state, st.new_names)

    st.state.recipes.extend(st.new_recipes)

    # ---- Target encoding emit (opt-in) ----
    # For each pair recipe, additionally emit a target-encoded col with OOF CV-aware shrinkage. Recipes carry ``kind="target_encoding"`` with the global cell-means table for transform() replay.
    _run_cat_interactio_each_pair_recipe_additionally(cfg, st.selected_idx, pairs_a, pairs_b, data, target_indices, classes_y, nbins, dtype, cols, st.new_names, st.new_recipes, st.state, verbose)

    # ---- Stamp quantile bin edges onto recipes built from numeric sources (leak-safe transform replay) ----
    # Without this, ``_apply_factorize`` / ``_apply_target_encoding`` would ``astype(int64)`` a raw numeric test
    # value (3.7 -> 3) instead of binning it through the fit-time quantile edges -> a silent train/serve skew.
    _run_cat_interactio_value_instead_binning_through(st.numeric_edges_by_name, st.state)

    # ---- Single concat onto data / cols / nbins ----
    # Restore the ORIGINAL arrays: the engineered cross block is appended to them, NOT to the include_numeric
    # working pool (whose transient quantile-coded numeric columns must never reach downstream screening).
    data, cols, nbins = st.orig_data, st.orig_cols, st.orig_nbins
    st.data_out = np.concatenate([data, st.new_data_block], axis=1)
    st.cols_out = list(cols) + st.new_names
    st.nbins_out = np.concatenate([nbins, np.asarray(st.new_nbins, dtype=nbins.dtype)])

    # ---- Build engineered_lineage map ----
    # Engineered cols land at indices [n_orig, n_orig + len(new_names)). For each, record the parent indices (in the ORIGINAL data layout) so screen_predictors
    # can skip ``(orig_parent, engineered_col)`` k-way candidates - they're redundant by construction.
    st.n_orig = data.shape[1]
    st.name_to_idx = {n: i for i, n in enumerate(cols)}  # original col name -> idx
    st.state.lineage = {}
    _run_cat_interactio_out_name_enumerate_new(st.new_names, st.n_orig, st.new_recipes, st.name_to_idx, st.state)

    return st.data_out, st.cols_out, st.nbins_out, st.state
