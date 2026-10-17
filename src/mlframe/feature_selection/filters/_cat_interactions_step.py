"""``run_cat_interaction_step`` carved out of
``mlframe.feature_selection.filters.cat_interactions``.

Re-imported at the parent's module bottom so historical
``from mlframe.feature_selection.filters.cat_interactions import run_cat_interaction_step``
resolves transparently.
"""
from __future__ import annotations

import logging
import math
from typing import Tuple, Any

import numpy as np

from ._fe_gpu_strict import _STRICT_MIN_CALL_P
from .cat_fe_state import CatFEConfig, CatFEState
from .engineered_recipes import EngineeredRecipe
from types import SimpleNamespace as _SimpleNamespace

logger = logging.getLogger(__name__)

# cfg.backend="auto" GPU-eligibility floor: the ORIGINAL two-independent-thresholds contract was
# "n_cols_eff >= 200 AND n_samples >= 500_000" (see the docstring at ``run_cat_interaction_step``'s
# work-based gate below for why that's structurally wrong). Kept as named constants - not inlined -
# so the SAME combined work magnitude is reused, not duplicated, at the one call site.
_AUTO_GPU_MIN_COLS = 200
_AUTO_GPU_MIN_N = 500_000


def _cat_fe_auto_wants_gpu(n_samples: int, n_cols_eff: int) -> bool:
    """Whether ``cfg.backend="auto"`` should attempt the GPU dispatch for this (n_samples, n_cols_eff)
    shape (CuPy availability is checked separately by the caller).

    WORK-BASED gate. The old rule required BOTH ``n_cols_eff >= _AUTO_GPU_MIN_COLS`` AND
    ``n_samples >= _AUTO_GPU_MIN_N`` as two INDEPENDENT thresholds - ignoring their PRODUCT, so a wide-
    but-under-500k-row fit could NEVER qualify purely for sitting under the row mark, no matter how many
    columns it had (and vice versa: a huge-row, ~100-column fit never qualified either, despite comparable
    total work). Converted to a work-PRODUCT check against the SAME combined magnitude the original two-
    threshold rule implied (no new/invented calibration - the same two constants, now multiplied instead
    of AND'ed), plus the SAME p-floor the validated n*p CPU/CUDA crossover uses elsewhere in MRMR
    (``_should_use_cuda`` / ``_fe_gpu_strict._STRICT_MIN_CALL_P``) so a degenerate huge-n/tiny-p shape with
    no real column-batching parallelism still declines GPU. Note this does NOT, by itself, flip the
    wellbore-100k shape (n=99401, ~500 candidate cols, work~49.7M) to GPU - that shape is still under the
    100M combined floor; a fit with either somewhat more columns or somewhat more rows at comparable width
    now correctly qualifies where the old AND-of-independent-thresholds rule never could, regardless of how
    large one dimension grew alone."""
    return n_cols_eff >= _STRICT_MIN_CALL_P and (n_samples * n_cols_eff) >= (_AUTO_GPU_MIN_COLS * _AUTO_GPU_MIN_N)


def _quantile_bin_with_edges(raw: np.ndarray, n_bins: int) -> tuple:
    """Quantile-bin a 1-D numeric array into ``[0, n_bins)`` ordinal codes; return ``(codes, inner_edges)``.

    ``inner_edges`` are the ``n_bins - 1`` interior quantile cut points (unique-deduped); the bin code is
    ``np.searchsorted(inner_edges, value, side="right")`` - the EXACT convention ``categorize_dataset``'s
    adaptive path uses (``_discretization_dataset.py``), so codes computed here at fit time are reproduced
    byte-for-byte by the recipe replay at transform time from the stored edges (no train/serve skew).

    Returns ``edges.size == 0`` for a constant / degenerate column (caller skips it: a 1-bin column carries
    no interaction signal). NaN-bearing columns must be filtered out by the caller - this v1 edge scheme has
    no dedicated NaN bin, so a NaN would ``searchsorted`` to the top real bin and silently corrupt the cross.
    """
    arr = np.asarray(raw, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(arr, qs))
    if edges.size == 0:
        return np.zeros(arr.shape[0], dtype=np.int64), edges
    codes = np.searchsorted(edges, arr, side="right").astype(np.int64)
    return codes, edges


# ============================================================================
# Streaming / incremental fit cache
#
# Cache marginal MIs and per-column distribution signatures across fit() calls. On re-fit with same CatFEConfig + similar data, skip recomputation for columns
# whose distribution hasn't drifted (KL < tau). Saves ~70% on production daily-refresh re-fits.
# ============================================================================


def enumerate_candidate_pairs(candidate_idxs_arr: np.ndarray, nbins: np.ndarray, max_combined: int) -> Tuple[np.ndarray, np.ndarray]:
    """Every unordered candidate pair whose combined cardinality clears the budget, in nested-loop order.

    Module-level so a test can execute THIS code rather than a copy of it. The identity test that guards
    the vectorisation used to hold two hand-copies side by side and import nothing from the package, so
    dropping the ``< 2**31`` mask or the int64 cast from production left both of its cases green while a
    46341x46341 pair was admitted and overflowed downstream int32 combined-code arithmetic.

    ``np.triu_indices`` enumerates every (ii, jj) pair in one call instead of a pure-Python nested loop
    (~125K iterations at p~500), and the cardinality-budget filter is a single boolean mask over the whole
    pair set rather than an if/continue per iteration. Selection is bit-identical to the loop: the SAME two
    conditions gate the SAME pairs in the SAME row-major upper-triangle order.

    ``nb_prod`` is computed in int64, NOT numpy's default-width product, so a wide-cardinality pair cannot
    silently wrap before the ``>= 2**31`` overflow check gets to see it -- which is the whole point of that
    guard, and load-bearing rather than cosmetic.
    """
    n_cand = len(candidate_idxs_arr)
    if n_cand < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    ii, jj = np.triu_indices(n_cand, k=1)
    i_arr = np.asarray(candidate_idxs_arr, dtype=np.int64)[ii]
    j_arr = np.asarray(candidate_idxs_arr, dtype=np.int64)[jj]
    nbins_i64 = np.asarray(nbins, dtype=np.int64)
    nb_prod = nbins_i64[i_arr] * nbins_i64[j_arr]
    keep = (nb_prod <= int(max_combined)) & (nb_prod < 2**31)
    return i_arr[keep], j_arr[keep]

def _skipped(st: Any, verbose: Any, message: str, *args: Any) -> tuple[Any, Any, Any, Any]:
    """Log (when verbose) why cat-FE did nothing and return the ORIGINAL arrays with the empty state, which is what every early exit yields."""
    if verbose:
        logger.info(message, *args)
    return st.orig_data, st.orig_cols, st.orig_nbins, st.state


def _warn_weighted_bandit_fallback(cfg, use_weights):
    """Warn that a weighted fit leaves the bandit UCB1 permutation path for the fixed-budget one, which is the weighted-correct path."""
    if use_weights and getattr(cfg, "perm_budget_strategy", "fixed") == "bandit_ucb1":
        logger.warning(
            "cat-FE: perm_budget_strategy='bandit_ucb1' has no weighted bulk-shuffle kernel yet; "
            "falling back to the fixed-budget permutation path (which IS correctly weighted) for this "
            "weighted fit. Set perm_budget_strategy='fixed' explicitly to silence this message."
        )


def _drop_bootstrap_ci_unstable(cfg, st, pairs_a, pairs_b, verbose) -> tuple[Any, Any, Any, Any] | None:
    """Drop survivors whose bootstrap lower-CI falls below the floor; returns the skipped result when none survive, else None."""
    from mlframe.feature_selection.filters.cat_interactions import resolve_min_interaction_information
    if st.bootstrap_ci_dict:
        # Drop survivors whose lower-CI < floor (unstable signal)
        floor_ci = resolve_min_interaction_information(cfg, st.n_samples)
        kept_after_ci: list[Any] = []
        _run_cat_interactio_drop_survivors_whose_lower(st.selected_idx, pairs_a, pairs_b, st.bootstrap_ci_dict, floor_ci, kept_after_ci, verbose)
        st.selected_idx = np.asarray(kept_after_ci, dtype=st.selected_idx.dtype)
        if len(st.selected_idx) == 0:
            return _skipped(st, verbose, "cat-FE: 0 pairs survived bootstrap CI floor")
    return None


def run_cat_interaction_step(
    *,
    data: np.ndarray,
    cols: list,
    nbins: np.ndarray,
    target_indices: np.ndarray,
    classes_y: np.ndarray,
    classes_y_safe: np.ndarray,
    freqs_y: np.ndarray,
    categorical_vars: list,
    cfg: CatFEConfig,
    selected_so_far: list | None = None,
    weights: np.ndarray | None = None,  # Per-row sample weights; None = uniform.
    streaming_cache: dict | None = None,  # Prior-fit cache for incremental re-fit.
    numeric_raw_values: dict | None = None,  # {orig_col_idx -> raw float values} for include_numeric quantile-binning.
    dtype: type = np.int32,
    verbose: int = 0,
) -> tuple:
    """One cat-FE iteration. Augments ``data`` / ``cols`` / ``nbins`` with new ordinal-encoded columns capturing pair (and k-way) synergies. Returns the augmented arrays
    plus a fresh ``CatFEState`` holding recipes + diagnostics.

    Args:
        data: ordinal-encoded ``(n_samples, n_cols)`` produced by ``categorize_dataset``.
        cols: list of column names matching ``data`` shape.
        nbins: cardinality per column.
        target_indices: precomputed by caller (avoids re-binning Y for every MI call).
        classes_y: precomputed class codes of Y, supplied by the caller.
        classes_y_safe: class codes of Y supplied by the caller; this step does not read it.
        freqs_y: precomputed class frequencies of Y, supplied by the caller.
        categorical_vars: indices into ``data`` of categorical (or pre-discretized numeric, when ``cfg.include_numeric=True``) columns to consider.
        cfg: the cat-FE config; ``cfg.enable=True`` is the user's opt-in switch (caller checks before calling us).
        selected_so_far: already-selected features used to re-rank survivors by their information with those features; empty or None skips it.
        weights: per-row sample weights; None means uniform. The weighted kernel is used only when the weights are non-uniform.
        streaming_cache: prior-fit cache whose marginal MIs are reused for columns whose distribution has not drifted; None disables reuse.
        numeric_raw_values: ``{orig_col_idx: raw float values}`` used to quantile-bin numeric columns when ``cfg.include_numeric`` is set.
        dtype: integer dtype of the code arrays passed to the MI kernels.
        verbose: when truthy, log progress and the reason for any early exit.

    Returns:
    - ``data_out``: augmented ``(n_samples, n_cols + n_engineered)``
    - ``cols_out``: ``cols + engineered_names``
    - ``nbins_out``: ``np.concatenate([nbins, engineered_nbins])``
    - ``state``: ``CatFEState`` with recipes / diagnostics populated.

    When the step adds no engineered columns (no pairs cleared the floor / all dropped by validation gates), returns the inputs unchanged with an empty ``CatFEState``.
    """
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    st = _SimpleNamespace()  # long-lived locals of this function (see the stage helpers below)
    from ._cat_interactions_step_stages import _stage_candidates, _stage_confirm_pairs, _stage_materialise, _stage_pair_search, _stage_select_survivors
    st.state = CatFEState()
    st.n_samples = data.shape[0]
    # Every early return below yields these ORIGINAL arrays; ``include_numeric`` (further down) shadows
    # data/cols/nbins with a transient working pool, so the originals are pinned here once.
    st.orig_data, st.orig_cols, st.orig_nbins = data, cols, nbins

    _res = _stage_candidates(st, data, cols, nbins, target_indices, classes_y, freqs_y, categorical_vars, cfg, streaming_cache, numeric_raw_values, dtype, verbose)
    if _res is not None:
        return _res
    data, cols, nbins, marginal_mi_full = st.data, st.cols, st.nbins, st.marginal_mi_full

    _res = _stage_pair_search(st, data, nbins, classes_y, freqs_y, cfg, weights, dtype, verbose, marginal_mi_full)
    if _res is not None:
        return _res
    pairs_a, pairs_b, max_combined, use_weights, weights = st.pairs_a, st.pairs_b, st.max_combined, st.use_weights, st.weights

    _res = _stage_select_survivors(st, selected_so_far, data, nbins, target_indices, classes_y, freqs_y, cfg, dtype, verbose, use_weights, weights, pairs_a, pairs_b)
    if _res is not None:
        return _res

    _res = _stage_confirm_pairs(st, cfg, use_weights, data, pairs_a, pairs_b, nbins, classes_y, freqs_y, dtype, verbose, weights, marginal_mi_full)
    if _res is not None:
        return _res

    return _stage_materialise(
        st, cfg, data, cols, nbins, target_indices, classes_y, freqs_y, dtype, pairs_a, pairs_b, marginal_mi_full, max_combined, use_weights, weights, verbose
    )


def _run_cat_interaction__step1_kernel_costs_extra(st, use_weights, verbose, pairs_a, cfg, data, pairs_b, nbins, classes_y, freqs_y, dtype, marginal_mi_full):
    """Step 1 of run_cat_interaction_step: lines starting at ``if st.use_gpu:``."""
    if st.use_gpu:
        from mlframe.feature_selection.filters.gpu import mi_direct_gpu_batched_pairs
        if use_weights:
            logger.warning("cat-FE: sample weights ignored on GPU path; " "falling back to CPU weighted kernel.")
            st.use_gpu = False
        else:
            if verbose:
                logger.info("cat-FE: pair search on GPU over %d pairs", len(pairs_a))
            st.use_gpu, st.ii_arr, st.joint_mi_arr, st.n_uniq_arr = _gpu_pair_search_or_none(
                cfg, mi_direct_gpu_batched_pairs, data, pairs_a, pairs_b, nbins, classes_y, freqs_y, dtype, marginal_mi_full,
            )


def _run_cat_interaction__step2_signal_detectable_top(cfg, st, pairs_a, pairs_b, data, nbins, classes_y, freqs_y, marginal_mi_full, max_combined, dtype, use_weights, weights, kway_results, verbose):
    """Step 2 of run_cat_interaction_step: lines starting at ``if cfg.max_kway_order > 2:``."""
    from mlframe.feature_selection.filters.cat_interactions import _greedy_expand_one_seed, _refine_kway_coordinate_ascent, resolve_min_interaction_information

    if cfg.max_kway_order > 2:
        min_inc_ii = resolve_min_interaction_information(cfg, st.n_samples)
        candidate_pool_arr = st.candidate_idxs_arr
        seen_kway: set = set()

        def _expand_seeds(seed_pair_indices):
            """Grow k-way candidates from each seed pair index, deduping via ``seen_kway``."""
            for k in seed_pair_indices:
                seed_a = int(pairs_a[k])
                seed_b = int(pairs_b[k])
                extension = _greedy_expand_one_seed(
                    factors_data=data,
                    seed_indices=(seed_a, seed_b),
                    candidate_pool=candidate_pool_arr,
                    nbins=nbins,
                    classes_y=classes_y, freqs_y=freqs_y,
                    marginal_mi=marginal_mi_full,
                    max_combined_nbins=max_combined,
                    max_kway_order=cfg.max_kway_order,
                    min_inc_ii=min_inc_ii,
                    dtype=dtype,
                    weights=weights if use_weights else None,
                )
                if extension is None:
                    continue
                idx_tuple = extension[0]
                if idx_tuple in seen_kway:
                    continue
                seen_kway.add(idx_tuple)
                kway_results.append(extension)

        # Phase 1: top-K seeds (cheap, ~O(top_k * N) merge_vars)
        _expand_seeds(list(st.selected_idx))
        _run_cat_interactio_phase_top_seeds_cheap(kway_results, pairs_a, st.selected_idx, verbose, _expand_seeds)

        # Sort k-way results by joint_MI desc and cap by top_k_pairs.
        # Secondary key on the var-index tuple so tied
        # joint_MI doesn't make the surviving k-way set drift across runs.
        kway_results.sort(key=lambda r: (-r[3], tuple(r[0]) if r and r[0] is not None else ()))
        kway_results = kway_results[: cfg.top_k_pairs]
        if verbose:
            logger.info(
                "cat-FE: greedy k-way expansion produced %d feature(s)",
                len(kway_results),
            )

        # Coordinate-ascent refinement (opt-in via refine_passes>0).
        if cfg.refine_passes > 0 and kway_results:
            kway_results = _refine_kway_coordinate_ascent(
                factors_data=data, kway_results=kway_results,
                candidate_pool=st.candidate_idxs_arr, nbins=nbins,
                classes_y=classes_y, freqs_y=freqs_y,
                max_combined_nbins=max_combined,
                n_passes=cfg.refine_passes,
                dtype=dtype, verbose=verbose,
                weights=weights if use_weights else None,
            )
    return kway_results


def _run_cat_interactio_bandit_ucb1_budget_allocation(cfg, use_full_wy, use_weights, data, pairs_a, pairs_b, selected_idx, ii_arr, nbins, classes_y, freqs_y, dtype, verbose, weights, marginal_mi_full):
    """Block of run_cat_interaction_step starting at ``if getattr(cfg, "perm_budget_strategy", "fixed") == "bandit_ucb1" and ``."""
    from mlframe.feature_selection.filters.cat_interactions import _compute_westfall_young_corrected_p, _confirm_pairs_bandit_ucb1, _confirm_pairs_via_permutation

    if getattr(cfg, "perm_budget_strategy", "fixed") == "bandit_ucb1" and cfg.full_npermutations > 0 and not use_full_wy and not use_weights:
        selected_idx, confidence_dict = _confirm_pairs_bandit_ucb1(
            factors_data=data, pairs_a=pairs_a, pairs_b=pairs_b,
            selected_idx=selected_idx, ii_arr=ii_arr,
            nbins=nbins, classes_y=classes_y, freqs_y=freqs_y,
            cfg=cfg, n_search_pairs=len(pairs_a),
            dtype=dtype, verbose=verbose,
            weights=weights if use_weights else None,
        )
    elif use_full_wy:
        # Full Westfall-Young path
        wy_corrected_p = _compute_westfall_young_corrected_p(
            factors_data=data, pairs_a=pairs_a, pairs_b=pairs_b,
            ii_obs_arr=ii_arr, selected_idx=selected_idx,
            nbins=nbins, classes_y=classes_y, freqs_y=freqs_y,
            marginal_mi=marginal_mi_full,
            n_perms=cfg.full_npermutations,
            dtype=dtype, verbose=verbose,
            weights=weights if use_weights else None,
        )
        confidence_dict = {ij: 1.0 - p for ij, p in wy_corrected_p.items()}
        min_conf = 0.95
        kept_mask = np.array([confidence_dict[(int(pairs_a[k]), int(pairs_b[k]))] >= min_conf for k in selected_idx])
        if verbose:
            for j, k in enumerate(selected_idx):
                ij = (int(pairs_a[k]), int(pairs_b[k]))
                if not kept_mask[j]:
                    logger.info(
                        "cat-FE WY: pair %s dropped (corrected_p=%.4f >= %.2f)",
                        ij, wy_corrected_p[ij], 1 - min_conf,
                    )
        selected_idx = selected_idx[kept_mask]
    else:
        selected_idx, confidence_dict = _confirm_pairs_via_permutation(
            factors_data=data, pairs_a=pairs_a, pairs_b=pairs_b,
            selected_idx=selected_idx, ii_arr=ii_arr,
            nbins=nbins, classes_y=classes_y, freqs_y=freqs_y,
            cfg=cfg, n_search_pairs=len(pairs_a),
            dtype=dtype, verbose=verbose,
            weights=weights if use_weights else None,
        )
    return confidence_dict, selected_idx


def _run_cat_interactio_materialise_way_survivors_alongside(kway_results, data, nbins, cols, dtype, cfg, new_data_block, new_names, new_nbins, new_recipes, n_samples, state):
    """Block of run_cat_interaction_step starting at ``if kway_results:``."""
    from mlframe.feature_selection.filters.cat_interactions import _materialize_kway

    if kway_results:
        kway_block, kway_names, kway_nbins, kway_recipes = _materialize_kway(
            factors_data=data,
            kway_results=kway_results,
            nbins=nbins,
            cols=cols,
            dtype=dtype,
            unknown_strategy=cfg.unknown_strategy,
        )
        new_data_block = np.concatenate([new_data_block, kway_block], axis=1)
        new_names.extend(kway_names)
        new_nbins.extend(kway_nbins)
        new_recipes.extend(kway_recipes)

        # K-way diagnostics
        if cfg.emit_diagnostics:
            for (idx_tuple, _, n_uniq, joint_mi), name in zip(kway_results, kway_names):
                state.diagnostics[name] = {
                    "II": float(joint_mi),  # k-way: II vs union-of-marginals would require recomputing; surface joint_MI as a coarse rank signal.
                    "joint_MI": float(joint_mi),
                    "joint_nclasses": int(n_uniq),
                    "src_indices": tuple(int(i) for i in idx_tuple),
                    "src_names": tuple(cols[i] for i in idx_tuple),
                    "kway_order": len(idx_tuple),
                    "n_obs_per_cell_p25": float(n_samples / max(int(n_uniq), 1)),
                    "joint_dependence_confidence": None,  # k-way perm-test not implemented
                }
    return new_data_block


def _run_cat_interactio_memmap_detection(data):
    """Block of run_cat_interaction_step starting at ``if isinstance(data.base, np.memmap):``."""
    if isinstance(data.base, np.memmap):
        try:
            import psutil
            avail = psutil.virtual_memory().available
        except ImportError:
            avail = -1
        if avail > 0 and data.nbytes > avail * 0.5:
            raise MemoryError(
                f"cat-FE refuses to copy a memory-mapped {data.nbytes / 2**30:.1f} GB array "
                f"into RAM (available: {avail / 2**30:.1f} GB). Disable cat-FE for memmap "
                f"inputs or ensure available RAM > 2 * data.nbytes."
            )


def _run_cat_interactio_transform_reproduces_identical_bin(cfg, numeric_raw_values, n_samples, cols, state, numeric_candidate_idxs, data, numeric_edges_by_name, nbins, verbose):
    """Block of run_cat_interaction_step starting at ``if cfg.include_numeric and numeric_raw_values:``."""
    from mlframe.feature_selection.filters.cat_interactions import resolve_max_combined_nbins

    if cfg.include_numeric and numeric_raw_values:
        # Cap the per-column bin count so a numeric x numeric pair fits the data-aware cardinality budget
        # (``nbins**2 <= max_combined_nbins``); otherwise EVERY numeric pair is rejected by the per-pair
        # ``nb_prod > max_combined`` gate and include_numeric silently produces nothing. ``floor(sqrt(budget))``
        # keeps ~the densest grid the Paninski ceiling allows (>= 2 so a median-split threshold cross survives).
        _budget_for_numeric = resolve_max_combined_nbins(cfg, n_samples)
        _num_nbins = int(getattr(cfg, "numeric_nbins", 10))
        _num_nbins = max(2, min(_num_nbins, math.isqrt(int(_budget_for_numeric))))
        _work_cols = list(cols)
        _extra_blocks: list = []
        _extra_nbins: list = []
        for _orig_idx, _raw in numeric_raw_values.items():
            _raw_arr = np.asarray(_raw, dtype=np.float64)
            if not np.isfinite(_raw_arr).all():
                # v1 skips NaN/inf-bearing numerics (the quantile-edge replay has no dedicated NaN bin).
                state.high_cardinality_warnings.append((int(_orig_idx), -1))
                continue
            _codes, _edges = _quantile_bin_with_edges(_raw_arr, _num_nbins)
            if _edges.size == 0:
                continue  # constant column -> no interaction signal
            _name = cols[int(_orig_idx)]
            numeric_candidate_idxs.append(len(_work_cols))
            _work_cols.append(_name)
            _extra_blocks.append(_codes.astype(data.dtype, copy=False).reshape(-1, 1))
            _extra_nbins.append(int(_codes.max()) + 1)
            numeric_edges_by_name[_name] = _edges
        if _extra_blocks:
            data = np.concatenate([data, np.concatenate(_extra_blocks, axis=1)], axis=1)
            nbins = np.concatenate([nbins, np.asarray(_extra_nbins, dtype=nbins.dtype)])
            cols = _work_cols
            if verbose:
                logger.info("cat-FE include_numeric: quantile-binned %d numeric column(s) into the candidate pool", len(_extra_blocks))
    return cols, data, nbins


def _run_cat_interactio_cache_active(cache_active, streaming_cache, data, candidate_idxs_arr, nbins, cfg, target_sig, verbose, classes_y, freqs_y, dtype, new_signatures):
    """Block of run_cat_interaction_step starting at ``if cache_active:``."""
    from mlframe.feature_selection.filters.cat_interactions import _column_signature, _marginal_screen_njit, _restore_cached_marginal_mis

    if cache_active:
        assert streaming_cache is not None  # cache_active requires streaming_cache is not None
        reusable_mask, mi_reused, new_signatures = _restore_cached_marginal_mis(
            factors_data=data, candidate_idxs=candidate_idxs_arr,
            nbins=nbins, cache=streaming_cache,
            kl_threshold=cfg.streaming_cache_kl_threshold,
            target_sig=target_sig,
        )
        n_reused = int(reusable_mask.sum())
        if verbose and n_reused:
            logger.info(
                "cat-FE streaming cache: reusing %d/%d cached marginal MIs",
                n_reused, len(candidate_idxs_arr),
            )
        # Compute MI only for the non-reusable cols
        if n_reused == len(candidate_idxs_arr):
            candidate_mi = mi_reused
        else:
            full_mi = _marginal_screen_njit(
                factors_data=data,
                candidate_idxs=candidate_idxs_arr,
                nbins=nbins,
                classes_y=classes_y,
                freqs_y=freqs_y,
                dtype=dtype,
            )
            # Splice: reuse cached where mask is True, full where False
            candidate_mi = np.where(reusable_mask, mi_reused, full_mi)
    else:
        candidate_mi = _marginal_screen_njit(
            factors_data=data,
            candidate_idxs=candidate_idxs_arr,
            nbins=nbins,
            classes_y=classes_y,
            freqs_y=freqs_y,
            dtype=dtype,
        )
        # Build signatures for next-fit cache (whether enabled or not)
        if getattr(cfg, "enable_streaming_cache", False):
            for _k, col_idx in enumerate(candidate_idxs_arr):
                new_signatures[int(col_idx)] = _column_signature(
                    data[:, int(col_idx)], int(nbins[int(col_idx)]),
                )
    return candidate_mi, new_signatures


def _run_cat_interactio_usability_gpu_py_batch(cfg, _gpu_disabled, candidate_idxs_arr, n_samples, verbose, use_gpu):
    """Block of run_cat_interaction_step starting at ``if cfg.backend == "gpu":``."""
    if cfg.backend == "gpu":
        if _gpu_disabled:
            raise RuntimeError("cat-FE: backend='gpu' requested but GPU is globally disabled " "(MLFRAME_DISABLE_GPU=1 / CUDA_VISIBLE_DEVICES=''). Set backend='cpu' or clear the opt-out.")
        # Bare `import cupy` succeeds on broken CUDA installs (cupy-cuda12x
        # against a CUDA-11 driver, renamed cublas/nvrtc DLLs, ...). Probe
        # via is_gpu_available() which compiles a kernel and catches the
        # RecursionError-loop that broken nvrtc DLLs trigger inside cupy's
        # _get_softlink retry path.
        from mlframe.feature_engineering.transformer import is_gpu_available
        if not is_gpu_available():
            raise RuntimeError("cat-FE: backend='gpu' requested but cupy/CUDA is not usable. " "Install cupy matching your CUDA toolkit, or set backend='cpu'.")
        use_gpu = True
    elif cfg.backend == "auto" and not _gpu_disabled:
        n_cols_eff = len(candidate_idxs_arr)
        if _cat_fe_auto_wants_gpu(n_samples, n_cols_eff):
            from mlframe.feature_engineering.transformer import is_gpu_available
            if is_gpu_available():
                use_gpu = True
            elif verbose:
                logger.info(
                    "cat-FE: backend='auto' wanted GPU at N=%d, n=%d " "but cupy is unavailable; falling back to CPU.",
                    n_cols_eff,
                    n_samples,
                )
    return use_gpu


def _run_cat_interactio_kernel_costs_extra_ops(weights, n_samples, use_weights):
    """Block of run_cat_interaction_step starting at ``if weights is not None and len(weights) == n_samples:``."""
    if weights is not None and len(weights) == n_samples:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.size > 0 and not np.allclose(weights, weights[0]):
            use_weights = True
    return use_weights, weights


def _run_cat_interactio_use_weights(use_weights, weights, data, candidate_idxs_arr, nbins, classes_y, dtype, marginal_mi_full):
    """Block of run_cat_interaction_step starting at ``if use_weights:``."""
    if use_weights:
        # marginal_mi_full above was built entirely via
        # the UNWEIGHTED _marginal_screen_njit - recompute it weighted now that use_weights is known True,
        # so the joint-vs-marginal difference (II) is consistent regardless of which kernel/backend
        # computes the joint term below. See _marginal_screen_weighted's docstring for the full rationale.
        assert weights is not None  # use_weights=True only when weights was matched to n_samples above
        from mlframe.feature_selection.filters.cat_interactions import _marginal_screen_weighted
        candidate_mi = _marginal_screen_weighted(
            factors_data=data,
            candidate_idxs=candidate_idxs_arr,
            nbins=nbins,
            classes_y=classes_y,
            weights=weights,
            dtype=dtype,
        )
        for _k, _idx in enumerate(candidate_idxs_arr):
            marginal_mi_full[int(_idx)] = candidate_mi[_k]


def _gpu_pair_search_or_none(cfg, gpu_fn, data, pairs_a, pairs_b, nbins, classes_y, freqs_y, dtype, marginal_mi_full):
    """GPU pair search -> ``(used_gpu, ii_arr, joint_mi_arr, n_uniq_arr)``.

    Under ``backend="auto"`` any GPU failure (allocation, launch, size guard) returns ``(False, None, None, None)`` so the caller runs the CPU kernel;
    an explicit ``backend="gpu"`` re-raises.
    """
    try:
        joint_mi_arr = gpu_fn(
            factors_data=data, pairs_a=pairs_a, pairs_b=pairs_b, factors_nbins=nbins, classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
        )
    except Exception as exc:
        if cfg.backend != "auto":
            raise
        logger.warning("cat-FE: GPU pair search failed (%s: %s); falling back to the CPU kernel", type(exc).__name__, exc)
        return False, None, None, None
    ii_arr = np.zeros(len(pairs_a), dtype=np.float64)
    n_uniq_arr = np.zeros(len(pairs_a), dtype=np.int64)
    for k in range(len(pairs_a)):
        i = int(pairs_a[k])
        j = int(pairs_b[k])
        ii_arr[k] = joint_mi_arr[k] - marginal_mi_full[i] - marginal_mi_full[j]
        n_uniq_arr[k] = int(nbins[i]) * int(nbins[j])
    return True, ii_arr, joint_mi_arr, n_uniq_arr


def _run_cat_interactio_use_gpu(use_gpu, use_weights, verbose, data, pairs_a, pairs_b, marginal_mi_full, nbins, classes_y, weights, dtype, freqs_y, ii_arr, joint_mi_arr, n_uniq_arr):
    """Block of run_cat_interaction_step starting at ``if not use_gpu:``."""
    from mlframe.feature_selection.filters.cat_interactions import _pair_search_kernel_njit, _pair_search_kernel_weighted_njit

    if not use_gpu:
        if use_weights:
            if verbose:
                logger.info("cat-FE: pair search with sample weights (CPU prange)")
            joint_mi_arr, ii_arr, n_uniq_arr = _pair_search_kernel_weighted_njit(
                factors_data=data,
                pairs_a=pairs_a, pairs_b=pairs_b,
                marginal_mi=marginal_mi_full,
                nbins=nbins,
                classes_y=classes_y,
                weights=np.asarray(weights, dtype=np.float64),
                dtype=dtype,
            )
        else:
            joint_mi_arr, ii_arr, n_uniq_arr = _pair_search_kernel_njit(
                factors_data=data,
                pairs_a=pairs_a, pairs_b=pairs_b,
                marginal_mi=marginal_mi_full,
                nbins=nbins,
                classes_y=classes_y,
                freqs_y=freqs_y,
                dtype=dtype,
            )
    return ii_arr, joint_mi_arr, n_uniq_arr


def _selection_keep_mask(cfg, ii_arr, selected_idx, floor):
    """Boolean mask of the selected pairs whose interaction information passes ``floor`` under ``cfg.select_on``."""
    if cfg.select_on == "synergy":
        keep = ii_arr[selected_idx] > floor
    elif cfg.select_on == "redundancy":
        keep = ii_arr[selected_idx] < floor
    else:  # absolute
        keep = np.abs(ii_arr[selected_idx]) > abs(floor)
    return keep


def _run_cat_interactio_drop_survivors_whose_lower(selected_idx, pairs_a, pairs_b, bootstrap_ci_dict, floor_ci, kept_after_ci, verbose):
    """Block of run_cat_interaction_step starting at ``for k in selected_idx:``."""
    for k in selected_idx:
        ij = (int(pairs_a[k]), int(pairs_b[k]))
        lower, _, _ = bootstrap_ci_dict.get(ij, (-np.inf, 0.0, np.inf))
        if lower >= floor_ci:
            kept_after_ci.append(k)
        elif verbose:
            logger.info(
                "cat-FE: pair %s dropped (bootstrap_lower_ci=%.4f < %.4f)",
                ij, lower, floor_ci,
            )


def _run_cat_interactio_phase_top_seeds_cheap(kway_results, pairs_a, selected_idx, verbose, _expand_seeds):
    """Block of run_cat_interaction_step starting at ``if not kway_results and len(pairs_a) > len(selected_idx):``."""
    if not kway_results and len(pairs_a) > len(selected_idx):
        if verbose:
            logger.info(
                "cat-FE: top-K seeds produced 0 k-way results; " "falling back to all %d pairs (quadratic cost)",
                len(pairs_a),
            )
        # Phase 2 (fallback): all pairs
        _expand_seeds(range(len(pairs_a)))


def _run_cat_interactio_diagnostics_always_cheap_gated(cfg, selected_idx, pairs_a, pairs_b, ii_arr, joint_mi_arr, marginal_mi_full, n_uniq_arr, cols, n_samples, confidence_dict, bootstrap_ci_dict, state, new_names):
    """Block of run_cat_interaction_step starting at ``if cfg.emit_diagnostics:``."""
    if cfg.emit_diagnostics:
        for k_out, k_in in enumerate(selected_idx):
            i = int(pairs_a[k_in])
            j = int(pairs_b[k_in])
            state.diagnostics[new_names[k_out]] = {
                "II": float(ii_arr[k_in]),
                "joint_MI": float(joint_mi_arr[k_in]),
                "marginal_X1_MI": float(marginal_mi_full[i]),
                "marginal_X2_MI": float(marginal_mi_full[j]),
                "joint_nclasses": int(n_uniq_arr[k_in]),
                "src_indices": (i, j),
                "src_names": (cols[i], cols[j]),
                "n_obs_per_cell_p25": float(n_samples / max(int(n_uniq_arr[k_in]), 1)),
                # Joint-dependence confidence: honest naming - this tests "(X1, X2) jointly independent of Y", not "no synergy".
                # ``None`` when no permutation test ran (full_npermutations=0).
                "joint_dependence_confidence": (float(confidence_dict[(i, j)]) if (i, j) in confidence_dict else None),
                # Bootstrap CI on II. ``None`` when disabled.
                "bootstrap_ii_ci": (bootstrap_ci_dict[(i, j)] if (i, j) in bootstrap_ci_dict else None),
            }


def _run_cat_interactio_each_pair_recipe_additionally(cfg, selected_idx, pairs_a, pairs_b, data, target_indices, classes_y, nbins, dtype, cols, new_names, new_recipes, state, verbose):
    """Block of run_cat_interaction_step starting at ``if cfg.emit_target_encoding:``."""
    from mlframe.feature_selection.filters.cat_interactions import _compute_target_encoding

    if cfg.emit_target_encoding:
        te_cols_list: list = []
        te_names: list = []
        te_recipes: list = []
        # Materialize TE alongside factorize-recipe survivors (pairs only; k-way TE is left as future work for simplicity).
        for k_out, k_in in enumerate(selected_idx):
            i = int(pairs_a[k_in])
            j = int(pairs_b[k_in])
            idx_tuple = (i, j)
            te_vals, cell_means_global = _compute_target_encoding(
                factors_data=data, idx_tuple=idx_tuple,
                target_indices=target_indices,
                classes_y=classes_y, nbins=nbins,
                n_oof_folds=cfg.target_encoding_oof_folds,
                smoothing=cfg.target_encoding_smoothing,
                dtype=dtype,
                seed=(i * 1000003 + j),  # deterministic per-interaction fold shuffle (decorrelates folds from row order)
            )
            # te_vals dtype is float64; we don't quantize - caller's downstream model handles continuous encoded values.
            te_name = f"te({cols[i]}__{cols[j]})"
            if te_name in cols or te_name in new_names or te_name in te_names:
                te_name = f"te_{k_out}({cols[i]}__{cols[j]})"
            te_cols_list.append(te_vals)
            te_names.append(te_name)
            # Build a target_encoding recipe with the cell-means table for transform() replay. Stored as ``extra``.
            te_recipes.append(
                EngineeredRecipe(
                    name=te_name,
                    kind="target_encoding",
                    src_names=(cols[i], cols[j]),
                    factorize_nbins=(int(nbins[i]), int(nbins[j])),
                    unknown_strategy=cfg.unknown_strategy,
                    extra={
                        "cell_means": cell_means_global,
                        "global_mean": float(classes_y.astype(np.float64).mean()),
                        "n_oof_folds": cfg.target_encoding_oof_folds,
                        "smoothing": cfg.target_encoding_smoothing,
                        # Need the factorize lookup to map (a, b) -> cell idx
                        "factorize_lookup": new_recipes[k_out].extra.get("lookup_table"),
                    },
                )
            )
        if te_cols_list:
            te_block = np.column_stack([v.astype(np.float64, copy=False) for v in te_cols_list])
            # Target-encoded cols are FLOAT, not ordinal. Add as a separate block; downstream screening will discretize them again per quantization_method. nbins[te_col]
            # is unknown until then; leave a sentinel that categorize_dataset will overwrite. We store these in state.diagnostics but NOT in the main data block
            # (the cat-FE pipeline assumes ordinal-encoded data; TE cols would need a quantization round-trip).
            state.diagnostics["__target_encoding__"] = {
                "te_block_shape": te_block.shape,
                "te_names": te_names,
            }
            state.recipes.extend(te_recipes)
            if verbose:
                logger.info(
                    "cat-FE: emitted %d target-encoded feature(s); "
                    "stored in state.recipes for transform() replay. "
                    "Note: TE cols are float, not in data_out; user "
                    "must round-trip through categorize_dataset to "
                    "include in MRMR screening.",
                    len(te_recipes),
                )


def _run_cat_interactio_value_instead_binning_through(numeric_edges_by_name, state):
    """Block of run_cat_interaction_step starting at ``if numeric_edges_by_name:``."""
    if numeric_edges_by_name:
        for _ri, _r in enumerate(state.recipes):
            _r_src_names = getattr(_r, "src_names", ())
            if _r_src_names is None:
                _r_src_names = ()
            _edges_for_recipe = {_src: numeric_edges_by_name[_src] for _src in _r_src_names if _src in numeric_edges_by_name}
            if _edges_for_recipe:
                state.recipes[_ri] = _r.with_extra(src_bin_edges=_edges_for_recipe)


def _run_cat_interactio_out_name_enumerate_new(new_names, n_orig, new_recipes, name_to_idx, state):
    """Block of run_cat_interaction_step starting at ``for k_out, _name in enumerate(new_names):``."""
    for k_out, _name in enumerate(new_names):
        eng_idx = n_orig + k_out
        # Parent indices come from the recipe's src_names. Recipes built here reference ORIGINAL data columns (no nested engineered parents yet).
        recipe = new_recipes[k_out]
        parent_idxs = [name_to_idx[src_name] for src_name in recipe.src_names if src_name in name_to_idx]
        if parent_idxs:
            state.lineage[eng_idx] = frozenset(parent_idxs)
