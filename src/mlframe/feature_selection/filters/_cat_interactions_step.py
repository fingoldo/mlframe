"""``run_cat_interaction_step`` carved out of
``mlframe.feature_selection.filters.cat_interactions``.

Re-imported at the parent's module bottom so historical
``from mlframe.feature_selection.filters.cat_interactions import run_cat_interaction_step``
resolves transparently.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from .cat_fe_state import CatFEConfig, CatFEState
from .engineered_recipes import EngineeredRecipe

logger = logging.getLogger(__name__)


def _quantile_bin_with_edges(raw: np.ndarray, n_bins: int) -> tuple:
    """Quantile-bin a 1-D numeric array into ``[0, n_bins)`` ordinal codes; return ``(codes, inner_edges)``.

    ``inner_edges`` are the ``n_bins - 1`` interior quantile cut points (unique-deduped); the bin code is
    ``np.searchsorted(inner_edges, value, side="right")`` -- the EXACT convention ``categorize_dataset``'s
    adaptive path uses (``_discretization_dataset.py``), so codes computed here at fit time are reproduced
    byte-for-byte by the recipe replay at transform time from the stored edges (no train/serve skew).

    Returns ``edges.size == 0`` for a constant / degenerate column (caller skips it: a 1-bin column carries
    no interaction signal). NaN-bearing columns must be filtered out by the caller -- this v1 edge scheme has
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

    Inputs:
    - ``data``: ordinal-encoded ``(n_samples, n_cols)`` produced by ``categorize_dataset``.
    - ``cols``: list of column names matching ``data`` shape.
    - ``nbins``: cardinality per column.
    - ``target_indices``, ``classes_y``, ``freqs_y``: precomputed by caller (avoids re-binning Y for every MI call).
    - ``categorical_vars``: indices into ``data`` of categorical (or pre-discretized numeric, when ``cfg.include_numeric=True``) columns to consider.
    - ``cfg``: the cat-FE config; ``cfg.enable=True`` is the user's opt-in switch (caller checks before calling us).

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
    from .cat_interactions import _anti_redundancy_rerank, _bootstrap_ii_cis, _column_signature, _compute_target_encoding, _compute_westfall_young_corrected_p, _confirm_pairs_bandit_ucb1, _confirm_pairs_via_permutation, _greedy_expand_one_seed, _kfold_stability_filter, _marginal_screen_njit, _materialize_kway, _materialize_pairs, _maybe_rerank_with_mm, _pair_search_kernel_njit, _pair_search_kernel_weighted_njit, _refine_kway_coordinate_ascent, _restore_cached_marginal_mis, _select_candidate_indices, _select_top_k_pairs, resolve_max_combined_nbins, resolve_min_interaction_information
    state = CatFEState()
    n_samples = data.shape[0]
    # Every early return below yields these ORIGINAL arrays; ``include_numeric`` (further down) shadows
    # data/cols/nbins with a transient working pool, so the originals are pinned here once.
    orig_data, orig_cols, orig_nbins = data, cols, nbins

    # ---- Pathological-input gates ----
    if target_indices.size == 0:
        raise ValueError("cat-FE: empty target_indices; cannot compute MI(X;Y).")
    if n_samples < cfg.min_n_samples:
        if verbose:
            logger.info(
                "cat-FE skipped: n_samples=%d < cfg.min_n_samples=%d",
                n_samples, cfg.min_n_samples,
            )
        return orig_data, orig_cols, orig_nbins, state

    # ---- Memmap detection ----
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

    # ---- include_numeric: quantile-bin eligible numeric columns into a transient working pool ----
    # Numeric columns are appended (quantile-coded, edges captured) to working copies of data / cols / nbins
    # and their positions joined to the candidate pool, so the existing pair / k-way machinery treats them
    # exactly like categoricals. The ORIGINAL data / cols / nbins are restored before the final concat, so the
    # numeric columns' own (MDLP) codes that flow to downstream screening are UNCHANGED -- only the engineered
    # cross columns are appended. The per-column quantile edges are stamped into each recipe (below) so
    # ``transform`` reproduces identical bin codes from raw test values (leak-free, no train/serve skew).
    numeric_candidate_idxs: list = []
    numeric_edges_by_name: dict = {}
    if cfg.include_numeric and numeric_raw_values:
        # Cap the per-column bin count so a numeric x numeric pair fits the data-aware cardinality budget
        # (``nbins**2 <= max_combined_nbins``); otherwise EVERY numeric pair is rejected by the per-pair
        # ``nb_prod > max_combined`` gate and include_numeric silently produces nothing. ``floor(sqrt(budget))``
        # keeps ~the densest grid the Paninski ceiling allows (>= 2 so a median-split threshold cross survives).
        _budget_for_numeric = resolve_max_combined_nbins(cfg, n_samples)
        _num_nbins = int(getattr(cfg, "numeric_nbins", 10))
        _num_nbins = max(2, min(_num_nbins, int(math.isqrt(int(_budget_for_numeric)))))
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

    # ---- Column-level validation ----
    candidate_idxs = _select_candidate_indices(
        nbins=nbins,
        categorical_vars=list(categorical_vars) + numeric_candidate_idxs,
        cfg=cfg, state=state,
        n_samples=n_samples,
    )
    if len(candidate_idxs) < 2:
        if verbose:
            logger.info(
                "cat-FE skipped: only %d eligible candidate columns after validation",
                len(candidate_idxs),
            )
        return orig_data, orig_cols, orig_nbins, state

    # ---- Marginal MI screen ----
    candidate_idxs_arr = np.asarray(candidate_idxs, dtype=np.int64)
    if verbose:
        logger.info("cat-FE: marginal MI screen over %d candidate columns", len(candidate_idxs))

    # Streaming cache check. If enabled AND cache provided, reuse cached marginal MIs for columns whose distribution hasn't drifted (KL < threshold).
    cache_active = getattr(cfg, "enable_streaming_cache", False) and streaming_cache is not None and streaming_cache  # non-empty
    # Content signature of the (discretized) target -- gates cache reuse so a changed Y invalidates the
    # cached MI(X;Y) even when X's distribution is unchanged.
    from .cat_interactions import _target_signature
    target_sig = _target_signature(data[:, target_indices])
    new_signatures: dict = {}
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

    # Persist updated cache for next fit() call
    if getattr(cfg, "enable_streaming_cache", False):
        state.streaming_cache_out = {
            "target_sig": target_sig,
            "col_signatures": new_signatures,
            "marginal_mis": {int(candidate_idxs_arr[k]): float(candidate_mi[k]) for k in range(len(candidate_idxs_arr))},
        }
    # marginal_floor prune
    if cfg.marginal_floor > 0:
        keep_mask = candidate_mi > cfg.marginal_floor
        candidate_idxs_arr = candidate_idxs_arr[keep_mask]
        candidate_mi = candidate_mi[keep_mask]
    if len(candidate_idxs_arr) < 2:
        if verbose:
            logger.info("cat-FE skipped: %d cols cleared marginal_floor", len(candidate_idxs_arr))
        return orig_data, orig_cols, orig_nbins, state

    # Build a marginal-MI lookup keyed by COLUMN INDEX (into data), so the pair kernel can look up by index without re-running the screen.
    marginal_mi_full = np.full(data.shape[1], np.nan, dtype=np.float64)
    for k, idx in enumerate(candidate_idxs_arr):
        marginal_mi_full[int(idx)] = candidate_mi[k]

    # ---- Pair enumeration with cardinality budget ----
    max_combined = resolve_max_combined_nbins(cfg, n_samples)
    pairs_a_list = []
    pairs_b_list = []
    for ii in range(len(candidate_idxs_arr)):
        for jj in range(ii + 1, len(candidate_idxs_arr)):
            i = int(candidate_idxs_arr[ii])
            j = int(candidate_idxs_arr[jj])
            nb_prod = int(nbins[i]) * int(nbins[j])
            if nb_prod > max_combined:
                continue
            # Strict int32-overflow gate: the inner merge_vars loop computes ``current_nclasses * sample_class`` in int32. With current_nclasses==nbins[i],
            # this multiplied by (nbins[j]-1) must stay below 2^31. Conservative: nb_prod < 2^31.
            if nb_prod >= 2**31:
                continue
            pairs_a_list.append(i)
            pairs_b_list.append(j)
    if not pairs_a_list:
        if verbose:
            logger.info("cat-FE skipped: 0 pairs cleared cardinality budget %d", max_combined)
        return orig_data, orig_cols, orig_nbins, state

    pairs_a = np.asarray(pairs_a_list, dtype=np.int64)
    pairs_b = np.asarray(pairs_b_list, dtype=np.int64)
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
    use_gpu = False
    if cfg.backend == "gpu":
        # Bare `import cupy` succeeds on broken CUDA installs (cupy-cuda12x
        # against a CUDA-11 driver, renamed cublas/nvrtc DLLs, ...). Probe
        # via is_gpu_available() which compiles a kernel and catches the
        # RecursionError-loop that broken nvrtc DLLs trigger inside cupy's
        # _get_softlink retry path.
        from mlframe.feature_engineering.transformer import is_gpu_available
        if not is_gpu_available():
            raise RuntimeError("cat-FE: backend='gpu' requested but cupy/CUDA is not usable. " "Install cupy matching your CUDA toolkit, or set backend='cpu'.")
        use_gpu = True
    elif cfg.backend == "auto":
        n_cols_eff = len(candidate_idxs_arr)
        if n_cols_eff >= 200 and n_samples >= 500_000:
            from mlframe.feature_engineering.transformer import is_gpu_available
            if is_gpu_available():
                use_gpu = True
            elif verbose:
                logger.info(
                    "cat-FE: backend='auto' wanted GPU at N=%d, n=%d " "but cupy is unavailable; falling back to CPU.",
                    n_cols_eff,
                    n_samples,
                )

    # Choose weighted vs unweighted kernel. Use weighted only when weights are actually non-uniform; uniform weights are equivalent to unweighted and the weighted
    # kernel costs extra ops, so skip in that case.
    use_weights = False
    if weights is not None and len(weights) == n_samples:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.size > 0 and not np.allclose(weights, weights[0]):
            use_weights = True
    if use_gpu:
        from .gpu import mi_direct_gpu_batched_pairs
        if use_weights:
            logger.warning("cat-FE: sample weights ignored on GPU path; " "falling back to CPU weighted kernel.")
            use_gpu = False
        else:
            if verbose:
                logger.info("cat-FE: pair search on GPU over %d pairs", len(pairs_a))
            joint_mi_arr = mi_direct_gpu_batched_pairs(
                factors_data=data,
                pairs_a=pairs_a, pairs_b=pairs_b,
                factors_nbins=nbins,
                classes_y=classes_y, freqs_y=freqs_y,
                dtype=dtype,
            )
            ii_arr = np.zeros(len(pairs_a), dtype=np.float64)
            n_uniq_arr = np.zeros(len(pairs_a), dtype=np.int64)
            for k in range(len(pairs_a)):
                i = int(pairs_a[k])
                j = int(pairs_b[k])
                ii_arr[k] = joint_mi_arr[k] - marginal_mi_full[i] - marginal_mi_full[j]
                n_uniq_arr[k] = int(nbins[i]) * int(nbins[j])
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

    # ---- Top-K selection ----
    selected_idx = _select_top_k_pairs(
        ii_arr=ii_arr,
        pairs_a=pairs_a, pairs_b=pairs_b,
        cfg=cfg, n_samples=n_samples,
    )
    if len(selected_idx) == 0:
        if verbose:
            logger.info("cat-FE: 0 pairs cleared min_interaction_information; no engineered cols")
        return orig_data, orig_cols, orig_nbins, state
    if verbose:
        logger.info("cat-FE: %d pair(s) selected for materialisation", len(selected_idx))

    # ---- Miller-Madow re-rank on top-K survivors ----
    # Only top-K pay the 5x merge_vars cost; saves N^2-scale compute while catching high-cardinality bias-driven false positives. The re-rank may shuffle the heap;
    # ``selected_idx`` order matters for downstream ``_materialize_pairs`` because it determines which engineered cols are added first (relevant when ``top_k_pairs``
    # cap binds).
    ii_arr, selected_idx = _maybe_rerank_with_mm(
        factors_data=data,
        pairs_a=pairs_a, pairs_b=pairs_b,
        selected_idx=selected_idx, ii_arr=ii_arr,
        nbins=nbins, target_indices=target_indices,
        classes_y=classes_y, freqs_y=freqs_y,
        cfg=cfg, dtype=dtype, verbose=verbose,
    )
    # After MM re-rank, some pairs may drop below the floor; re-filter.
    floor = resolve_min_interaction_information(cfg, n_samples)
    if cfg.select_on == "synergy":
        keep = ii_arr[selected_idx] > floor
    elif cfg.select_on == "redundancy":
        keep = ii_arr[selected_idx] < floor
    else:  # absolute
        keep = np.abs(ii_arr[selected_idx]) > abs(floor)
    selected_idx = selected_idx[keep]
    if len(selected_idx) == 0:
        if verbose:
            logger.info("cat-FE: 0 pairs survived MM re-rank floor; no engineered cols")
        return orig_data, orig_cols, orig_nbins, state

    # ---- Anti-redundancy re-rank (opt-in via anti_redundancy_beta>0) ----
    # Adjusts each survivor's score by ``beta * mean_z I(merged; Z)`` where Z ranges over already-selected features in ``selected_so_far``. No-op when beta=0 or selected_so_far is empty.
    if selected_so_far:
        ii_arr, selected_idx = _anti_redundancy_rerank(
            factors_data=data, pairs_a=pairs_a, pairs_b=pairs_b,
            selected_idx=selected_idx, ii_arr=ii_arr,
            nbins=nbins, selected_so_far=selected_so_far,
            classes_y=classes_y, cfg=cfg, dtype=dtype, verbose=verbose,
        )
        # Re-apply the floor after anti-redundancy correction
        floor = resolve_min_interaction_information(cfg, n_samples)
        if cfg.select_on == "synergy":
            keep = ii_arr[selected_idx] > floor
        elif cfg.select_on == "redundancy":
            keep = ii_arr[selected_idx] < floor
        else:
            keep = np.abs(ii_arr[selected_idx]) > abs(floor)
        selected_idx = selected_idx[keep]
        if len(selected_idx) == 0:
            if verbose:
                logger.info("cat-FE: 0 pairs survived anti-redundancy floor")
            return orig_data, orig_cols, orig_nbins, state

    # ---- K-fold II stability filter (opt-in via n_folds_stability>0) ----
    # Drops pairs whose II is unstable across K folds (signal driven by outlier rows). Runs BEFORE permutation so we don't pay perm budget on pairs that fail stability.
    selected_idx, per_fold_ii_dict = _kfold_stability_filter(
        factors_data=data, pairs_a=pairs_a, pairs_b=pairs_b,
        selected_idx=selected_idx, nbins=nbins,
        target_indices=target_indices,
        cfg=cfg, dtype=dtype, verbose=verbose,
    )
    # Surface fold IIs into the state UNCONDITIONALLY -- even when 0 pairs survived, the per-fold values are useful diagnostics (user can see WHY the pair was rejected).
    # Must happen BEFORE the early return.
    if per_fold_ii_dict:
        state.ii_stability.update(per_fold_ii_dict)
    if len(selected_idx) == 0:
        if verbose:
            logger.info("cat-FE: 0 pairs survived K-fold stability filter")
        return orig_data, orig_cols, orig_nbins, state

    # ---- Permutation confirmation + FWER correction ----
    # Runs only when ``cfg.full_npermutations > 0`` (default 100). Tests joint-independence null; failed pairs are dropped from ``selected_idx``. The resulting
    # ``confidence_dict`` is surfaced via diagnostics for user inspection. ``n_search_pairs`` is the family size for FWER correction -- the count of pairs CONSIDERED
    # in the search phase, NOT the top-K count.
    # Full Westfall-Young requires the per-shuffle max-II across ALL search pairs, materially more expensive than per-survivor permutation. To get the full WY
    # behaviour, we substitute the per-pair p-values from the joint-independence test with the WY-corrected versions BEFORE the orchestrator applies the floor.
    # Memory budget: full WY pre-merges m * n int32 cells; if that exceeds e.g. 500 MB we fall back to Bonferroni-on-survivors via the _apply_fwer_correction path.
    use_full_wy = cfg.fwer_correction == "westfall_young" and cfg.full_npermutations > 0 and len(pairs_a) * n_samples * 4 < 500 * 1024 * 1024

    # Bandit UCB1 budget allocation overrides the fixed path when cfg.perm_budget_strategy='bandit_ucb1' (and full_npermutations>0 AND not using full WY which has its own coordination).
    if getattr(cfg, "perm_budget_strategy", "fixed") == "bandit_ucb1" and cfg.full_npermutations > 0 and not use_full_wy:
        selected_idx, confidence_dict = _confirm_pairs_bandit_ucb1(
            factors_data=data, pairs_a=pairs_a, pairs_b=pairs_b,
            selected_idx=selected_idx, ii_arr=ii_arr,
            nbins=nbins, classes_y=classes_y, freqs_y=freqs_y,
            cfg=cfg, n_search_pairs=len(pairs_a),
            dtype=dtype, verbose=verbose,
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
        )
    if len(selected_idx) == 0:
        if verbose:
            logger.info("cat-FE: 0 pairs cleared permutation confirmation")
        return orig_data, orig_cols, orig_nbins, state

    # ---- Bootstrap CIs on II (opt-in via bootstrap_ci_n_replicates>0) ----
    bootstrap_ci_dict = _bootstrap_ii_cis(
        factors_data=data, pairs_a=pairs_a, pairs_b=pairs_b,
        selected_idx=selected_idx,
        nbins=nbins, classes_y=classes_y, freqs_y=freqs_y,
        cfg=cfg, dtype=dtype, verbose=verbose,
    )
    if bootstrap_ci_dict:
        # Drop survivors whose lower-CI < floor (unstable signal)
        floor_ci = resolve_min_interaction_information(cfg, n_samples)
        kept_after_ci = []
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
        selected_idx = np.asarray(kept_after_ci, dtype=selected_idx.dtype)
        if len(selected_idx) == 0:
            if verbose:
                logger.info("cat-FE: 0 pairs survived bootstrap CI floor")
            return orig_data, orig_cols, orig_nbins, state

    # ---- K-way greedy expansion (opt-in via max_kway_order > 2) ----
    # HYBRID seeding -- first try only the top-K confirmed pairs, which is O(top_k * N) = ~6400 merge_vars at top_k=64, N=100. If that produces ZERO k-way results
    # (the pure-k-way-XOR case where all 2-way IIs are noise around 0 and top-K is random), fall back to seeding from ALL pairs. This avoids the quadratic cost
    # when the signal is detectable from top-K, but preserves the all-pairs path for pathological niches.
    kway_results: list = []
    if cfg.max_kway_order > 2:
        min_inc_ii = resolve_min_interaction_information(cfg, n_samples)
        candidate_pool_arr = candidate_idxs_arr
        seen_kway: set = set()

        def _expand_seeds(seed_pair_indices):
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
                )
                if extension is None:
                    continue
                idx_tuple = extension[0]
                if idx_tuple in seen_kway:
                    continue
                seen_kway.add(idx_tuple)
                kway_results.append(extension)

        # Phase 1: top-K seeds (cheap, ~O(top_k * N) merge_vars)
        _expand_seeds(list(selected_idx))
        if not kway_results and len(pairs_a) > len(selected_idx):
            if verbose:
                logger.info(
                    "cat-FE: top-K seeds produced 0 k-way results; " "falling back to all %d pairs (quadratic cost)",
                    len(pairs_a),
                )
            # Phase 2 (fallback): all pairs
            _expand_seeds(range(len(pairs_a)))

        # Sort k-way results by joint_MI desc and cap by top_k_pairs.
        # Wave 58 (2026-05-20): secondary key on the var-index tuple so tied
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
                candidate_pool=candidate_idxs_arr, nbins=nbins,
                classes_y=classes_y, freqs_y=freqs_y,
                max_combined_nbins=max_combined,
                n_passes=cfg.refine_passes,
                dtype=dtype, verbose=verbose,
            )

    # ---- Materialise pair survivors (single concat) ----
    new_data_block, new_names, new_nbins, new_recipes = _materialize_pairs(
        factors_data=data,
        pairs_a=pairs_a, pairs_b=pairs_b,
        selected_idx=selected_idx,
        nbins=nbins,
        cols=cols,
        dtype=dtype,
        unknown_strategy=cfg.unknown_strategy,
    )

    # ---- Materialise k-way survivors (alongside pairs) ----
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
    # Diagnostics (always cheap, gated by cfg).
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
                # Joint-dependence confidence: honest naming -- this tests "(X1, X2) jointly independent of Y", not "no synergy".
                # ``None`` when no permutation test ran (full_npermutations=0).
                "joint_dependence_confidence": (float(confidence_dict[(i, j)]) if (i, j) in confidence_dict else None),
                # Bootstrap CI on II. ``None`` when disabled.
                "bootstrap_ii_ci": (bootstrap_ci_dict[(i, j)] if (i, j) in bootstrap_ci_dict else None),
            }

    state.recipes.extend(new_recipes)

    # ---- Target encoding emit (opt-in) ----
    # For each pair recipe, additionally emit a target-encoded col with OOF CV-aware shrinkage. Recipes carry ``kind="target_encoding"`` with the global cell-means table for transform() replay.
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
            # te_vals dtype is float64; we don't quantize -- caller's downstream model handles continuous encoded values.
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

    # ---- Stamp quantile bin edges onto recipes built from numeric sources (leak-safe transform replay) ----
    # Without this, ``_apply_factorize`` / ``_apply_target_encoding`` would ``astype(int64)`` a raw numeric test
    # value (3.7 -> 3) instead of binning it through the fit-time quantile edges -> a silent train/serve skew.
    if numeric_edges_by_name:
        for _ri, _r in enumerate(state.recipes):
            _edges_for_recipe = {_src: numeric_edges_by_name[_src] for _src in (getattr(_r, "src_names", ()) or ()) if _src in numeric_edges_by_name}
            if _edges_for_recipe:
                state.recipes[_ri] = _r.with_extra(src_bin_edges=_edges_for_recipe)

    # ---- Single concat onto data / cols / nbins ----
    # Restore the ORIGINAL arrays: the engineered cross block is appended to them, NOT to the include_numeric
    # working pool (whose transient quantile-coded numeric columns must never reach downstream screening).
    data, cols, nbins = orig_data, orig_cols, orig_nbins
    data_out = np.concatenate([data, new_data_block], axis=1)
    cols_out = list(cols) + new_names
    nbins_out = np.concatenate([nbins, np.asarray(new_nbins, dtype=nbins.dtype)])

    # ---- Build engineered_lineage map ----
    # Engineered cols land at indices [n_orig, n_orig + len(new_names)). For each, record the parent indices (in the ORIGINAL data layout) so screen_predictors
    # can skip ``(orig_parent, engineered_col)`` k-way candidates -- they're redundant by construction.
    n_orig = data.shape[1]
    name_to_idx = {n: i for i, n in enumerate(cols)}  # original col name -> idx
    state.lineage = {}
    for k_out, _name in enumerate(new_names):
        eng_idx = n_orig + k_out
        # Parent indices come from the recipe's src_names. Recipes built here reference ORIGINAL data columns (no nested engineered parents yet).
        recipe = new_recipes[k_out]
        parent_idxs = []
        for src_name in recipe.src_names:
            if src_name in name_to_idx:
                parent_idxs.append(name_to_idx[src_name])
        if parent_idxs:
            state.lineage[eng_idx] = frozenset(parent_idxs)

    return data_out, cols_out, nbins_out, state
