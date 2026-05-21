"""Categorical-feature interaction generator for ``MRMR``.

For each pair of categorical (or pre-discretized numeric) columns ``(X_i, X_j)``, compute the Jakulin Interaction Information

::

    II(X_i ; X_j ; Y) = I(X_i, X_j ; Y) - I(X_i ; Y) - I(X_j ; Y)

and keep the top-K pairs whose II indicates *synergy* (positive II: the joint tells the target more than the sum of marginals -- the canonical XOR-style hidden pair).
Surviving pairs become new ordinal-encoded columns appended to ``data`` / ``cols`` / ``nbins``, and recipes (``EngineeredRecipe(kind="factorize")``) land in
``self._cat_fe_state_.recipes`` so ``MRMR.transform`` can replay them on test data.

Features hooked off ``CatFEConfig``: Miller-Madow correction across the six entropies, two-stage permutation budget with same-shuffle three-MI test, Westfall-Young
multi-test correction, greedy k-way expansion to triplets / quartets, K-fold II stability filter, anti-redundancy vs already-selected features, GPU dispatch shim
``mi_direct_gpu_batched_pairs``.

References
----------
* Jakulin & Bratko 2003, *Quantifying and Visualizing Attribute Interactions*
* Williams & Beer 2010 (PID -- documented limitation)
* Paninski 2003 *Estimation of Entropy and Mutual Information* (Miller-Madow, bias formulas, sample-size guidance)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np
from numba import njit, prange

from .cat_fe_state import CatFEConfig, CatFEState
from .engineered_recipes import EngineeredRecipe
from .info_theory import (
    compute_mi_from_classes,
    entropy,
    entropy_miller_madow,
    merge_vars,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Streaming / incremental fit cache
#
# Cache marginal MIs and per-column distribution signatures across fit() calls. On re-fit with same CatFEConfig + similar data, skip recomputation for columns
# whose distribution hasn't drifted (KL < tau). Saves ~70% on production daily-refresh re-fits.
# ============================================================================


def _column_signature(values: np.ndarray, nbins: int) -> np.ndarray:
    """Per-column distribution signature ``bincount / n``, used for KL-based cache invalidation."""
    n = max(len(values), 1)
    counts = np.bincount(values.astype(np.int64), minlength=nbins).astype(np.float64)
    return counts / n


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """KL(p || q) with epsilon smoothing for zero cells."""
    p_safe = p + eps
    q_safe = q + eps
    p_safe = p_safe / p_safe.sum()
    q_safe = q_safe / q_safe.sum()
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def _restore_cached_marginal_mis(
    factors_data: np.ndarray,
    candidate_idxs: np.ndarray,
    nbins: np.ndarray,
    cache: dict,
    kl_threshold: float,
) -> tuple:
    """Decide per candidate column whether the cached marginal MI is reusable; returns ``(reusable_mask, marginal_mi_reused, new_signatures)``.

    The mask is True for columns whose signature KL-divergence vs the cached version is below ``kl_threshold`` -- those rows reuse the cached MI and skip the screen pass.
    """
    reusable_mask = np.zeros(len(candidate_idxs), dtype=bool)
    marginal_mi_reused = np.full(len(candidate_idxs), np.nan, dtype=np.float64)
    new_signatures: dict = {}
    cached_sigs = cache.get("col_signatures", {})
    cached_mis = cache.get("marginal_mis", {})
    for k, col_idx in enumerate(candidate_idxs):
        col_int = int(col_idx)
        sig = _column_signature(factors_data[:, col_int], int(nbins[col_int]))
        new_signatures[col_int] = sig
        if col_int in cached_sigs and col_int in cached_mis:
            kl = _kl_divergence(sig, cached_sigs[col_int])
            if kl < kl_threshold:
                reusable_mask[k] = True
                marginal_mi_reused[k] = cached_mis[col_int]
    return reusable_mask, marginal_mi_reused, new_signatures


# ============================================================================
# Default resolution (data-aware)
# ============================================================================


def resolve_max_combined_nbins(
    cfg: CatFEConfig, n_samples: int, hard_cap: int = 10_000_000
) -> int:
    """Resolve ``cfg.max_combined_nbins`` to a concrete int.

    ``None`` -> Paninski-derived data-aware ceiling: ``max(4, int(n * 0.05 / 3) + 1)``. Empirically this keeps per-cell observation count above ~3 at the sample sizes
    where MI estimation stops being noise.

    Always clamped to ``hard_cap`` (10**7) regardless of user value -- prevents OOM via misconfig like ``max_combined_nbins=10**9`` (4 GB freqs allocation).
    """
    if cfg.max_combined_nbins is None:
        # Paninski bias ~ (k-1)/(2n) per entropy term. For 0.05 nat tolerance across 3 entropies: 3*(k-1)/(2n) < 0.05 -> k < n*0.05/1.5 + 1 Р Р†РІР‚В°РІвЂљВ¬ n/30 + 1.
        # Default tolerance 0.05 is folklore, not analytical.
        resolved = max(4, int(n_samples * 0.05 / 3) + 1)
    else:
        resolved = int(cfg.max_combined_nbins)
    return min(resolved, hard_cap)


def resolve_min_interaction_information(
    cfg: CatFEConfig, n_samples: int
) -> float:
    """Resolve ``cfg.min_interaction_information`` to a concrete float.

    ``None`` -> ``-3 / sqrt(n)`` -- small-negative absorbs finite-sample noise around the synergy boundary so that pure k-way XOR (where all 2-way IIs are 0 in
    expectation but noisy) can still bubble survivors to the heap.
    """
    if cfg.min_interaction_information is None:
        return -3.0 / math.sqrt(max(n_samples, 1))
    return float(cfg.min_interaction_information)


# ============================================================================
# Validation gates -- run BEFORE the heavy kernels
# ============================================================================


def _select_candidate_indices(
    nbins: np.ndarray,
    categorical_vars: list,
    cfg: CatFEConfig,
    state: CatFEState,
    n_samples: int,
) -> list:
    """Filter candidate column indices to those eligible for cat-FE.

    Drops:
    - constants / all-NaN (``nbins[i] == 1``)
    - high-cardinality (``nbins[i] > sqrt(n) * 2``)

    Side effects: records dropped names in ``state.dropped_singleton_nbins`` and ``state.high_cardinality_warnings`` for downstream debugging.

    Returns the surviving index list.
    """
    high_card_threshold = math.sqrt(n_samples) * 2
    kept: list = []
    for idx in categorical_vars:
        nb = int(nbins[idx])
        if nb <= 1:
            state.dropped_singleton_nbins.append(int(idx))
            continue
        if nb > high_card_threshold:
            state.high_cardinality_warnings.append((int(idx), nb))
            # Refuse rather than warn: legacy upstream truncation to int16 already silently mangles >32k-cardinality cols, so downstream MI is on garbage.
            # Hard error gives the user a clear "this column shouldn't be cat".
            raise ValueError(
                f"cat-FE: column index {idx} has nbins={nb}, exceeding the "
                f"safe ceiling sqrt(n)*2={high_card_threshold:.0f} for "
                f"n={n_samples}. High-cardinality categorical columns "
                f"(IDs, hashes, free-text) produce unstable MI estimates "
                f"and likely violate the int16 ceiling upstream. "
                f"Drop the column or reconsider whether it's truly categorical."
            )
        kept.append(int(idx))
    return kept


# ============================================================================
# Marginal MI screen (njit prange over candidate columns)
# ============================================================================


@njit(parallel=True, cache=True)
def _marginal_screen_njit(
    factors_data: np.ndarray,
    candidate_idxs: np.ndarray,
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    dtype,
) -> np.ndarray:
    """Compute ``I(X_i ; Y)`` for every ``i`` in ``candidate_idxs``.

    Runs ``prange`` over candidates -- each thread merges its column into ``classes_x`` independently, then computes MI. No per-thread state shared.

    Returns a 1-D float64 array of length ``len(candidate_idxs)``, aligned with the input order. Cells for unmergeable / zero-MI columns simply produce ``0.0``
    (downstream logic handles those via ``cfg.marginal_floor``).
    """
    n_candidates = len(candidate_idxs)
    out = np.zeros(n_candidates, dtype=np.float64)
    for k in prange(n_candidates):
        idx = candidate_idxs[k]
        # Build a single-element vars_indices array; merge_vars handles k=1 as a degenerate case (just renumbers the column densely).
        vi = np.empty(1, dtype=np.int64)
        vi[0] = idx
        classes_x, freqs_x, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=vi,
            var_is_nominal=None,
            factors_nbins=nbins,
            dtype=dtype,
        )
        out[k] = compute_mi_from_classes(
            classes_x=classes_x,
            freqs_x=freqs_x,
            classes_y=classes_y,
            freqs_y=freqs_y,
            dtype=dtype,
        )
    return out


# ============================================================================
# Pair search (njit prange over candidate pairs)
# ============================================================================


@njit(parallel=True, cache=True)
def _pair_search_kernel_njit(
    factors_data: np.ndarray,
    pairs_a: np.ndarray,        # (n_pairs,) int -- left column index
    pairs_b: np.ndarray,        # (n_pairs,) int -- right column index
    marginal_mi: np.ndarray,    # (n_cols_in_data,) -- I(X_i; Y) for ALL cols
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    dtype,
) -> tuple:
    """For each pair ``(a, b)`` compute joint MI and Jakulin II.

    Returns ``(joint_mi_arr, ii_arr, n_uniq_arr)`` -- three 1-D arrays of length ``len(pairs_a)`` aligned with the input order.

    Optimisation: instead of calling ``merge_vars`` per pair (which allocates per-call), we compute the joint code directly as
    ``code = data[row, i] + data[row, j] * nbins[i]``. This works because for plug-in MI estimation, empty cells contribute 0 to entropy and the dense-renumbering
    done by ``merge_vars`` doesn't affect MI (MI is invariant under bijective relabeling of the alphabet). Joint histogram is built in-place into a thread-local
    buffer of size ``nbins[i] * nbins[j] * nbins_y``. Cuts the per-pair cost ~3-5x vs the merge_vars path by eliminating the renumber + lookup table phase per pair.
    """
    n_pairs = len(pairs_a)
    n_samples = factors_data.shape[0]
    nbins_y = int(classes_y.max()) + 1 if classes_y.size > 0 else 1
    joint_mi_out = np.zeros(n_pairs, dtype=np.float64)
    ii_out = np.zeros(n_pairs, dtype=np.float64)
    n_uniq_out = np.zeros(n_pairs, dtype=np.int64)
    inv_n = 1.0 / float(n_samples)

    for k in prange(n_pairs):
        i = pairs_a[k]
        j = pairs_b[k]
        nba = int(nbins[i])
        nbb = int(nbins[j])
        merged_card = nba * nbb

        # Build joint histogram (merged, Y). Thread-local buffer.
        joint_hist = np.zeros(merged_card * nbins_y, dtype=np.int64)
        # Per-pair marginals so we don't pay another pass.
        m_merged = np.zeros(merged_card, dtype=np.int64)
        for row in range(n_samples):
            va = factors_data[row, i]
            vb = factors_data[row, j]
            code = va + vb * nba
            cy = classes_y[row]
            joint_hist[code * nbins_y + cy] += 1
            m_merged[code] += 1

        # Compute MI: I(merged; Y) = H(merged) + H(Y) - H(merged, Y). Direct formula sum jc/n * log(jc * n / (m_m * m_y)); m_y is freqs_y * n_samples.
        mi = 0.0
        n_uniq = 0
        for m in range(merged_card):
            mm = m_merged[m]
            if mm == 0:
                continue
            n_uniq += 1
            for y in range(nbins_y):
                jc = joint_hist[m * nbins_y + y]
                if jc == 0:
                    continue
                # freqs_y[y] is a probability; multiply by n_samples to recover the count form.
                my = freqs_y[y] * n_samples
                if my <= 0:
                    continue
                jf = jc * inv_n
                mi += jf * np.log(jc * n_samples / (mm * my))

        joint_mi_out[k] = mi
        ii_out[k] = mi - marginal_mi[i] - marginal_mi[j]
        n_uniq_out[k] = n_uniq
    return joint_mi_out, ii_out, n_uniq_out


# Miller-Madow bias correction for pair-II moved to
# ``_cat_mm_correction.py``; re-exported below so the orchestrator
# continues to call them via the same names. See sibling for SSOT.
from ._cat_mm_correction import (  # noqa: E402,F401
    _entropy_for_mode,
    _should_apply_mm_for_pair_analytical, _should_apply_mm_for_pair,
    _compute_pair_ii_mm, _maybe_rerank_with_mm,
)
# UCB1-bandit pair-confirmation moved to ``_cat_confirm_bandit.py``;
# re-exported below so the orchestrator continues to call it via the same
# name. See sibling for SSOT.
from ._cat_confirm_bandit import _confirm_pairs_bandit_ucb1  # noqa: E402,F401
# Target encoding + weighted pair-search kernel + group-aware shuffle moved
# to ``_cat_target_encoding_and_weighted.py``; re-exported below so the
# orchestrator continues to call them via the same names. See sibling for
# SSOT.
from ._cat_target_encoding_and_weighted import (  # noqa: E402,F401
    _compute_target_encoding,
    _pair_search_kernel_weighted_njit,
    _group_aware_shuffle,
)
# Permutation-based pair-confirmation moved to
# ``_cat_confirm_permutation.py``; re-exported below so the orchestrator
# continues to call it via the same name. See sibling for SSOT.
from ._cat_confirm_permutation import (  # noqa: E402,F401
    _full_conditional_shuffle_ipf, _conditional_shuffle_within_strata,
    _count_nfailed_joint_indep_prange, _count_nfailed_joint_indep_cupy,
    _perm_kernel_dispatch_use_gpu,
    _shuffle_and_compute_three_mis, _bulk_shuffle_and_compute_three_mis,
    _compute_westfall_young_corrected_p, _apply_fwer_correction,
    _confirm_pairs_via_permutation,
)
# Post-screen refinement moved to ``_cat_post_refine.py``; re-exported
# below so the orchestrator continues to call them via the same names.
# See sibling for SSOT.
from ._cat_post_refine import (  # noqa: E402,F401
    _bootstrap_ii_cis, _anti_redundancy_rerank,
    _kfold_stability_filter, _refine_kway_coordinate_ascent,
)
# K-way expansion + pair/k-way materialisation moved to
# ``_cat_kway_materialize.py``; re-exported below so the orchestrator
# continues to call them via the same names. See sibling for SSOT.
from ._cat_kway_materialize import (  # noqa: E402,F401
    _greedy_expand_one_seed, _build_kway_chained_lookup, _materialize_kway,
    _select_top_k_pairs, _build_factorize_lookup, _materialize_pairs,
)


# ============================================================================
# Orchestrator
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
    selected_so_far: list = None,
    weights: np.ndarray = None,    # Per-row sample weights; None = uniform.
    streaming_cache: dict = None,  # Prior-fit cache for incremental re-fit.
    dtype=np.int32,
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
    state = CatFEState()
    n_samples = data.shape[0]

    # ---- Pathological-input gates ----
    if target_indices.size == 0:
        raise ValueError("cat-FE: empty target_indices; cannot compute MI(X;Y).")
    if n_samples < cfg.min_n_samples:
        if verbose:
            logger.info(
                "cat-FE skipped: n_samples=%d < cfg.min_n_samples=%d",
                n_samples, cfg.min_n_samples,
            )
        return data, cols, nbins, state

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

    # ---- Column-level validation ----
    candidate_idxs = _select_candidate_indices(
        nbins=nbins,
        categorical_vars=categorical_vars,
        cfg=cfg, state=state,
        n_samples=n_samples,
    )
    if len(candidate_idxs) < 2:
        if verbose:
            logger.info(
                "cat-FE skipped: only %d eligible candidate columns after validation",
                len(candidate_idxs),
            )
        return data, cols, nbins, state

    # ---- Marginal MI screen ----
    candidate_idxs_arr = np.asarray(candidate_idxs, dtype=np.int64)
    if verbose:
        logger.info("cat-FE: marginal MI screen over %d candidate columns", len(candidate_idxs))

    # Streaming cache check. If enabled AND cache provided, reuse cached marginal MIs for columns whose distribution hasn't drifted (KL < threshold).
    cache_active = (
        getattr(cfg, "enable_streaming_cache", False)
        and streaming_cache is not None
        and streaming_cache  # non-empty
    )
    new_signatures: dict = {}
    if cache_active:
        reusable_mask, mi_reused, new_signatures = _restore_cached_marginal_mis(
            factors_data=data, candidate_idxs=candidate_idxs_arr,
            nbins=nbins, cache=streaming_cache,
            kl_threshold=cfg.streaming_cache_kl_threshold,
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
            "col_signatures": new_signatures,
            "marginal_mis": {
                int(candidate_idxs_arr[k]): float(candidate_mi[k])
                for k in range(len(candidate_idxs_arr))
            },
        }
    # marginal_floor prune
    if cfg.marginal_floor > 0:
        keep_mask = candidate_mi > cfg.marginal_floor
        candidate_idxs_arr = candidate_idxs_arr[keep_mask]
        candidate_mi = candidate_mi[keep_mask]
    if len(candidate_idxs_arr) < 2:
        if verbose:
            logger.info("cat-FE skipped: %d cols cleared marginal_floor", len(candidate_idxs_arr))
        return data, cols, nbins, state

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
        return data, cols, nbins, state

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
        from mlframe.feature_engineering.transformer._utils import is_gpu_available
        if not is_gpu_available():
            raise RuntimeError(
                "cat-FE: backend='gpu' requested but cupy/CUDA is not usable. "
                "Install cupy matching your CUDA toolkit, or set backend='cpu'."
            )
        use_gpu = True
    elif cfg.backend == "auto":
        n_cols_eff = len(candidate_idxs_arr)
        if n_cols_eff >= 200 and n_samples >= 500_000:
            from mlframe.feature_engineering.transformer._utils import is_gpu_available
            if is_gpu_available():
                use_gpu = True
            elif verbose:
                logger.info(
                    "cat-FE: backend='auto' wanted GPU at N=%d, n=%d "
                    "but cupy is unavailable; falling back to CPU.",
                    n_cols_eff, n_samples,
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
            logger.warning(
                "cat-FE: sample weights ignored on GPU path; "
                "falling back to CPU weighted kernel."
            )
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
        return data, cols, nbins, state
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
        return data, cols, nbins, state

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
            return data, cols, nbins, state

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
        return data, cols, nbins, state

    # ---- Permutation confirmation + FWER correction ----
    # Runs only when ``cfg.full_npermutations > 0`` (default 100). Tests joint-independence null; failed pairs are dropped from ``selected_idx``. The resulting
    # ``confidence_dict`` is surfaced via diagnostics for user inspection. ``n_search_pairs`` is the family size for FWER correction -- the count of pairs CONSIDERED
    # in the search phase, NOT the top-K count.
    # Full Westfall-Young requires the per-shuffle max-II across ALL search pairs, materially more expensive than per-survivor permutation. To get the full WY
    # behaviour, we substitute the per-pair p-values from the joint-independence test with the WY-corrected versions BEFORE the orchestrator applies the floor.
    # Memory budget: full WY pre-merges m * n int32 cells; if that exceeds e.g. 500 MB we fall back to Bonferroni-on-survivors via the _apply_fwer_correction path.
    use_full_wy = (
        cfg.fwer_correction == "westfall_young"
        and cfg.full_npermutations > 0
        and len(pairs_a) * n_samples * 4 < 500 * 1024 * 1024
    )

    # Bandit UCB1 budget allocation overrides the fixed path when cfg.perm_budget_strategy='bandit_ucb1' (and full_npermutations>0 AND not using full WY which has its own coordination).
    if (
        getattr(cfg, "perm_budget_strategy", "fixed") == "bandit_ucb1"
        and cfg.full_npermutations > 0
        and not use_full_wy
    ):
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
        kept_mask = np.array([
            confidence_dict[(int(pairs_a[k]), int(pairs_b[k]))] >= min_conf
            for k in selected_idx
        ])
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
        return data, cols, nbins, state

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
            return data, cols, nbins, state

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
                    "cat-FE: top-K seeds produced 0 k-way results; "
                    "falling back to all %d pairs (quadratic cost)",
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
                "joint_dependence_confidence": (
                    float(confidence_dict[(i, j)])
                    if (i, j) in confidence_dict
                    else None
                ),
                # Bootstrap CI on II. ``None`` when disabled.
                "bootstrap_ii_ci": (
                    bootstrap_ci_dict[(i, j)]
                    if (i, j) in bootstrap_ci_dict
                    else None
                ),
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
            te_block = np.column_stack([
                v.astype(np.float64, copy=False) for v in te_cols_list
            ])
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

    # ---- Single concat onto data / cols / nbins ----
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
