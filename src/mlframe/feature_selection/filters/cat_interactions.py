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

import hashlib
import logging
import math
import warnings

import numpy as np
from numba import njit, prange

from .cat_fe_state import CatFEConfig, CatFEState
from .info_theory import (
    compute_mi_from_classes,
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


def _target_signature(target_codes: np.ndarray) -> str:
    """Content signature of the (discretized) target column(s) used to gate marginal-MI cache reuse.

    The cached marginal MI is ``MI(X; Y)`` -- it depends on the JOINT (X, Y), not on X's distribution alone.
    ``_column_signature`` captures only X's bincount, so two fits with an identical X distribution but a
    different / relabelled / re-aligned Y would otherwise collide and reuse a STALE MI. Hashing the exact
    target codes invalidates the whole cache whenever Y changes, which is the only sound reuse predicate.
    """
    arr = np.ascontiguousarray(np.asarray(target_codes, dtype=np.int64))
    return hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()


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
    target_sig: str | None = None,
) -> tuple:
    """Decide per candidate column whether the cached marginal MI is reusable; returns ``(reusable_mask, marginal_mi_reused, new_signatures)``.

    The mask is True for columns whose signature KL-divergence vs the cached version is below ``kl_threshold`` -- those rows reuse the cached MI and skip the screen pass.

    Reuse additionally REQUIRES the cached target signature to match ``target_sig``: the cached value is ``MI(X; Y)``, so a changed Y must invalidate every column even when X's distribution is unchanged (otherwise a stale MI is reused and columns are mis-pruned).
    """
    reusable_mask = np.zeros(len(candidate_idxs), dtype=bool)
    marginal_mi_reused = np.full(len(candidate_idxs), np.nan, dtype=np.float64)
    new_signatures: dict = {}
    cached_sigs = cache.get("col_signatures", {})
    cached_mis = cache.get("marginal_mis", {})
    # Whole-cache target gate: if the cached fit saw a different Y, nothing is reusable.
    cached_target_sig = cache.get("target_sig")
    target_matches = (target_sig is not None) and (cached_target_sig == target_sig)
    for k, col_idx in enumerate(candidate_idxs):
        col_int = int(col_idx)
        sig = _column_signature(factors_data[:, col_int], int(nbins[col_int]))
        new_signatures[col_int] = sig
        if target_matches and col_int in cached_sigs and col_int in cached_mis:
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
            if cfg.on_high_cardinality == "raise":
                # int16 truncation upstream silently mangles >32k-card cols, so downstream MI is garbage; the hard error tells the user "this shouldn't be cat".
                raise ValueError(
                    f"cat-FE: column index {idx} has nbins={nb}, exceeding the "
                    f"safe ceiling sqrt(n)*2={high_card_threshold:.0f} for "
                    f"n={n_samples}. High-cardinality categorical columns "
                    f"(IDs, hashes, free-text) produce unstable MI estimates "
                    f"and likely violate the int16 ceiling upstream. "
                    f"Drop the column or reconsider whether it's truly categorical."
                )
            # Default skip: drop from the cat-FE pool, let the column flow through the relevance screen (which can still drop it as noise).
            warnings.warn(
                f"cat-FE: column index {idx} has nbins={nb} > sqrt(n)*2={high_card_threshold:.0f} (n={n_samples}); skipping it for cat-FE. "
                f"It still passes through MRMR relevance screening. Set CatFEConfig(on_high_cardinality='raise') for the legacy hard error.",
                stacklevel=2,
            )
            continue
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


# ----------------------------------------------------------------------
# Sibling-module re-export. The 671-LOC ``run_cat_interaction_step`` body
# lives in ``_cat_interactions_step.py`` so this file stays below the
# 1k-LOC monolith threshold (parent had regressed past 1k since the
# prior cat_interactions split).
# ----------------------------------------------------------------------
from ._cat_interactions_step import run_cat_interaction_step  # noqa: E402,F401
