"""UCB1 bandit-driven pair-confirmation for ``cat_interactions``.

Split out of ``cat_interactions.py`` to keep the parent below the 1k-line
monolith threshold. Behaviour preserved bit-for-bit; the parent re-exports
``_confirm_pairs_bandit_ucb1`` so the orchestrator in
``run_cat_interaction_step`` continues to call it via the same name.
"""
from __future__ import annotations

import logging
import math

import numpy as np
from numba import njit, prange

from .cat_fe_state import CatFEConfig
from .info_theory import merge_vars
# The bandit per-arm pull pipes through the same shuffle + FWER helpers as
# the bulk permutation confirm path; importing them eagerly here is safe
# (no cycle: bandit -> permutation, permutation does not import bandit).
from ._cat_confirm_permutation import (
    _shuffle_and_compute_three_mis,
    _bulk_shuffle_and_compute_three_mis,
    _apply_fwer_correction,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Bandit UCB1 permutation budget allocation
#
# Sequential allocation that gives more shuffles to pairs whose p-value CI straddles the rejection threshold. Pairs clearly above/below the threshold get cut off
# early (Besag-Clifford style); ambiguous pairs receive extra. Saves 2-5x total perms vs fixed allocation in typical workloads.
# References: Auer 2002 (UCB1), Besag & Clifford 1991 (sequential perm tests).
# ============================================================================


def _confirm_pairs_bandit_ucb1(
    factors_data: np.ndarray,
    pairs_a: np.ndarray,
    pairs_b: np.ndarray,
    selected_idx: np.ndarray,
    ii_arr: np.ndarray,
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    cfg: CatFEConfig,
    n_search_pairs: int,
    dtype,
    verbose: int,
) -> tuple:
    """Adaptive permutation budget via UCB1-style allocation.

    Total budget = ``cfg.full_npermutations * len(selected_idx)``. Each pair starts with ``min_perms = max(10, full_npermutations // 4)`` shuffles, then we allocate
    remaining shuffles to the pair with the widest binomial CI on its current ``p_estimate`` (highest uncertainty about rejection).

    Early-stops a pair when its 95% CI lies entirely above/below the rejection threshold (1 - min_nonzero_confidence = 0.05).
    Returns ``(selected_idx_kept, confidence_dict)``.
    """
    n_perms_total = cfg.full_npermutations
    if n_perms_total <= 0 or len(selected_idx) == 0:
        return selected_idx, {}

    min_perms = max(10, n_perms_total // 4)
    min_conf = 0.95
    alpha = 1.0 - min_conf

    # Pre-merge per-pair classes_pair / X1 / X2 (same as the fixed path).
    nfailed = np.zeros(len(selected_idx), dtype=np.int64)
    nshuf = np.zeros(len(selected_idx), dtype=np.int64)
    active_mask = np.ones(len(selected_idx), dtype=bool)
    pair_cache: list = []  # cached merge results per survivor

    # The single-variable merge of feature ``idx`` depends only on
    # ``factors_data[:, idx]`` + ``nbins[idx]`` (default current_nclasses=1,
    # fresh final_classes), so it is identical for every pair that contains
    # ``idx``. Features recur across the selected pairs, so memoising the
    # single-var merge by feature index removes the redundant recomputation
    # (previously 2 single-var merges per pair -> one per UNIQUE feature). The
    # cached (classes, freqs) arrays are read-only downstream (the shuffle
    # mutates only a copy of classes_y, never classes_x1/x2), so sharing one
    # array across pairs is safe. Bit-identical: same merge_vars output, memoised.
    _single_merge_cache: dict = {}

    def _single_merge(idx: int):
        cached = _single_merge_cache.get(idx)
        if cached is None:
            cls, fq, _ = merge_vars(
                factors_data=factors_data,
                vars_indices=np.array([idx], dtype=np.int64),
                var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
            )
            cached = (cls, fq)
            _single_merge_cache[idx] = cached
        return cached

    for k in selected_idx:
        i = int(pairs_a[k])
        jj = int(pairs_b[k])
        cls_pair, fq_pair, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=np.array([i, jj], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        cls_x1, fq_x1 = _single_merge(i)
        cls_x2, fq_x2 = _single_merge(jj)
        pair_cache.append((cls_pair, fq_pair, cls_x1, fq_x1, cls_x2, fq_x2))

    def _step_pair(j: int, ii_obs: float):
        """One shuffle for pair j. Updates nfailed, nshuf in place."""
        cls_pair, fq_pair, cls_x1, fq_x1, cls_x2, fq_x2 = pair_cache[j]
        classes_y_safe = classes_y.copy()
        i_pair, i_x1, i_x2 = _shuffle_and_compute_three_mis(
            cls_pair, fq_pair, cls_x1, fq_x1, cls_x2, fq_x2,
            classes_y_safe, freqs_y, dtype,
        )
        if (i_pair - i_x1 - i_x2) >= ii_obs:
            nfailed[j] += 1
        nshuf[j] += 1

    # Phase 1: initial allocation to every survivor. Bulk-shuffle ``min_perms``
    # permutations per pair via ``_bulk_shuffle_and_compute_three_mis`` (prange
    # over independent shuffles, each with its own LCG state + local Y buffer);
    # ~6x faster on 8 cores vs the serial _step_pair loop on n=200k, K=10/5/5/3
    # (bench 2026-05-20). Phase 2 still uses _step_pair because UCB1 needs the
    # latest result to pick the next allocation.
    _phase1_base_seed = np.uint64(0xC0FFEE)
    for j, k in enumerate(selected_idx):
        ii_obs = float(ii_arr[k])
        cls_pair, fq_pair, cls_x1, fq_x1, cls_x2, fq_x2 = pair_cache[j]
        # Per-pair base seed so different pairs see different permutation
        # sequences (deterministic given the fixed _phase1_base_seed).
        _base_seed = _phase1_base_seed + np.uint64(j) * np.uint64(0x9E3779B1)
        ip_arr, ix1_arr, ix2_arr = _bulk_shuffle_and_compute_three_mis(
            cls_pair, fq_pair, cls_x1, fq_x1, cls_x2, fq_x2,
            classes_y, freqs_y, min_perms, _base_seed, dtype,
        )
        for _p in range(min_perms):
            if (ip_arr[_p] - ix1_arr[_p] - ix2_arr[_p]) >= ii_obs:
                nfailed[j] += 1
            nshuf[j] += 1

    # Phase 2: adaptive allocation. Budget remaining = total - phase1
    total_budget = max(n_perms_total * len(selected_idx), min_perms * len(selected_idx))
    used = min_perms * len(selected_idx)

    if verbose:
        logger.info(
            "cat-FE bandit UCB1: phase 1 used %d perms; %d remaining budget",
            used, total_budget - used,
        )

    # Phase 2 burst size: take K shuffles for the UCB1-selected pair via the
    # parallel-prange bulk kernel (iter11) rather than 1 shuffle per loop. The
    # MOST AMBIGUOUS pair stays the most ambiguous until its CI narrows, so
    # committing K consecutive shuffles to it before re-checking UCB1 costs
    # almost no fidelity and gives ~Kx speedup on the bulk vs serial path
    # (iter11 bench: 6.03x on n_perms=8, n=200k, 8-core box). The burst is
    # capped to remaining budget so we never overshoot total_budget.
    _PHASE2_BURST = 8
    _burst_seed_base = np.uint64(0xDEADBEEF)
    _burst_counter = 0

    while used < total_budget:
        # Decide which pair to allocate next shuffle to. UCB1-style: pair with widest CI on current p_estimate, AMONG ACTIVES.
        best_j = -1
        best_score = -np.inf
        for j in range(len(selected_idx)):
            if not active_mask[j]:
                continue
            p_est = nfailed[j] / max(nshuf[j], 1)
            # Binomial CI half-width ~1.96 * sqrt(p*(1-p)/n); approx with min(p, 1-p) for the variance term. UCB bonus = sqrt(2*ln(used)/n_j).
            ucb_bonus = math.sqrt(2.0 * math.log(max(used, 2)) / max(nshuf[j], 1))
            score = ucb_bonus + p_est * (1.0 - p_est)  # prefer ambiguous
            if score > best_score:
                best_score = score
                best_j = j
        if best_j < 0:
            break  # all converged

        k = selected_idx[best_j]
        ii_obs = float(ii_arr[k])
        # Burst: how many shuffles to allocate to best_j in one bulk call.
        # Bounded by remaining budget AND the configured burst cap.
        _burst = min(_PHASE2_BURST, total_budget - used)
        if _burst <= 1:
            # Last-shuffle case: fall through to the cheap sequential path
            # (the bulk kernel's njit-dispatch overhead isn't worth it for n=1).
            _step_pair(best_j, ii_obs)
            used += 1
        else:
            cls_pair, fq_pair, cls_x1, fq_x1, cls_x2, fq_x2 = pair_cache[best_j]
            _burst_seed = _burst_seed_base + np.uint64(_burst_counter) * np.uint64(0x9E3779B1)
            _burst_counter += 1
            ip_arr, ix1_arr, ix2_arr = _bulk_shuffle_and_compute_three_mis(
                cls_pair, fq_pair, cls_x1, fq_x1, cls_x2, fq_x2,
                classes_y, freqs_y, _burst, _burst_seed, dtype,
            )
            for _p in range(_burst):
                if (ip_arr[_p] - ix1_arr[_p] - ix2_arr[_p]) >= ii_obs:
                    nfailed[best_j] += 1
                nshuf[best_j] += 1
            used += _burst

        # Early-stop check: 95% Clopper-Pearson-ish bound on p. Conservative bound p +/- z * sqrt(p*(1-p)/n), z=1.96.
        n_j = nshuf[best_j]
        p_j = nfailed[best_j] / n_j
        margin = 1.96 * math.sqrt(p_j * (1 - p_j) / n_j + 1e-9)
        upper = p_j + margin
        lower = p_j - margin
        # If lower > alpha: pair confidently rejected. If upper < alpha: pair confidently accepted. Either way, no more shuffles needed.
        if lower > alpha + 0.02 or upper < alpha - 0.02:
            active_mask[best_j] = False

    # Build confidence dict
    confidence_dict: dict = {}
    for j, k in enumerate(selected_idx):
        i = int(pairs_a[k])
        jj = int(pairs_b[k])
        p = (nfailed[j] + 1) / (nshuf[j] + 1)  # continuity-corrected
        confidence_dict[(i, jj)] = 1.0 - p

    # FWER correction (reuse same path)
    corrected_conf = _apply_fwer_correction(
        confidence_dict, cfg, n_search_pairs=n_search_pairs,
    )
    kept_mask = np.array([
        corrected_conf[(int(pairs_a[k]), int(pairs_b[k]))] >= min_conf
        for k in selected_idx
    ])
    if verbose:
        for j, k in enumerate(selected_idx):
            ij = (int(pairs_a[k]), int(pairs_b[k]))
            logger.info(
                "cat-FE bandit: pair %s nshuf=%d conf=%.3f corrected=%.3f%s",
                ij, nshuf[j], confidence_dict[ij], corrected_conf[ij],
                "" if kept_mask[j] else " [REJECTED]",
            )
    return selected_idx[kept_mask], corrected_conf
