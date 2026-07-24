"""FWER/multiple-testing correction for ``_cat_confirm_permutation``'s pair-confirmation phase.

Split out of ``_cat_confirm_permutation.py`` (X_EFFICIENCY_ARCHITECTURE-1 fix, mrmr_audit_2026-07-22)
to clear the repo's enforced hard 1000-LOC CI gate (``tests/test_meta/test_no_file_over_1k_loc.py``) --
that file was 1106 lines and absent from the gate's exempt list. Behaviour preserved bit-for-bit; the
parent re-exports both functions so ``_confirm_pairs_via_permutation`` continues to call them unchanged.

What lives here:
  - Full Westfall-Young max-II null distribution + corrected p-value (``_compute_westfall_young_corrected_p``)
  - Multiple-testing correction dispatch: none / bonferroni / bh_fdr / westfall_young-fallback
    (``_apply_fwer_correction``)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .cat_fe_state import CatFEConfig
from .info_theory import compute_mi_from_classes, compute_mi_from_classes_weighted, merge_vars

logger = logging.getLogger("mlframe.feature_selection.filters._cat_confirm_permutation")  # matches the pre-carve logger name; preserves log-filter/caplog compatibility for existing callers/tests

# Stable base seed for the Westfall-Young shuffle -- mirrors _cat_confirm_permutation._CAT_CONFIRM_BASE_SEED
# (kept as a separate module-level constant so this file has no import-time dependency on the parent).
_CAT_CONFIRM_BASE_SEED = 1_000_003


def _compute_westfall_young_corrected_p(
    factors_data: np.ndarray,
    pairs_a: np.ndarray,
    pairs_b: np.ndarray,
    ii_obs_arr: np.ndarray,
    selected_idx: np.ndarray,
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    marginal_mi: np.ndarray,
    n_perms: int,
    dtype,
    verbose: int,
    base_seed: int = _CAT_CONFIRM_BASE_SEED,
    weights: Optional[np.ndarray] = None,
) -> dict:
    """Full Westfall-Young: per shuffle, compute II_perm for ALL search-phase pairs, take the MAX, and accumulate the max-II distribution. Each survivor's p-value is
    ``(1 + #{b: max_II_perm[b] >= II_obs}) / (B + 1)``.

    ``weights`` (mrmr_audit_2026-07-20 B-19), when given, route every per-shuffle MI through the
    weighted kernel so the max-II null distribution matches the weighted search-phase ``ii_obs_arr``.

    The proper WY procedure (Westfall & Young 1993) naturally accounts for inter-pair correlation: pairs that share a column have correlated permutation distributions and
    the max-II statistic captures this. Strictly more powerful than Bonferroni on the same B.

    Cost: per shuffle, compute joint MI for all m = ``len(pairs_a)`` pairs. At m=4950 and B=100 that's 495k MI computations. Heavy -- enable only when
    ``cfg.fwer_correction='westfall_young'`` AND the user accepts the cost. Savings vs Bonferroni: typically need 2-5x fewer permutations for the same effective alpha.

    Returns ``{(i, j): corrected_p_value}`` ONLY for the survivors in
    ``selected_idx``.
    """
    n_samples = factors_data.shape[0]
    m = len(pairs_a)
    classes_y_safe = classes_y.copy()
    # Local RNG: the prior code used the unseeded global ``np.random.shuffle`` -- non-reproducible
    # AND it polluted the process-global RNG stream for every downstream caller (the cupy sibling
    # already uses a local stream). Seed from ``base_seed`` so re-runs are bit-stable.
    wy_rng = np.random.default_rng(int(base_seed))

    if verbose:
        logger.info(
            "cat-FE: full Westfall-Young permutation -- %d pairs x %d shuffles",
            m, n_perms,
        )

    # Pre-merge classes for all m search pairs ONCE. Memory: m * n * 4 B
    # = 19.8 MB at m=4950, n=1000; 198 MB at n=10000. Heavy but bounded.
    # If memory is a concern, fall back to Bonferroni-equivalent path.
    pair_classes_buf = np.empty((m, n_samples), dtype=dtype)
    pair_freqs_list: list = []
    for k in range(m):
        i = int(pairs_a[k])
        jj = int(pairs_b[k])
        cls_pair, fq_pair, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=np.array([i, jj], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        pair_classes_buf[k] = cls_pair
        pair_freqs_list.append(fq_pair)

    # Pre-merge marginal classes for each unique column in pairs_a/pairs_b
    unique_cols = np.unique(np.concatenate([pairs_a, pairs_b]))
    marginal_classes: dict = {}
    marginal_freqs: dict = {}
    for c in unique_cols:
        ci = int(c)
        cls_c, fq_c, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=np.array([ci], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        marginal_classes[ci] = cls_c
        marginal_freqs[ci] = fq_c

    # Permutation loop: for each shuffle, compute max II across all m pairs
    max_ii_per_perm = np.zeros(n_perms, dtype=np.float64)
    for b in range(n_perms):
        wy_rng.shuffle(classes_y_safe)
        # Compute MI(merged; Y_shuffled) for all pairs, and marginals for
        # all touched columns. Then II = joint - marginal_i - marginal_j.
        marginal_perm: dict = {}
        for ci, cls_c in marginal_classes.items():
            if weights is not None:
                marginal_perm[ci] = compute_mi_from_classes_weighted(cls_c, classes_y_safe, weights, dtype)
            else:
                marginal_perm[ci] = compute_mi_from_classes(
                    classes_x=cls_c, freqs_x=marginal_freqs[ci],
                    classes_y=classes_y_safe, freqs_y=freqs_y, dtype=dtype,
                )
        max_ii = -np.inf
        for k in range(m):
            if weights is not None:
                joint_perm = compute_mi_from_classes_weighted(pair_classes_buf[k], classes_y_safe, weights, dtype)
            else:
                joint_perm = compute_mi_from_classes(
                    classes_x=pair_classes_buf[k], freqs_x=pair_freqs_list[k],
                    classes_y=classes_y_safe, freqs_y=freqs_y, dtype=dtype,
                )
            ii_perm = joint_perm - marginal_perm[int(pairs_a[k])] - marginal_perm[int(pairs_b[k])]
            if ii_perm > max_ii:
                max_ii = ii_perm
        max_ii_per_perm[b] = max_ii

    # Compute corrected p for each survivor
    corrected_p: dict = {}
    for k in selected_idx:
        i = int(pairs_a[k])
        jj = int(pairs_b[k])
        ii_obs = float(ii_obs_arr[k])
        n_exceed = int((max_ii_per_perm >= ii_obs).sum())
        corrected_p[(i, jj)] = (n_exceed + 1) / (n_perms + 1)
    return corrected_p


def _apply_fwer_correction(
    confidence_dict: dict, cfg: CatFEConfig, n_search_pairs: int,
) -> dict:
    """Apply multiple-testing correction to per-pair p-values, returning the corrected confidence dict. Supports:

    - ``"none"``: identity. FWER unchecked; user accepts inflation.
    - ``"bonferroni"``: ``p_corr = min(1, p * m)`` where m is the effective search-family size (``n_search_pairs``, NOT len of survivors). Conservative.
    - ``"bh_fdr"``: Benjamini-Hochberg step-up FDR. Less conservative than Bonferroni, controls expected proportion of false discoveries.
    - ``"westfall_young"``: proper WY requires recomputing the max-II across ALL search-phase pairs under each shuffle. This branch is reached only as a FALLBACK
      (typically when memory is too tight for ``_compute_westfall_young_corrected_p``); it approximates with Bonferroni-on-survivors, which is conservative-equivalent
      for the typical case where survivors' II values dominate the per-shuffle max.

    ``n_search_pairs`` is the count of pairs CONSIDERED at search time, NOT len(survivors). A user who screened 100 candidate cols saw N(N-1)/2 = 4950 pairs; that's the
    family size, not 64 survivors.
    """
    if cfg.fwer_correction == "none" or not confidence_dict:
        return dict(confidence_dict)

    # Convert confidences to p-values
    p_vals = {k: 1.0 - v for k, v in confidence_dict.items()}
    m = max(n_search_pairs, len(p_vals))

    if cfg.fwer_correction == "bonferroni":
        return {k: 1.0 - min(1.0, p * m) for k, p in p_vals.items()}

    if cfg.fwer_correction == "bh_fdr":
        # Benjamini-Hochberg step-up
        sorted_items = sorted(p_vals.items(), key=lambda kv: kv[1])
        n = len(sorted_items)
        # Adjusted p_(i) = min over j>=i of (p_(j) * m / j)
        corrected = {}
        prev = 1.0
        for rank, (k, p) in enumerate(reversed(sorted_items), start=1):
            i = n - rank + 1  # 1-indexed BH position
            adj = min(prev, p * m / i)
            prev = adj
            corrected[k] = adj
        return {k: 1.0 - corrected[k] for k in p_vals}

    if cfg.fwer_correction == "westfall_young":
        # Fallback path: orchestrator normally runs ``_compute_westfall_young_corrected_p`` directly. Bonferroni-on-survivors is the conservative-equivalent fallback.
        return {k: 1.0 - min(1.0, p * m) for k, p in p_vals.items()}

    raise ValueError(f"Unknown fwer_correction: {cfg.fwer_correction!r}")
