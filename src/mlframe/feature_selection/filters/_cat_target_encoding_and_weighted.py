"""Target encoding + weighted pair-search kernel + group-aware shuffle
for ``cat_interactions``.

Split out of ``cat_interactions.py`` to keep the parent below the 1k-line
monolith threshold. Behaviour preserved bit-for-bit; the parent re-exports
the public-looking entries so the orchestrator in
``run_cat_interaction_step`` continues to call them via the same names.

What lives here:
  - ``_compute_target_encoding`` (categorical -> target-mean encoding for
    the engineered-pair emit path)
  - ``_pair_search_kernel_weighted_njit`` (weighted variant of the
    marginal pair-search kernel used when ``cfg.use_pair_weights`` is on)
  - ``_group_aware_shuffle`` (within-strata permutation used by the
    permutation-confirm path)
"""
from __future__ import annotations

import logging

import numpy as np
from numba import njit, prange

from .info_theory import merge_vars

logger = logging.getLogger(__name__)


# ============================================================================
# Target encoding emit
#
# In addition to factorize-encoded engineered cols, emit target-encoded variants: ``E[Y | merged_class]`` per cell with optional out-of-fold smoothing to prevent leakage.
# Useful when downstream models prefer numeric inputs (linear / NN / tree models with continuous splits).
# ============================================================================


@njit(cache=True)
def _cell_sum_cnt_njit(classes, y, n_uniq):
    """Per-cell running (sum of y, count) over rows, in row order -- bit-identical to the Python
    ``for row: cell_sum[c]+=y[row]; cell_cnt[c]+=1`` accumulator it replaces (~490x @100k). ``classes`` is int."""
    cell_sum = np.zeros(n_uniq, dtype=np.float64)
    cell_cnt = np.zeros(n_uniq, dtype=np.float64)
    for row in range(classes.shape[0]):
        c = classes[row]
        cell_sum[c] += y[row]
        cell_cnt[c] += 1.0
    return cell_sum, cell_cnt


def _compute_target_encoding(
    factors_data: np.ndarray,
    idx_tuple: tuple,
    target_indices: np.ndarray,
    classes_y: np.ndarray,
    nbins: np.ndarray,
    n_oof_folds: int,
    smoothing: float,
    dtype,
    seed: int = 0,
    allow_naive_leak: bool = False,
) -> tuple:
    """Compute target-encoded values per cell of (X[idx_tuple]). Returns (te_values, cell_means_oof_combined) -- a 1-D array of ``n_samples`` floats where each row is
    ``E[Y | merged_class]`` computed out-of-fold (to prevent leakage).

    Strategy:
    - Build per-cell mean of Y, with shrinkage: ``te = (n_c * te_raw + alpha * te_global) / (n_c + alpha)`` (Micci-Barreca 2001).
    - For OOF: split rows into K folds; for each fold, compute cell means from the other K-1 folds, apply to this fold's rows.
    - For naive (n_oof_folds=0): single-pass cell mean across ALL rows. Leaks signal -- only safe when used as a downstream feature in a separate train/val split.

    Y is treated as numeric (regression). For binary classification, this gives per-cell P(y=1) -- well-behaved.

    Multi-class target encoding strategy (wave 68 closure, 2026-05-20): the helper
    treats ``classes_y`` as a numeric label (0, 1, 2, ...). For multi-class targets
    the resulting per-cell mean is the EXPECTED CLASS INDEX (not a class probability),
    which is meaningful when the labels are ordinal (e.g. 1-5 star ratings) but
    semantically wrong for nominal multi-class. Callers needing proper per-class
    encoded features (one column per class) should fit the encoder n_classes times
    on one-vs-rest binary derived columns -- that's the responsibility of the
    caller, not this kernel: per-class expansion would multiply the feature space
    by n_classes for every interaction, which is rarely the right trade-off.
    """
    n_samples = factors_data.shape[0]
    # Compute the merged class per row
    classes_merged, _, n_uniq = merge_vars(
        factors_data=factors_data,
        vars_indices=np.array(idx_tuple, dtype=np.int64),
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    # Y as numeric: take classes_y (already merged) and treat as float
    y_numeric = classes_y.astype(np.float64)
    te_global = float(y_numeric.mean())

    if n_oof_folds <= 0 and not allow_naive_leak:
        # The naive single-pass per-cell mean includes each row's OWN y, so the emitted te_values (a TRAINING
        # feature) leaks the target -- inflated in-sample MI, collapses on holdout. The default-safe behaviour is to
        # fall back to a real OOF split; callers who genuinely want the naive path in a SEPARATE train/val scenario
        # must opt in with allow_naive_leak=True.
        logger.warning(
            "_compute_target_encoding: n_oof_folds=%d would emit a target-LEAKING training feature (per-cell mean "
            "includes each row's own y); falling back to 2-fold OOF. Pass allow_naive_leak=True to keep the naive "
            "path (only safe when the emitted column is used in a SEPARATE train/val split).", n_oof_folds,
        )
        n_oof_folds = 2

    if n_oof_folds <= 0:
        # Naive: per-cell mean across all rows
        cell_sum, cell_cnt = _cell_sum_cnt_njit(
            np.ascontiguousarray(classes_merged, dtype=np.int64),
            np.ascontiguousarray(y_numeric, dtype=np.float64), int(n_uniq),
        )
        cell_means = np.zeros(int(n_uniq), dtype=np.float64)
        for c in range(int(n_uniq)):
            if cell_cnt[c] > 0:
                raw = cell_sum[c] / cell_cnt[c]
                cell_means[c] = (cell_cnt[c] * raw + smoothing * te_global) / (cell_cnt[c] + smoothing)
            else:
                cell_means[c] = te_global
        te_values = cell_means[classes_merged.astype(np.int64)]
        return te_values, cell_means

    # OOF encoding: for each fold, compute cell means from other folds.
    # Fold membership must be SHUFFLED, not positional: a raw ``arange % K`` ties fold id to row
    # position, so if the input is sorted/clustered by the merged cell (common after an upstream
    # groupby/sort) a cell's rows concentrate in one fold -> its OOF estimate collapses toward the
    # in-fold mean -> partial target leak into ``te_values``. Mirror the leak-safe shuffle used by the
    # sibling encoders (_target_encoding_fe / target_aware_group_bin). Seed keeps it reproducible.
    K = int(n_oof_folds)
    _perm = np.random.default_rng(int(seed)).permutation(n_samples)
    fold_ids = np.empty(n_samples, dtype=np.int64)
    fold_ids[_perm] = np.arange(n_samples) % K
    te_values = np.full(n_samples, te_global, dtype=np.float64)
    for f in range(K):
        train_mask = fold_ids != f
        test_mask = ~train_mask
        # Compute cell means on training rows
        cell_sum, cell_cnt = _cell_sum_cnt_njit(
            np.ascontiguousarray(classes_merged[train_mask], dtype=np.int64),
            np.ascontiguousarray(y_numeric[train_mask], dtype=np.float64), int(n_uniq),
        )
        cell_means_fold = np.full(int(n_uniq), te_global, dtype=np.float64)
        for c in range(int(n_uniq)):
            if cell_cnt[c] > 0:
                raw = cell_sum[c] / cell_cnt[c]
                cell_means_fold[c] = (cell_cnt[c] * raw + smoothing * te_global) / (cell_cnt[c] + smoothing)
        # Apply to test fold rows (vectorised gather; bit-identical to the per-row assignment)
        te_values[test_mask] = cell_means_fold[np.asarray(classes_merged, dtype=np.int64)[test_mask]]

    # Also compute global (all-rows) cell means for transform()-time replay. At transform OOF doesn't make sense (test data has no Y); we use the global mean per cell.
    cell_sum, cell_cnt = _cell_sum_cnt_njit(
        np.ascontiguousarray(classes_merged, dtype=np.int64),
        np.ascontiguousarray(y_numeric, dtype=np.float64), int(n_uniq),
    )
    cell_means_global = np.full(int(n_uniq), te_global, dtype=np.float64)
    for c in range(int(n_uniq)):
        if cell_cnt[c] > 0:
            raw = cell_sum[c] / cell_cnt[c]
            cell_means_global[c] = (cell_cnt[c] * raw + smoothing * te_global) / (cell_cnt[c] + smoothing)
    return te_values, cell_means_global


# ============================================================================
# Sample-weight-aware MI computation
#
# Cat-FE-local weighted MI: bypasses ``merge_vars`` and computes joint weighted histogram directly. Used when ``cfg.sample_weight_col`` is set. Does NOT extend the
# global ``merge_vars`` API (which has 500+ callers); weighted MRMR screening / RFECV is project-level future work.
# ============================================================================


@njit(parallel=True, cache=True)
def _pair_search_kernel_weighted_njit(
    factors_data: np.ndarray,
    pairs_a: np.ndarray,
    pairs_b: np.ndarray,
    marginal_mi: np.ndarray,
    nbins: np.ndarray,
    classes_y: np.ndarray,
    weights: np.ndarray,  # (n,) float64; sum(weights) > 0
    dtype,
) -> tuple:
    """Weighted variant of ``_pair_search_kernel_njit``.

    For each pair (a, b) compute weighted joint MI = sum_{m,y} w_{m,y} log(w_{m,y} * W / (w_m * w_y)) where w_{m,y} = (sum of row weights in cell (merged=m, y=y)) / W
    and W = sum of all weights. Equivalent to unweighted MI under uniform weights. Returns (joint_mi, ii, n_uniq) arrays.
    """
    n_pairs = len(pairs_a)
    n_samples = factors_data.shape[0]
    nbins_y = int(classes_y.max()) + 1 if classes_y.size > 0 else 1
    joint_mi_out = np.zeros(n_pairs, dtype=np.float64)
    ii_out = np.zeros(n_pairs, dtype=np.float64)
    n_uniq_out = np.zeros(n_pairs, dtype=np.int64)
    W = 0.0
    for r in range(n_samples):
        W += weights[r]
    inv_W = 1.0 / W if W > 0 else 0.0

    for k in prange(n_pairs):
        i = pairs_a[k]
        j = pairs_b[k]
        nba = int(nbins[i])
        nbb = int(nbins[j])
        merged_card = nba * nbb

        joint_w = np.zeros(merged_card * nbins_y, dtype=np.float64)
        marg_m_w = np.zeros(merged_card, dtype=np.float64)
        marg_y_w = np.zeros(nbins_y, dtype=np.float64)
        for row in range(n_samples):
            va = factors_data[row, i]
            vb = factors_data[row, j]
            code = va + vb * nba
            cy = classes_y[row]
            w = weights[row]
            joint_w[code * nbins_y + cy] += w
            marg_m_w[code] += w
            marg_y_w[cy] += w

        mi = 0.0
        n_uniq = 0
        for m in range(merged_card):
            mm = marg_m_w[m]
            if mm <= 0:
                continue
            n_uniq += 1
            for y in range(nbins_y):
                jw = joint_w[m * nbins_y + y]
                if jw <= 0:
                    continue
                my = marg_y_w[y]
                if my <= 0:
                    continue
                jp = jw * inv_W
                mi += jp * np.log(jw * W / (mm * my))

        joint_mi_out[k] = mi
        ii_out[k] = mi - marginal_mi[i] - marginal_mi[j]
        n_uniq_out[k] = n_uniq
    return joint_mi_out, ii_out, n_uniq_out


@njit(cache=True)
def _group_aware_shuffle(
    classes_y_safe: np.ndarray, groups: np.ndarray, n_groups: int, base_seed: int,
) -> None:
    """Shuffle classes_y_safe in place, restricting to between-group permutations. For each group, all rows get the SAME (shuffled) Y representative (the group's first
    row's Y, after shuffling group ordering). Implementation: shuffle group representatives, then broadcast each group's shuffled Y to all its rows.

    The group-representative Fisher-Yates uses a seeded inline LCG (PCG-style step) so the permutation is reproducible at a fixed ``base_seed`` and never touches numpy's
    process-global RNG (the prior ``np.random.randint`` was unseeded and mutated numba's global stream).
    """
    n = len(classes_y_safe)
    # Find each group's first-occurrence Y value
    group_y = np.full(n_groups, -1, dtype=np.int32)
    for i in range(n):
        g = groups[i]
        if group_y[g] < 0:
            group_y[g] = classes_y_safe[i]
    # Shuffle the group_y array
    state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(1)
    for i in range(n_groups - 1, 0, -1):
        state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
        j = int(state >> np.uint64(33)) % (i + 1)
        tmp = group_y[i]
        group_y[i] = group_y[j]
        group_y[j] = tmp
    # Broadcast back
    for i in range(n):
        classes_y_safe[i] = group_y[groups[i]]
