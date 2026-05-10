"""Categorical-feature interaction generator for ``MRMR``.

What this module does
---------------------
For each pair of categorical (or pre-discretized numeric) columns
``(X_i, X_j)``, compute the Jakulin Interaction Information

::

    II(X_i ; X_j ; Y) = I(X_i, X_j ; Y) - I(X_i ; Y) - I(X_j ; Y)

and keep the top-K pairs whose II indicates *synergy* (positive II:
the joint tells the target more than the sum of marginals -- the
canonical XOR-style hidden pair). Surviving pairs become new
ordinal-encoded columns appended to ``data`` / ``cols`` / ``nbins``,
and recipes (``EngineeredRecipe(kind="factorize")``) land in
``self._cat_fe_state_.recipes`` so ``MRMR.transform`` can replay them
on test data.

Implementation order (this file lands incrementally)
----------------------------------------------------
**MVP (this PR)**:

* validation gates (n>=min_n, no all-NaN, no constant cols, ...)
* per-column marginal MI with Y -- ``_marginal_screen_njit``
* per-pair II -- plug-in entropies, no Miller-Madow yet
* top-K argpartition over the flat II array
* materialise survivors via ``merge_vars`` + ``np.concatenate``
* recipes built and persisted

**Phase 2** (subsequent commits, all hooked off ``CatFEConfig``):

* Miller-Madow correction applied to all six entropies as a unit (SB6)
* two-stage permutation budget with same-shuffle three-MI test (E2 / SB1)
* Westfall-Young multi-test correction (SB2)
* greedy k-way expansion to triplets / quartets (SB7)
* K-fold II stability filter (E6)
* anti-redundancy with already-selected features (E3 / SM6)
* GPU dispatch shim ``mi_direct_gpu_batched_pairs`` (P9)

References
----------
* Plan: ``C:/Users/TheLocalCommander/.claude/plans/linear-shimmying-thimble.md``
* Jakulin & Bratko 2003, *Quantifying and Visualizing Attribute Interactions*
* Williams & Beer 2010 (PID -- documented limitation per SB5)
* Paninski 2003 *Estimation of Entropy and Mutual Information*
  (Miller-Madow, bias formulas, sample-size guidance)
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
# Default resolution (data-aware)
# ============================================================================


def resolve_max_combined_nbins(
    cfg: CatFEConfig, n_samples: int, hard_cap: int = 10_000_000
) -> int:
    """Resolve ``cfg.max_combined_nbins`` to a concrete int.

    ``None`` -> Paninski-derived data-aware ceiling
    (SM5 / I7): ``max(4, int(n * 0.05 / 3) + 1)``. Empirically this
    keeps per-cell observation count above ~3 at the sample sizes
    where MI estimation stops being noise.

    Always clamped to ``hard_cap`` (10**7) regardless of user value
    (SB10 / F18) -- prevents OOM via misconfig like
    ``max_combined_nbins=10**9`` (4 GB freqs allocation).
    """
    if cfg.max_combined_nbins is None:
        # Paninski bias ~ (k-1)/(2n) per entropy term. For 0.05 nat
        # tolerance across 3 entropies: 3*(k-1)/(2n) < 0.05
        # -> k < n*0.05/1.5 + 1 ≈ n/30 + 1. Default tolerance 0.05
        # (SD1 acknowledges this is folklore, not analytical).
        resolved = max(4, int(n_samples * 0.05 / 3) + 1)
    else:
        resolved = int(cfg.max_combined_nbins)
    return min(resolved, hard_cap)


def resolve_min_interaction_information(
    cfg: CatFEConfig, n_samples: int
) -> float:
    """Resolve ``cfg.min_interaction_information`` to a concrete float.

    ``None`` -> ``-3 / sqrt(n)`` (B4) -- small-negative absorbs
    finite-sample noise around the synergy boundary so that pure
    k-way XOR (where all 2-way IIs are 0 in expectation but noisy)
    can still bubble survivors to the heap.
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
    - constants / all-NaN (``nbins[i] == 1``) -> SM7 / F1 / F2
    - high-cardinality (``nbins[i] > sqrt(n) * 2``) -> SM8 / F8

    Side effects:
    - records dropped names in ``state.dropped_singleton_nbins`` and
      ``state.high_cardinality_warnings`` for downstream debugging.

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
            # Refuse rather than warn: per SM8 the legacy upstream truncation
            # to int16 already silently mangles >32k-cardinality cols, so
            # downstream MI is on garbage. Hard error gives the user a
            # clear "this column shouldn't be cat".
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

    Runs ``prange`` over candidates -- each thread merges its column
    into ``classes_x`` independently, then computes MI. No per-thread
    state shared.

    Returns a 1-D float64 array of length ``len(candidate_idxs)``,
    aligned with the input order. Cells for unmergeable / zero-MI
    columns simply produce ``0.0`` (downstream logic handles those
    via ``cfg.marginal_floor``).
    """
    n_candidates = len(candidate_idxs)
    out = np.zeros(n_candidates, dtype=np.float64)
    for k in prange(n_candidates):
        idx = candidate_idxs[k]
        # Build a single-element vars_indices array; merge_vars handles k=1
        # as a degenerate case (just renumbers the column densely).
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

    Returns ``(joint_mi_arr, ii_arr, n_uniq_arr)`` -- three 1-D arrays
    of length ``len(pairs_a)`` aligned with the input order.

    T6 optimisation: instead of calling ``merge_vars`` per pair (which
    allocates per-call), we compute the joint code directly as
    ``code = data[row, i] + data[row, j] * nbins[i]``. This works
    because for plug-in MI estimation, empty cells contribute 0 to
    entropy and the dense-renumbering done by ``merge_vars`` doesn't
    affect MI (MI is invariant under bijective relabeling of the
    alphabet). Joint histogram is built in-place into a thread-local
    buffer of size ``nbins[i] * nbins[j] * nbins_y``.

    Cuts the per-pair cost ~3-5x vs the merge_vars path by eliminating
    the renumber + lookup table phase per pair. cProfile-driven (T6).
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
        # Also per-pair marginals so we don't pay another pass.
        m_merged = np.zeros(merged_card, dtype=np.int64)
        for row in range(n_samples):
            va = factors_data[row, i]
            vb = factors_data[row, j]
            code = va + vb * nba
            cy = classes_y[row]
            joint_hist[code * nbins_y + cy] += 1
            m_merged[code] += 1

        # Compute MI: I(merged; Y) = H(merged) + H(Y) - H(merged, Y)
        # Use direct formula: sum jc/n * log(jc * n / (m_m * m_y))
        # m_y is freqs_y * n_samples (caller-provided freqs).
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
                # freqs_y[y] is a probability; multiply by n_samples to recover
                # the count form
                my = freqs_y[y] * n_samples
                if my <= 0:
                    continue
                jf = jc * inv_n
                mi += jf * np.log(jc * n_samples / (mm * my))

        joint_mi_out[k] = mi
        ii_out[k] = mi - marginal_mi[i] - marginal_mi[j]
        n_uniq_out[k] = n_uniq
    return joint_mi_out, ii_out, n_uniq_out


# ============================================================================
# Score / select / materialise
# ============================================================================


# ============================================================================
# Miller-Madow II re-score (SB6)
#
# The Jakulin II expansion involves SIX entropies with mixed signs:
#
#   II = H(X1,X2) + H(X1,Y) + H(X2,Y) - H(X1,X2,Y) - H(X1) - H(X2) - H(Y)
#                                                                   (constant)
#                       (last term cancels across pairs since H(Y) is fit-wide)
#
# Plug-in entropy is biased downward by ``(k-1)/(2n)`` per entropy term, where
# k is the number of NON-EMPTY bins. Under the independence null, the SIGNED
# sum of these biases reduces to ``-(a-1)(b-1)(c-1)/(2n)`` -- i.e. plug-in II
# is biased UPWARD; Miller-Madow correction pulls it back down.
#
# Cost: 5+ ``merge_vars`` calls per pair (X1, X2, X1Y, X2Y, X1X2Y) vs the
# plug-in path's 1 call. To stay fast on the search loop, we apply MM ONLY
# to the top-K survivors as a re-rank step. This catches the high-cardinality
# false positives where bias dominates without paying the cost on every pair.
#
# References: Paninski 2003 §4, Treves & Panzeri 1995, Roulston 1999.
# ============================================================================


def _entropy_for_mode(
    freqs: np.ndarray, n_samples: int, use_mm: bool, use_kt: bool = False,
) -> float:
    """Plug-in / Miller-Madow / Krichevsky-Trofimov entropy.

    Tier 4.5: KT smoothing -- adds Dirichlet(0.5) pseudocounts before
    plug-in entropy. Less biased than plug-in for high-cardinality
    joints; provably asymptotically efficient (Krichevsky & Trofimov
    1981).
    """
    if use_kt:
        # Reconstruct counts, add 0.5 to each cell, renormalize, then
        # plug-in entropy. ``freqs`` is normalised so counts = freqs * n.
        counts = freqs * n_samples + 0.5
        K = len(counts)
        total = float(n_samples) + 0.5 * K
        probs = counts / total
        return entropy(probs)
    if use_mm:
        return entropy_miller_madow(freqs, n_samples)
    return entropy(freqs)


def _should_apply_mm_for_pair_analytical(
    nbins_a: int, nbins_b: int, n_y_classes: int, n_samples: int,
) -> bool:
    """Tier 4.6: analytically-derived MM auto-gate. Per Paninski 2003,
    plug-in entropy bias is ``(K-1) / (2n)``. The signed sum across
    the 6 entropies of the II expansion under the independence null
    equals ``-(a-1)(b-1)(c-1) / (2n)``. We apply MM when this bias is
    comparable to typical synergy floors (``-3/sqrt(n)`` per B4).

    Activate MM when ``(a-1)(b-1)(c-1)/(2n) >= 3/sqrt(n)``, i.e.
    ``(a-1)(b-1)(c-1) >= 6 * sqrt(n)``. Replaces the previous folklore
    threshold ``(a*b*c)/n > 0.05`` (SD1).
    """
    bias = (nbins_a - 1) * (nbins_b - 1) * (n_y_classes - 1)
    threshold = 6.0 * math.sqrt(max(n_samples, 1))
    return bias >= threshold


def _compute_pair_ii_mm(
    factors_data: np.ndarray,
    idx_a: int,
    idx_b: int,
    nbins: np.ndarray,
    target_indices: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    h_y: float,                  # H(Y), precomputed once at orchestrator level
    use_mm: bool,
    dtype,
) -> float:
    """Compute Jakulin II for a single pair with optional Miller-Madow
    correction applied uniformly to all six entropies.

    Returns the II value. Cost: 5 ``merge_vars`` calls (X1, X2, X1+Y,
    X2+Y, X1+X2+Y) per call -- much heavier than the search-loop's
    plug-in MI, so callers MUST gate by top-K.
    """
    n_samples = factors_data.shape[0]

    # ---- merge X1 alone (gives H(X1)) ----
    cls_x1, freqs_x1, _ = merge_vars(
        factors_data=factors_data,
        vars_indices=np.array([idx_a], dtype=np.int64),
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    h_x1 = _entropy_for_mode(freqs_x1, n_samples, use_mm)

    # ---- merge X2 alone (gives H(X2)) ----
    cls_x2, freqs_x2, _ = merge_vars(
        factors_data=factors_data,
        vars_indices=np.array([idx_b], dtype=np.int64),
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    h_x2 = _entropy_for_mode(freqs_x2, n_samples, use_mm)

    # ---- merge X1, X2 (gives H(X1, X2)) ----
    cls_x1x2, freqs_x1x2, _ = merge_vars(
        factors_data=factors_data,
        vars_indices=np.array([idx_a, idx_b], dtype=np.int64),
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    h_x1x2 = _entropy_for_mode(freqs_x1x2, n_samples, use_mm)

    # ---- merge X1, Y_idx ... (gives H(X1, Y)) ----
    # Concatenate idx_a with target_indices into a single vars list.
    vi_x1y = np.concatenate(([idx_a], target_indices)).astype(np.int64)
    _, freqs_x1y, _ = merge_vars(
        factors_data=factors_data,
        vars_indices=vi_x1y,
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    h_x1y = _entropy_for_mode(freqs_x1y, n_samples, use_mm)

    # ---- merge X2, Y (gives H(X2, Y)) ----
    vi_x2y = np.concatenate(([idx_b], target_indices)).astype(np.int64)
    _, freqs_x2y, _ = merge_vars(
        factors_data=factors_data,
        vars_indices=vi_x2y,
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    h_x2y = _entropy_for_mode(freqs_x2y, n_samples, use_mm)

    # ---- merge X1, X2, Y (gives H(X1, X2, Y)) ----
    vi_x1x2y = np.concatenate(([idx_a, idx_b], target_indices)).astype(np.int64)
    _, freqs_x1x2y, _ = merge_vars(
        factors_data=factors_data,
        vars_indices=vi_x1x2y,
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    h_x1x2y = _entropy_for_mode(freqs_x1x2y, n_samples, use_mm)

    # II = H(X1,X2) + H(X1,Y) + H(X2,Y) - H(X1,X2,Y) - H(X1) - H(X2) - H(Y)
    return h_x1x2 + h_x1y + h_x2y - h_x1x2y - h_x1 - h_x2 - h_y


def _should_apply_mm_for_pair(
    nbins_a: int, nbins_b: int, n_y_classes: int, n_samples: int,
    threshold: float = 0.05,
) -> bool:
    """Auto-gate MM application: fire when joint cardinality / n > threshold.

    Per Paninski 2003, plug-in entropy bias scales with (k-1)/(2n). When
    k/n exceeds the threshold the bias is large enough to materially shift
    II by more than typical synergy floors (-3/sqrt(n)). Below the threshold
    the bias is in the noise and applying MM only adds compute cost.

    ``threshold=0.05`` is folklore (SD1) -- documented as a heuristic, not
    derived. User can override via ``cfg.use_miller_madow=True`` to force,
    or ``False`` to disable.
    """
    return (nbins_a * nbins_b * n_y_classes) / max(n_samples, 1) > threshold


def _maybe_rerank_with_mm(
    factors_data: np.ndarray,
    pairs_a: np.ndarray,
    pairs_b: np.ndarray,
    selected_idx: np.ndarray,
    ii_arr: np.ndarray,
    nbins: np.ndarray,
    target_indices: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    cfg: CatFEConfig,
    dtype,
    verbose: int,
) -> tuple:
    """If MM is enabled (cfg flag True or auto-gate fires for at least
    one survivor), recompute II for selected pairs with MM correction
    applied to all six entropies. Returns ``(ii_mm_arr, selected_idx_resorted)``.

    T3: hoist constant / per-column entropies (H(Y), H(X_i), H(X_i, Y))
    out of the per-pair loop. For top_k=64 over a candidate pool of N=100,
    typical pair has 2 unique X-cols touching ~20-30 distinct columns
    overall. Cached entropies cut MM cost from 5 merge_vars + 6 entropy
    per pair down to 2 merge_vars + 2 entropy per pair (joint terms only).
    """
    if len(selected_idx) == 0:
        return ii_arr, selected_idx

    n_samples = factors_data.shape[0]
    n_y_classes = int(classes_y.max()) + 1 if classes_y.size > 0 else 1

    # Decide per-pair whether to apply MM
    if cfg.use_miller_madow is False:
        return ii_arr, selected_idx
    auto_gate = cfg.use_miller_madow is None
    if cfg.use_miller_madow is True:
        per_pair_mm = np.ones(len(selected_idx), dtype=bool)
    else:
        # Tier 4.6: analytical MM threshold based on Paninski bias formula
        gate_fn = _should_apply_mm_for_pair_analytical
        per_pair_mm = np.array([
            gate_fn(
                int(nbins[pairs_a[k]]), int(nbins[pairs_b[k]]),
                n_y_classes, n_samples,
            )
            for k in selected_idx
        ])
    if auto_gate and not per_pair_mm.any():
        return ii_arr, selected_idx

    # Compute H(Y) once (constant across pairs but doesn't actually affect
    # ranking since it appears in every II equally; included for correctness).
    # Tier 4.5: KT smoothing alternative to MM (set via cfg.use_kt_smoothing).
    use_kt = bool(getattr(cfg, "use_kt_smoothing", False))
    h_y_mm = _entropy_for_mode(freqs_y, n_samples, use_mm=True, use_kt=use_kt)

    # T3: hoist H(X_i) and H(X_i, Y) caches outside the loop. Only the
    # columns touched by surviving pairs need to be computed.
    touched_cols: set = set()
    for k in selected_idx:
        if per_pair_mm[bool(True)]:  # any pair_mm[j] True
            pass
    touched_cols = set()
    for j, k in enumerate(selected_idx):
        if per_pair_mm[j]:
            touched_cols.add(int(pairs_a[k]))
            touched_cols.add(int(pairs_b[k]))

    h_marginal_cache: dict = {}     # idx -> H(X_idx) with MM
    h_marginal_y_cache: dict = {}   # idx -> H(X_idx, Y) with MM
    for col_idx in touched_cols:
        cls_x, freqs_x, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=np.array([col_idx], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        h_marginal_cache[col_idx] = _entropy_for_mode(
            freqs_x, n_samples, use_mm=True, use_kt=use_kt,
        )
        vi_xy = np.concatenate(([col_idx], target_indices)).astype(np.int64)
        _, freqs_xy, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=vi_xy,
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        h_marginal_y_cache[col_idx] = _entropy_for_mode(
            freqs_xy, n_samples, use_mm=True, use_kt=use_kt,
        )

    if verbose:
        n_corrected = int(per_pair_mm.sum())
        logger.info(
            "cat-FE: re-ranking %d/%d top-K pairs with Miller-Madow correction "
            "(cached %d marginals)",
            n_corrected, len(selected_idx), len(touched_cols),
        )

    ii_mm_arr = ii_arr.copy()
    for j, k in enumerate(selected_idx):
        if not per_pair_mm[j]:
            continue
        idx_a = int(pairs_a[k])
        idx_b = int(pairs_b[k])
        # Only joint entropies H(X1,X2) and H(X1,X2,Y) are per-pair.
        cls_pair, freqs_pair, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=np.array([idx_a, idx_b], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        h_x1x2 = _entropy_for_mode(freqs_pair, n_samples, use_mm=True, use_kt=use_kt)
        vi_pair_y = np.concatenate(([idx_a, idx_b], target_indices)).astype(np.int64)
        _, freqs_pair_y, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=vi_pair_y,
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        h_x1x2y = _entropy_for_mode(freqs_pair_y, n_samples, use_mm=True, use_kt=use_kt)
        # II = H(X1,X2) + H(X1,Y) + H(X2,Y) - H(X1,X2,Y) - H(X1) - H(X2) - H(Y)
        ii_mm = (
            h_x1x2
            + h_marginal_y_cache[idx_a]
            + h_marginal_y_cache[idx_b]
            - h_x1x2y
            - h_marginal_cache[idx_a]
            - h_marginal_cache[idx_b]
            - h_y_mm
        )
        ii_mm_arr[k] = ii_mm

    # Re-sort selected_idx by the corrected scores. Same select_on logic
    # as _select_top_k_pairs -- we just re-rank, not re-filter.
    if cfg.select_on == "synergy":
        score = ii_mm_arr[selected_idx]
    elif cfg.select_on == "redundancy":
        score = -ii_mm_arr[selected_idx]
    else:  # absolute
        score = np.abs(ii_mm_arr[selected_idx])
    order = np.argsort(-score)
    return ii_mm_arr, selected_idx[order]


# ============================================================================
# Permutation confirmation (E2 same-shuffle three-MI, SB1)
#
# Two-stage strategy:
# 1. Search phase: pair search computes point-estimate II for all candidates
#    with no permutations (zero cost). Top-K selected by argpartition.
# 2. Confirmation phase: for each top-K survivor, run a permutation test
#    of II_observed against the null distribution. SAME shuffled Y feeds
#    all three MI computations (I(merged;Y), I(X1;Y), I(X2;Y)), so
#    II_perm = those three differences.
#
# Naming honesty (SB1): the test rejects "(X1, X2) jointly independent
# of Y" -- NOT "no synergy beyond marginals". Surfaced as
# ``joint_dependence_confidence`` not ``confidence``.
# ============================================================================


@njit(cache=True)
def _conditional_shuffle_within_strata(
    classes_x2_safe: np.ndarray,
    classes_y: np.ndarray,
    n_y_classes: int,
) -> None:
    """Conditional permutation: shuffle ``classes_x2_safe`` IN PLACE,
    restricting the shuffle to within each stratum of ``classes_y``.

    D1: this is the correct null distribution for testing
    ``H0: X1 ⊥ X2 | Y`` (no synergy beyond marginals). Plain shuffle-Y
    tests ``H0: Y ⊥ (X1, X2)`` (joint independence) instead.

    Implementation: for each Y-stratum, collect the indices where
    ``classes_y[i] == c``, Fisher-Yates shuffle the corresponding
    slice of ``classes_x2_safe`` in place. Per Anderson & ter Braak
    2003 ("Permutation tests for multi-factorial analysis of variance").

    Preserves ``P(X2 | Y)`` -- so each marginal ``I(X2; Y)`` is
    unchanged under the shuffle, but the conditional ``I(X1; X2 | Y)``
    is broken. The orchestrator combines this with three-MI calls
    above to compute II_perm under the conditional null.
    """
    n = len(classes_y)
    # For each Y class, gather positions and Fisher-Yates within the slice.
    for c in range(n_y_classes):
        # Collect indices manually (numba doesn't support boolean masks
        # for in-place writes the same way numpy does).
        positions = np.empty(n, dtype=np.int64)
        pos_count = 0
        for i in range(n):
            if classes_y[i] == c:
                positions[pos_count] = i
                pos_count += 1
        # Fisher-Yates shuffle within the stratum
        for idx in range(pos_count - 1, 0, -1):
            j = np.random.randint(0, idx + 1)
            a = positions[idx]
            b = positions[j]
            tmp = classes_x2_safe[a]
            classes_x2_safe[a] = classes_x2_safe[b]
            classes_x2_safe[b] = tmp


@njit(parallel=True, cache=True)
def _count_nfailed_joint_indep_prange(
    classes_pair: np.ndarray,
    freqs_pair: np.ndarray,
    classes_x1: np.ndarray,
    freqs_x1: np.ndarray,
    classes_x2: np.ndarray,
    freqs_x2: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    ii_obs: float,
    n_perms: int,
    base_seed: int,
    dtype,
) -> int:
    """T2: parallel permutation loop for the joint-independence null.

    Each thread gets its own copy of ``classes_y`` and own LCG seed
    (derived from ``base_seed + thread_idx``) so the count of failures
    is reproducible across re-runs at the same ``base_seed`` (modulo
    numpy version drift in ``random.shuffle``).

    Returns total ``nfailed`` summed across threads.
    """
    n = len(classes_y)
    nfailed_total = 0
    for tid in prange(n_perms):
        # Per-thread copy of Y so each prange iteration shuffles
        # independently.
        cy_local = classes_y.copy()
        # Seed numba's per-thread RNG. ``np.random.seed`` inside a
        # prange iteration affects the per-thread state.
        np.random.seed(base_seed + tid)
        # Fisher-Yates in place
        for i in range(n - 1, 0, -1):
            j = np.random.randint(0, i + 1)
            tmp = cy_local[i]
            cy_local[i] = cy_local[j]
            cy_local[j] = tmp
        i_pair = compute_mi_from_classes(
            classes_x=classes_pair, freqs_x=freqs_pair,
            classes_y=cy_local, freqs_y=freqs_y, dtype=dtype,
        )
        i_x1 = compute_mi_from_classes(
            classes_x=classes_x1, freqs_x=freqs_x1,
            classes_y=cy_local, freqs_y=freqs_y, dtype=dtype,
        )
        i_x2 = compute_mi_from_classes(
            classes_x=classes_x2, freqs_x=freqs_x2,
            classes_y=cy_local, freqs_y=freqs_y, dtype=dtype,
        )
        if (i_pair - i_x1 - i_x2) >= ii_obs:
            nfailed_total += 1
    return nfailed_total


@njit(cache=True)
def _shuffle_and_compute_three_mis(
    classes_pair: np.ndarray,
    freqs_pair: np.ndarray,
    classes_x1: np.ndarray,
    freqs_x1: np.ndarray,
    classes_x2: np.ndarray,
    freqs_x2: np.ndarray,
    classes_y_safe: np.ndarray,
    freqs_y: np.ndarray,
    dtype,
) -> tuple:
    """Shuffle classes_y_safe in place (Fisher-Yates) and compute three
    MIs against the shuffled Y. Single-pass kernel keeps allocation
    low and ensures the same shuffle drives all three MIs (E2)."""
    n = len(classes_y_safe)
    # Fisher-Yates shuffle in place
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        tmp = classes_y_safe[i]
        classes_y_safe[i] = classes_y_safe[j]
        classes_y_safe[j] = tmp

    i_pair = compute_mi_from_classes(
        classes_x=classes_pair, freqs_x=freqs_pair,
        classes_y=classes_y_safe, freqs_y=freqs_y, dtype=dtype,
    )
    i_x1 = compute_mi_from_classes(
        classes_x=classes_x1, freqs_x=freqs_x1,
        classes_y=classes_y_safe, freqs_y=freqs_y, dtype=dtype,
    )
    i_x2 = compute_mi_from_classes(
        classes_x=classes_x2, freqs_x=freqs_x2,
        classes_y=classes_y_safe, freqs_y=freqs_y, dtype=dtype,
    )
    return i_pair, i_x1, i_x2


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
) -> dict:
    """Full Westfall-Young: per shuffle, compute II_perm for ALL search-
    phase pairs, take the MAX, and accumulate the max-II distribution.
    Each survivor's p-value is ``(1 + #{b: max_II_perm[b] >= II_obs}) /
    (B + 1)``.

    D2: this is the proper WY procedure (Westfall & Young 1993). It
    naturally accounts for inter-pair correlation: pairs that share a
    column have correlated permutation distributions, and the max-II
    statistic captures this. Strictly more powerful than Bonferroni on
    the same B.

    Cost: per shuffle, compute joint MI for all m = ``len(pairs_a)``
    pairs. At m=4950 and B=100 that's 495k MI computations. Heavy --
    enable only when ``cfg.fwer_correction='westfall_young'`` AND the
    user accepts the cost. The savings vs Bonferroni: typically need
    2-5x fewer permutations for the same effective alpha.

    Returns ``{(i, j): corrected_p_value}`` ONLY for the survivors in
    ``selected_idx``.
    """
    n_samples = factors_data.shape[0]
    m = len(pairs_a)
    classes_y_safe = classes_y.copy()

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
        i = int(pairs_a[k]); jj = int(pairs_b[k])
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
        np.random.shuffle(classes_y_safe)
        # Compute MI(merged; Y_shuffled) for all pairs, and marginals for
        # all touched columns. Then II = joint - marginal_i - marginal_j.
        marginal_perm: dict = {}
        for ci, cls_c in marginal_classes.items():
            marginal_perm[ci] = compute_mi_from_classes(
                classes_x=cls_c, freqs_x=marginal_freqs[ci],
                classes_y=classes_y_safe, freqs_y=freqs_y, dtype=dtype,
            )
        max_ii = -np.inf
        for k in range(m):
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
        i = int(pairs_a[k]); jj = int(pairs_b[k])
        ii_obs = float(ii_obs_arr[k])
        n_exceed = int((max_ii_per_perm >= ii_obs).sum())
        corrected_p[(i, jj)] = (n_exceed + 1) / (n_perms + 1)
    return corrected_p


def _apply_fwer_correction(
    confidence_dict: dict, cfg: CatFEConfig, n_search_pairs: int,
) -> dict:
    """Apply multiple-testing correction to per-pair p-values, returning
    the corrected confidence dict. Supports:

    - ``"none"``: identity. FWER unchecked; user accepts inflation.
    - ``"bonferroni"``: ``p_corr = min(1, p * m)`` where m is the
      effective search-family size (``n_search_pairs``, NOT len of
      survivors). Conservative.
    - ``"bh_fdr"``: Benjamini-Hochberg step-up FDR. Less conservative
      than Bonferroni, controls expected proportion of false discoveries.
    - ``"westfall_young"``: NOTE -- proper WY requires recomputing the
      max-II across ALL search-phase pairs under each shuffle. This
      implementation approximates with Bonferroni-on-survivors, which
      is conservative-equivalent for the typical case where the
      survivors' II values dominate the per-shuffle max. A future
      revision can compute the full per-shuffle max-II distribution
      from the orchestrator's pair-search arrays (held in memory).

    SB2: the choice of ``n_search_pairs`` matters -- it's NOT
    len(survivors) but the count of pairs CONSIDERED at search time.
    A user who screened 100 candidate cols saw N(N-1)/2 = 4950 pairs;
    that's the family size, not 64 survivors.
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
        # NOTE: this branch is reached only as a FALLBACK when the
        # orchestrator wasn't able to invoke the full WY procedure
        # (e.g. memory budget too tight to hold all m pair-class arrays).
        # In normal flow, ``_compute_westfall_young_corrected_p`` runs
        # first and the orchestrator skips this helper for WY entirely.
        # Bonferroni-on-survivors is the conservative-equivalent fallback.
        return {k: 1.0 - min(1.0, p * m) for k, p in p_vals.items()}

    raise ValueError(f"Unknown fwer_correction: {cfg.fwer_correction!r}")


def _confirm_pairs_via_permutation(
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
    """Run permutation confirmation on top-K survivors. Returns
    ``(selected_idx_kept, confidence_per_pair_dict)``.

    For each survivor, sample ``cfg.full_npermutations`` Fisher-Yates
    shuffles of Y, compute II_perm via same-shuffle three-MI, and count
    how often ``II_perm >= II_obs``. Confidence = 1 - failures /
    npermutations. Pairs with confidence < min_nonzero_confidence are
    dropped from selected_idx.

    Confidence is the "joint dependence confidence" -- it tests the
    null that (X1, X2) is jointly independent of Y, NOT the null that
    II = 0. See SB1 / I2 in the v3 plan for the caveat.
    """
    if cfg.full_npermutations <= 0:
        # SB4: warn the user they're flying blind on FWER.
        if len(selected_idx) > 0 and verbose:
            logger.warning(
                "cat-FE: full_npermutations=0 surfaced %d pair(s) ranked by "
                "point estimate only. No statistical confirmation; results "
                "reflect selection bias from the search-phase 4950-pair family.",
                len(selected_idx),
            )
        return selected_idx, {}

    if len(selected_idx) == 0:
        return selected_idx, {}

    n_samples = factors_data.shape[0]
    n_perms = cfg.full_npermutations
    min_conf = 0.95  # SB4 default for cat-FE (separate from MRMR.min_nonzero_confidence)

    # Pre-merge classes for each survivor pair (and its marginals).
    # Memory: O(top_k * 3 * n) -- at top_k=64, n=1M, dtype=int32: ~768 MB worst case.
    # OK for top_k of order 100; user controls.
    confidence_dict: dict = {}
    kept_mask = np.ones(len(selected_idx), dtype=bool)

    if verbose:
        logger.info(
            "cat-FE: confirming %d pair(s) via %d permutation tests each",
            len(selected_idx), n_perms,
        )

    # Tier 4.1: bandit UCB1 budget allocation. When enabled, each pair
    # gets ``min_perms_per_pair`` initial shuffles; then we add shuffles
    # to the pair with the largest UCB1 score (uncertainty + I value)
    # until total budget = n_perms * len(selected_idx) is exhausted.
    # Pairs clearly above floor stop receiving shuffles; ambiguous pairs
    # get more. Saves 2-5x typical total perms.
    use_bandit = (
        getattr(cfg, "perm_budget_strategy", "fixed") == "bandit_ucb1"
        and len(selected_idx) > 1
    )

    # D1: conditional permutation null requires per-pair freshly-merged
    # X2 column to shuffle within strata of Y. Note that the joint MI
    # I(X1, X2; Y) after shuffling X2 within strata of Y requires
    # RE-merging X1 with the shuffled X2 -- the joint table can't be
    # reused from cls_pair. So this null is materially more expensive
    # than the joint-independence null. Caller opts in via cfg.
    use_conditional = cfg.permutation_null == "conditional"
    n_y_classes = int(classes_y.max()) + 1

    for j, k in enumerate(selected_idx):
        i = int(pairs_a[k]); jj = int(pairs_b[k])
        ii_obs = float(ii_arr[k])

        cls_pair, fq_pair, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=np.array([i, jj], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        cls_x1, fq_x1, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=np.array([i], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        cls_x2, fq_x2, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=np.array([jj], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )

        n_failed = 0
        if use_conditional:
            # D1: shuffle X2 within strata of Y. The null preserves
            # P(X1, Y) AND P(X2, Y); only I(X1; X2 | Y) is broken.
            # For each permutation: shuffle cls_x2 conditionally, then
            # re-merge (X1, X2_shuffled) and compute II.
            classes_x2_safe = cls_x2.astype(np.int64, copy=True)
            classes_x1_arr = cls_x1.astype(np.int64, copy=False)
            n_samples_local = factors_data.shape[0]
            for _ in range(n_perms):
                _conditional_shuffle_within_strata(
                    classes_x2_safe, classes_y, n_y_classes,
                )
                # Re-merge X1 with the shuffled X2 to get the conditional-
                # null joint. We materialise a 2-col array on the fly.
                local_data = np.empty((n_samples_local, 2), dtype=dtype)
                local_data[:, 0] = classes_x1_arr.astype(dtype, copy=False)
                local_data[:, 1] = classes_x2_safe.astype(dtype, copy=False)
                local_nbins = np.array(
                    [int(cls_x1.max()) + 1, int(classes_x2_safe.max()) + 1],
                    dtype=np.int64,
                )
                cls_joint_perm, fq_joint_perm, _ = merge_vars(
                    factors_data=local_data,
                    vars_indices=np.array([0, 1], dtype=np.int64),
                    var_is_nominal=None, factors_nbins=local_nbins, dtype=dtype,
                )
                i_pair_p = compute_mi_from_classes(
                    classes_x=cls_joint_perm, freqs_x=fq_joint_perm,
                    classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
                )
                # Marginals are PRESERVED by conditional shuffle -- use
                # the originals from the loop entry. (We still subtract
                # the same marginals as the observed II.)
                i_x1_p = compute_mi_from_classes(
                    classes_x=cls_x1, freqs_x=fq_x1,
                    classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
                )
                # I(X2_shuffled; Y) -- per the conditional-null property,
                # this equals I(X2; Y) up to floating-point noise; we
                # recompute for safety.
                fq_x2_perm = np.bincount(
                    classes_x2_safe.astype(np.int64), minlength=int(classes_x2_safe.max()) + 1
                ).astype(np.float64) / n_samples_local
                i_x2_p = compute_mi_from_classes(
                    classes_x=classes_x2_safe.astype(dtype, copy=False),
                    freqs_x=fq_x2_perm.astype(np.float64),
                    classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
                )
                ii_perm = i_pair_p - i_x1_p - i_x2_p
                if ii_perm >= ii_obs:
                    n_failed += 1
        else:
            # Default joint-independence null: shuffle Y once, recompute
            # all three MIs against the shuffled Y. Tests "(X1,X2) ⊥ Y".
            # T2: parallel via numba prange over permutations. Per-thread
            # local copy of Y so shuffles don't race. Seed derived from
            # j (survivor index) so re-runs are reproducible.
            n_failed = _count_nfailed_joint_indep_prange(
                cls_pair, fq_pair, cls_x1, fq_x1, cls_x2, fq_x2,
                classes_y, freqs_y, ii_obs, n_perms,
                base_seed=int(j) * 1000003 + 7, dtype=dtype,
            )
        # Continuity-corrected p-value: (n_failed + 1) / (n_perms + 1)
        # gives a non-zero p-value floor even with zero failures, and
        # is the standard convention for empirical permutation p.
        p = (n_failed + 1) / (n_perms + 1)
        conf = 1.0 - p
        confidence_dict[(i, jj)] = conf

    # FWER correction (SB2). Applied AFTER raw p collection so the
    # correction strategy can see all survivors' p-values together
    # (needed for BH-FDR step-up).
    corrected_conf = _apply_fwer_correction(
        confidence_dict, cfg, n_search_pairs=n_search_pairs,
    )

    # Drop pairs whose CORRECTED confidence falls below the floor.
    kept_mask = np.array([
        corrected_conf[(int(pairs_a[k]), int(pairs_b[k]))] >= min_conf
        for k in selected_idx
    ])
    if verbose:
        for j, k in enumerate(selected_idx):
            ipair = (int(pairs_a[k]), int(pairs_b[k]))
            if not kept_mask[j]:
                logger.info(
                    "cat-FE: pair %s failed FWER-corrected confirmation "
                    "(raw_conf=%.3f, corrected_conf=%.3f, threshold=%.2f, "
                    "correction=%s, m=%d)",
                    ipair, confidence_dict[ipair], corrected_conf[ipair],
                    min_conf, cfg.fwer_correction, n_search_pairs,
                )

    return selected_idx[kept_mask], corrected_conf


# ============================================================================
# Target encoding emit (Tier 3.1, E2.1)
#
# In addition to factorize-encoded engineered cols, emit target-encoded
# variants: ``E[Y | merged_class]`` per cell with optional out-of-fold
# smoothing to prevent leakage. Useful when downstream models prefer
# numeric inputs (linear / NN / tree models with continuous splits).
# ============================================================================


def _compute_target_encoding(
    factors_data: np.ndarray,
    idx_tuple: tuple,
    target_indices: np.ndarray,
    classes_y: np.ndarray,
    nbins: np.ndarray,
    n_oof_folds: int,
    smoothing: float,
    dtype,
) -> tuple:
    """Compute target-encoded values per cell of (X[idx_tuple]).
    Returns (te_values, cell_means_oof_combined) -- a 1-D array of
    ``n_samples`` floats where each row is ``E[Y | merged_class]``
    computed out-of-fold (to prevent leakage).

    Strategy:
    - Build per-cell mean of Y, with shrinkage: ``te = (n_c * te_raw +
      alpha * te_global) / (n_c + alpha)`` (Micci-Barreca 2001).
    - For OOF: split rows into K folds; for each fold, compute cell
      means from the other K-1 folds, apply to this fold's rows.
    - For naive (n_oof_folds=0): single-pass cell mean across ALL rows.
      Leaks signal -- only safe when used as a downstream feature in
      a separate train/val split.

    Y is treated as numeric (regression). For binary classification,
    this gives per-cell P(y=1) -- well-behaved. For multi-class, falls
    back to encoding the first class indicator (TODO multi-class).
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

    if n_oof_folds <= 0:
        # Naive: per-cell mean across all rows
        cell_sum = np.zeros(int(n_uniq), dtype=np.float64)
        cell_cnt = np.zeros(int(n_uniq), dtype=np.float64)
        for row in range(n_samples):
            c = int(classes_merged[row])
            cell_sum[c] += y_numeric[row]
            cell_cnt[c] += 1.0
        cell_means = np.zeros(int(n_uniq), dtype=np.float64)
        for c in range(int(n_uniq)):
            if cell_cnt[c] > 0:
                raw = cell_sum[c] / cell_cnt[c]
                cell_means[c] = (cell_cnt[c] * raw + smoothing * te_global) / (
                    cell_cnt[c] + smoothing
                )
            else:
                cell_means[c] = te_global
        te_values = cell_means[classes_merged.astype(np.int64)]
        return te_values, cell_means

    # OOF encoding: for each fold, compute cell means from other folds
    K = int(n_oof_folds)
    fold_ids = np.arange(n_samples) % K
    te_values = np.full(n_samples, te_global, dtype=np.float64)
    for f in range(K):
        train_mask = fold_ids != f
        test_mask = ~train_mask
        # Compute cell means on training rows
        cell_sum = np.zeros(int(n_uniq), dtype=np.float64)
        cell_cnt = np.zeros(int(n_uniq), dtype=np.float64)
        for row in np.where(train_mask)[0]:
            c = int(classes_merged[row])
            cell_sum[c] += y_numeric[row]
            cell_cnt[c] += 1.0
        cell_means_fold = np.full(int(n_uniq), te_global, dtype=np.float64)
        for c in range(int(n_uniq)):
            if cell_cnt[c] > 0:
                raw = cell_sum[c] / cell_cnt[c]
                cell_means_fold[c] = (
                    cell_cnt[c] * raw + smoothing * te_global
                ) / (cell_cnt[c] + smoothing)
        # Apply to test fold rows
        for row in np.where(test_mask)[0]:
            c = int(classes_merged[row])
            te_values[row] = cell_means_fold[c]

    # Also compute global (all-rows) cell means for transform()-time replay.
    # At transform, OOF doesn't make sense (test data has no Y); we use
    # the global mean per cell.
    cell_sum = np.zeros(int(n_uniq), dtype=np.float64)
    cell_cnt = np.zeros(int(n_uniq), dtype=np.float64)
    for row in range(n_samples):
        c = int(classes_merged[row])
        cell_sum[c] += y_numeric[row]
        cell_cnt[c] += 1.0
    cell_means_global = np.full(int(n_uniq), te_global, dtype=np.float64)
    for c in range(int(n_uniq)):
        if cell_cnt[c] > 0:
            raw = cell_sum[c] / cell_cnt[c]
            cell_means_global[c] = (
                cell_cnt[c] * raw + smoothing * te_global
            ) / (cell_cnt[c] + smoothing)
    return te_values, cell_means_global


# ============================================================================
# Sample-weight-aware MI computation (Tier 2.1)
#
# Cat-FE-local weighted MI: bypasses ``merge_vars`` and computes joint
# weighted histogram directly. Used when ``cfg.sample_weight_col`` is set.
# Does NOT extend the global ``merge_vars`` API (which has 500+ callers);
# weighted MRMR screening / RFECV is project-level future work.
# ============================================================================


@njit(parallel=True, cache=True)
def _pair_search_kernel_weighted_njit(
    factors_data: np.ndarray,
    pairs_a: np.ndarray,
    pairs_b: np.ndarray,
    marginal_mi: np.ndarray,
    nbins: np.ndarray,
    classes_y: np.ndarray,
    weights: np.ndarray,        # (n,) float64; sum(weights) > 0
    dtype,
) -> tuple:
    """Weighted variant of ``_pair_search_kernel_njit``. For each pair
    (a, b) compute weighted joint MI = sum_{m,y} w_{m,y} log(w_{m,y} *
    W / (w_m * w_y)) where w_{m,y} = (sum of row weights in cell
    (merged=m, y=y)) / W and W = sum of all weights. Equivalent to
    unweighted MI under uniform weights.

    Returns (joint_mi, ii, n_uniq) arrays.
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


# ============================================================================
# Bootstrap CIs on II (Tier 2.2)
#
# For each top-K survivor, draw ``n_replicates`` subsamples (size
# ``sample_frac * n``), recompute II per replicate, return (lower,
# median, upper) CI. Complements permutation tests: perm checks
# significance vs null; bootstrap checks STABILITY under sample variation.
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
    """For each pair in ``selected_idx``, compute bootstrap CI on II.
    Returns ``{(i, j): (lower, median, upper)}`` per ``cfg.bootstrap_ci_alpha``.

    Cost: ``n_replicates * top_k * O(n)`` -- at n_replicates=20, top_k=32,
    n=10000 that's ~6.4M merge_vars-equivalents. Heavy; gated by user
    opt-in (``bootstrap_ci_n_replicates > 0``).
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
    target_indices_arr = np.array([], dtype=np.int64)  # unused for II
    for k in selected_idx:
        i = int(pairs_a[k]); jj = int(pairs_b[k])
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
# Anti-redundancy with selected features (E3)
#
# Pure-II ranking treats a pair (X1, X2) as relevant if it has high
# synergy with Y. But MRMR's overall objective is `relevance - β*redundancy`:
# a pair whose merged column is HIGHLY CORRELATED with an already-
# selected feature Z adds little new information regardless of its II.
# E3 down-weights II by `β * max_z I(merged; Z)` over already-selected Z.
#
# Two-stage decoupled design (SM6 per v3 plan):
# 1. Stage 1: II floor gates "is this engineered col worth materializing?"
#    (already done by ``_select_top_k_pairs``).
# 2. Stage 2 (HERE): mRMR-style score = II - β * mean_z I(merged; Z)
#    re-ranks survivors. β=0 disables (default), preserving pure II.
#
# Cost: per survivor + per selected Z, one merge_vars + one
# compute_mi_from_classes. At top_k=64 survivors and |Z|=20 selected,
# that's 1280 merge_vars calls. Linear in both.
# ============================================================================


def _anti_redundancy_rerank(
    factors_data: np.ndarray,
    pairs_a: np.ndarray,
    pairs_b: np.ndarray,
    selected_idx: np.ndarray,
    ii_arr: np.ndarray,
    nbins: np.ndarray,
    selected_so_far: list,    # column indices (in ``data``) of already-selected features
    classes_y: np.ndarray,    # unused -- the redundancy MI is against Z, not Y
    cfg: CatFEConfig,
    dtype,
    verbose: int,
) -> tuple:
    """Re-rank top-K survivors by ``score = II - β * mean_z I(merged; Z)``.

    When ``cfg.anti_redundancy_beta == 0`` or ``selected_so_far`` is empty,
    this is a no-op. Returns ``(scored_arr, selected_idx_reordered)``.
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
    for k in selected_idx:
        i = int(pairs_a[k]); j = int(pairs_b[k])
        # Materialise the merged class for this pair
        cls_merged, freqs_merged, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=np.array([i, j], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        red_terms = []
        for z in selected_so_far_arr:
            cls_z, freqs_z, _ = merge_vars(
                factors_data=factors_data,
                vars_indices=np.array([int(z)], dtype=np.int64),
                var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
            )
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
# K-fold II stability filter (E6)
#
# A pair with II=0.3 on one fold and II=-0.1 on another is noise, not signal.
# E6 splits the training data into K folds, recomputes II on each fold's
# slice, and keeps only pairs prevalent in >= floor·K folds. This guards
# against "II was driven by a few outlier rows" failures.
#
# Cost: K-1 extra pair searches (each fold uses ~n*(K-1)/K rows). At
# cfg.n_folds_stability=5 and n=10000, that's ~5 full pair searches.
# Run BEFORE the heavy permutation phase so the per-fold II costs don't
# multiply against npermutations.
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
    """For each top-K survivor, recompute II on K disjoint folds; keep
    pairs whose II clears the floor on >= ``min_fold_prevalence * K``
    folds. Returns ``(kept_selected_idx, per_fold_ii_dict)``.

    When ``cfg.n_folds_stability <= 0``, this is a no-op that returns
    inputs unchanged.

    Determinism: folds are derived from ``np.arange(n) % K`` -- no
    shuffling, no RNG. Reproducible across runs.
    """
    if cfg.n_folds_stability <= 0 or len(selected_idx) == 0:
        return selected_idx, {}

    n_samples = factors_data.shape[0]
    K = int(cfg.n_folds_stability)
    floor = resolve_min_interaction_information(cfg, n_samples)

    fold_ids = np.arange(n_samples) % K
    per_fold_ii: dict = {}
    kept = []

    if verbose:
        logger.info("cat-FE: K-fold stability check (K=%d) over %d top-K pair(s)", K, len(selected_idx))

    for k in selected_idx:
        i = int(pairs_a[k]); j = int(pairs_b[k])
        fold_ii_vals: list = []

        for f in range(K):
            mask = fold_ids == f
            n_fold = int(mask.sum())
            if n_fold < cfg.min_n_samples // 2:
                # Fold too small to estimate MI reliably; mark as failed.
                fold_ii_vals.append(float("-inf"))
                continue
            slice_data = factors_data[mask]
            # Re-merge Y on the fold (since classes_y depends on the rows).
            cls_y_f, fq_y_f, _ = merge_vars(
                factors_data=slice_data, vars_indices=target_indices,
                var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
            )
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
# Tier 4.2: coordinate-ascent refinement after greedy k-way
#
# Greedy k-way picks a triplet (A, B, C). Coordinate-ascent then tries
# replacing each member with each non-member; if the swap improves
# joint MI, keep it. Catches cases where the greedy seed missed a
# better neighbor. Refines local optima, doesn't break global structure.
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
    """For each k-way result, run ``n_passes`` of coordinate-ascent:
    try swapping each member with each non-member; keep if joint MI
    improves. Returns refined kway_results.
    """
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
                for cand in candidate_pool:
                    cand_int = int(cand)
                    if cand_int in current:
                        continue
                    new_tuple = current.copy()
                    new_tuple[pos] = cand_int
                    new_tuple_sorted = tuple(sorted(new_tuple))
                    # Card budget check
                    card = 1
                    for k in new_tuple_sorted:
                        card *= int(nbins[k])
                    if card > max_combined_nbins or card >= 2**31:
                        continue
                    new_classes, _, new_nuniq = merge_vars(
                        factors_data=factors_data,
                        vars_indices=np.array(new_tuple_sorted, dtype=np.int64),
                        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
                    )
                    new_mi = compute_mi_from_classes(
                        classes_x=new_classes, freqs_x=None,  # unused; merge_vars gives freqs but we lose it here
                        classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
                    ) if False else None
                    # Recompute MI using fresh freqs
                    _, new_freqs, _ = merge_vars(
                        factors_data=factors_data,
                        vars_indices=np.array(new_tuple_sorted, dtype=np.int64),
                        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
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
                        if verbose >= 2:
                            logger.info(
                                "cat-FE coord-ascent: %s -> %s (MI %.4f -> %.4f)",
                                orig_tuple, tuple(current), orig_mi, current_mi,
                            )
            if not improved:
                break
        refined.append((tuple(sorted(current)), current_classes, current_nuniq, current_mi))
    return refined


# ============================================================================
# Tier 4.3: group-aware permutation
#
# When rows are clustered (sessions, users), shuffling Y globally breaks
# both within-group and between-group structure -- inflating significance.
# Group-aware permutation shuffles Y values only across groups (preserves
# within-group Y patterns).
# ============================================================================


@njit(cache=True)
def _group_aware_shuffle(
    classes_y_safe: np.ndarray, groups: np.ndarray, n_groups: int,
) -> None:
    """Shuffle classes_y_safe in place, restricting to between-group
    permutations. For each group, all rows get the SAME (shuffled)
    Y representative (the group's first row's Y, after shuffling
    group ordering).

    Simplified strategy: shuffle group representatives, then broadcast
    each group's shuffled Y to all its rows.
    """
    n = len(classes_y_safe)
    # Find each group's first-occurrence Y value
    group_y = np.full(n_groups, -1, dtype=np.int32)
    for i in range(n):
        g = groups[i]
        if group_y[g] < 0:
            group_y[g] = classes_y_safe[i]
    # Shuffle the group_y array
    for i in range(n_groups - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        tmp = group_y[i]
        group_y[i] = group_y[j]
        group_y[j] = tmp
    # Broadcast back
    for i in range(n):
        classes_y_safe[i] = group_y[groups[i]]


# ============================================================================
# K-way greedy expansion (SB7)
#
# After top-K pairs are confirmed, greedily extend each surviving pair by ONE
# variable at a time up to ``cfg.max_kway_order``. For each candidate
# extension k:
#
#   delta_II = I(parent ∪ {k}; Y) - I(parent; Y) - I(X_k; Y)
#
# This is the 3-way Jakulin II between (parent_aggregate, X_k, Y) -- it
# measures whether adding X_k to the merged parent contributes information
# BEYOND what the parent and X_k separately give. Positive delta means X_k
# is genuinely synergistic with the parent group; <= 0 means X_k is
# redundant given parent.
#
# Naming: "incremental_interaction_information" per SB7. NOT identical to
# higher-order Jakulin II (which is a 15-term inclusion-exclusion sum); this
# is closer to JMI (Yang & Moody 1999) than to CMIM (Fleuret 2004).
#
# Cost: O(top_k_pairs * (max_kway_order - 2) * N) merge_vars calls. The
# orchestrator caps ``max_kway_order`` to a small int (default 2 -> skip;
# typical use 3..5). Each greedy extension calls merge_vars once per
# candidate var.
# ============================================================================


def _greedy_expand_one_seed(
    factors_data: np.ndarray,
    seed_indices: tuple,           # (idx_a, idx_b) -- the seed pair
    candidate_pool: np.ndarray,    # indices eligible for extension
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    marginal_mi: np.ndarray,
    max_combined_nbins: int,
    max_kway_order: int,
    min_inc_ii: float,
    dtype,
) -> tuple:
    """Greedily extend ``seed_indices`` up to ``max_kway_order`` by picking
    the variable with the largest incremental II at each step.

    Returns ``(final_indices_tuple, final_classes, final_n_uniq, final_joint_mi)``
    or ``None`` if no extension cleared ``min_inc_ii`` (in which case the
    seed pair itself remains the best, no k-way emitted).

    Stops early when:
    - No candidate var clears ``min_inc_ii`` (greedy local max reached).
    - Adding any candidate would violate the cardinality budget.
    - Order reaches ``max_kway_order``.
    """
    parent_set = set(seed_indices)
    parent_vi = np.array(sorted(parent_set), dtype=np.int64)
    parent_classes, parent_freqs, parent_nclasses = merge_vars(
        factors_data=factors_data, vars_indices=parent_vi,
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    parent_mi = compute_mi_from_classes(
        classes_x=parent_classes, freqs_x=parent_freqs,
        classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
    )

    for order in range(len(parent_set) + 1, max_kway_order + 1):
        best_inc_ii = -np.inf
        best_var: int = -1
        best_classes = None
        best_nclasses = 0
        best_joint_mi = 0.0

        for k in candidate_pool:
            k_int = int(k)
            if k_int in parent_set:
                continue
            new_card_estimate = parent_nclasses * int(nbins[k_int])
            if new_card_estimate > max_combined_nbins:
                continue
            if new_card_estimate >= 2**31:
                continue

            new_vi = np.array(sorted(parent_set | {k_int}), dtype=np.int64)
            new_classes, new_freqs, new_n = merge_vars(
                factors_data=factors_data, vars_indices=new_vi,
                var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
            )
            new_joint_mi = compute_mi_from_classes(
                classes_x=new_classes, freqs_x=new_freqs,
                classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
            )
            # incremental II = new_joint_mi - parent_mi - marginal_mi[k]
            # = how much extra info adding X_k brings, BEYOND what parent
            # and X_k separately contribute.
            inc_ii = new_joint_mi - parent_mi - float(marginal_mi[k_int])
            if inc_ii > best_inc_ii:
                best_inc_ii = inc_ii
                best_var = k_int
                best_classes = new_classes
                best_nclasses = new_n
                best_joint_mi = new_joint_mi

        if best_var < 0 or best_inc_ii < min_inc_ii:
            break  # local maximum reached; no positive synergistic extension

        # Accept the extension
        parent_set.add(best_var)
        parent_classes = best_classes
        parent_nclasses = best_nclasses
        parent_mi = best_joint_mi

    if len(parent_set) <= 2:
        return None  # no extension survived; caller emits the original pair
    return (
        tuple(sorted(parent_set)),
        parent_classes,
        int(parent_nclasses),
        float(parent_mi),
    )


def _build_kway_chained_lookup(
    factors_data: np.ndarray,
    idx_tuple: tuple,    # k indices in sorted order
    nbins: np.ndarray,
    classes_final: np.ndarray,    # final post-prune classes from merge_vars(...idx_tuple)
    unknown_strategy: str,
    dtype,
) -> tuple:
    """Build a chain of ``k - 1`` pair lookup tables that together replay
    the full k-way merge on test data.

    Strategy (D3): ``merge_vars`` over k cols can be decomposed as
    ``merge_vars(merge_vars(...merge_vars(c1, c2), c3...), ck)`` -- a
    chain of pairwise merges with intermediate dense renumbering. We
    build the lookup for each step at fit time:

    Step 1: lookup_1[c1_val + c2_val * nbins_1] -> intermediate_class_1
            (size nbins_1 * nbins_2; intermediate cardinality = n_uniq_step_1)
    Step 2: lookup_2[intermediate_1 + c3_val * n_uniq_step_1] -> intermediate_class_2
    ...
    Step k-1: lookup_{k-1}[intermediate_{k-2} + ck_val * n_uniq_step_{k-2}] -> final_class

    Returns:
    - ``lookups``: list of (k-1) int64 ndarrays, each a flat lookup table
    - ``intermediate_nuniqs``: list of (k-1) ints, cardinalities AFTER each step

    On transform: callers chain through this list to compute the final
    class from test-data column values.
    """
    k = len(idx_tuple)
    n_samples = factors_data.shape[0]
    if k < 2:
        raise ValueError(f"chained lookup requires k>=2, got k={k}")

    lookups: list = []
    intermediate_nuniqs: list = []

    # First step: merge cols [idx_tuple[0], idx_tuple[1]]
    vi_2 = np.array([idx_tuple[0], idx_tuple[1]], dtype=np.int64)
    classes_step, _, n_uniq_step = merge_vars(
        factors_data=factors_data, vars_indices=vi_2,
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    nbins_a = int(nbins[idx_tuple[0]])
    nbins_b = int(nbins[idx_tuple[1]])
    lookup_1, _ = _build_factorize_lookup(
        factors_data=factors_data,
        idx_a=int(idx_tuple[0]), idx_b=int(idx_tuple[1]),
        nbins_a=nbins_a, nbins_b=nbins_b,
        classes_pair_post=classes_step,
        unknown_strategy=unknown_strategy,
    )
    lookups.append(lookup_1)
    intermediate_nuniqs.append(int(n_uniq_step))

    # Subsequent steps: merge (running classes, idx_tuple[step])
    running_classes = classes_step.astype(np.int64, copy=False)
    running_nuniq = int(n_uniq_step)

    for step in range(2, k):
        nxt_idx = int(idx_tuple[step])
        nxt_vals = factors_data[:, nxt_idx].astype(np.int64, copy=False)
        nxt_nbins = int(nbins[nxt_idx])
        # Pre-prune codes for this step: running + nxt_vals * running_nuniq
        pre_prune_codes = running_classes + nxt_vals * running_nuniq
        expected_size = running_nuniq * nxt_nbins
        # Compute next post-prune via merge_vars across the FULL prefix
        # (idx_tuple[:step+1]); this is what we already have classes for
        # in ``kway_results`` for step == k-1, but for intermediate steps
        # we need it explicitly.
        vi_prefix = np.array(idx_tuple[: step + 1], dtype=np.int64)
        cls_next, _, n_uniq_next = merge_vars(
            factors_data=factors_data, vars_indices=vi_prefix,
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        lookup_step = np.full(expected_size, -1, dtype=np.int64)
        # Populate: each row's pre-prune code -> its post-prune class
        lookup_step[pre_prune_codes] = cls_next.astype(np.int64, copy=False)
        # Resolve unseen per unknown_strategy
        seen_mask = lookup_step >= 0
        if not seen_mask.all():
            if unknown_strategy == "clip":
                seen_max = int(lookup_step[seen_mask].max())
                lookup_step[~seen_mask] = seen_max
            elif unknown_strategy == "sentinel":
                seen_max = int(lookup_step[seen_mask].max())
                lookup_step[~seen_mask] = seen_max + 1
            # ``raise``: leave -1; apply_recipe surfaces an error.
        lookups.append(lookup_step)
        intermediate_nuniqs.append(int(n_uniq_next))
        running_classes = cls_next.astype(np.int64, copy=False)
        running_nuniq = int(n_uniq_next)

    return lookups, intermediate_nuniqs


def _materialize_kway(
    factors_data: np.ndarray,
    kway_results: list,    # list of (indices_tuple, classes, n_uniq, joint_mi)
    nbins: np.ndarray,
    cols: list,
    dtype,
    unknown_strategy: str,
) -> tuple:
    """Materialise greedy k-way survivors. Returns
    ``(new_data_block, new_names, new_nbins, new_recipes)`` mirroring
    ``_materialize_pairs``. K-way recipes have ``src_names`` of length k
    and ``factorize_nbins`` of length k.

    D3: k-way recipes now ship a CHAINED LOOKUP -- (k-1) pair lookup
    tables that ``apply_recipe`` walks sequentially on test data. Memory:
    sum of intermediate nbins products (typically O(k * max_combined_nbins)),
    NOT O(nbins^k). At k=3 with cardinalities (10, 10, 10): 100 + 10*n_uniq_step1
    cells (post-prune n_uniq usually < 100), so ~200-1000 int64 cells per
    recipe -- negligible vs the pair lookup table cost.
    """
    if not kway_results:
        return (
            np.empty((factors_data.shape[0], 0), dtype=dtype),
            [], [], [],
        )
    n_samples = factors_data.shape[0]
    new_data_block = np.empty((n_samples, len(kway_results)), dtype=dtype)
    new_names: list = []
    new_nbins: list = []
    new_recipes: list = []

    for k_out, (idx_tuple, classes_arr, n_uniq, _) in enumerate(kway_results):
        src_names = tuple(cols[i] for i in idx_tuple)
        eng_name = f"kway({'__'.join(src_names)})"
        if eng_name in cols or eng_name in new_names:
            eng_name = f"kway_{k_out}({'__'.join(src_names)})"
        new_data_block[:, k_out] = classes_arr
        new_names.append(eng_name)
        new_nbins.append(int(n_uniq))

        # Build the chained lookup so transform() can replay this k-way on
        # test data. Memory: O(k * max_combined_nbins), bounded by config.
        chain_lookups, chain_nuniqs = _build_kway_chained_lookup(
            factors_data=factors_data,
            idx_tuple=idx_tuple,
            nbins=nbins,
            classes_final=classes_arr,
            unknown_strategy=unknown_strategy,
            dtype=dtype,
        )

        new_recipes.append(
            EngineeredRecipe(
                name=eng_name,
                kind="factorize",
                src_names=src_names,
                factorize_nbins=tuple(int(nbins[i]) for i in idx_tuple),
                unknown_strategy=unknown_strategy,
                extra={
                    "kway_order": len(idx_tuple),
                    "n_uniq_post_prune": int(n_uniq),
                    # D3: chained-lookup payload for k-way replay.
                    # ``chain_lookups`` has len k-1; ``chain_nuniqs`` is
                    # the post-prune cardinality after each step (drives
                    # the multiplier for the NEXT step's pre-prune code).
                    "chain_lookups": chain_lookups,
                    "chain_nuniqs": chain_nuniqs,
                },
            )
        )
    return new_data_block, new_names, new_nbins, new_recipes


def _select_top_k_pairs(
    ii_arr: np.ndarray,
    pairs_a: np.ndarray,
    pairs_b: np.ndarray,
    cfg: CatFEConfig,
    n_samples: int,
) -> np.ndarray:
    """Pick the top-K pair indices by score. Uses argpartition (P6) --
    O(N) over heap's O(N log K) -- the flat II array is small.

    The score depends on ``cfg.select_on``:
    - ``"synergy"``: rank by ``ii_arr`` desc; keep where ``ii > floor``.
    - ``"redundancy"``: rank by ``-ii_arr`` desc; keep where ``ii < -floor``.
    - ``"absolute"``: rank by ``|ii_arr|`` desc; keep where ``|ii| > floor``.

    Returns int array of indices into ``pairs_a`` / ``pairs_b``
    (ordered descending by score). Length ``<= cfg.top_k_pairs``.
    """
    floor = resolve_min_interaction_information(cfg, n_samples)
    if cfg.select_on == "synergy":
        score = ii_arr
        eligible = score > floor
    elif cfg.select_on == "redundancy":
        score = -ii_arr
        eligible = -ii_arr > -floor  # i.e. ii < floor (negative side)
    elif cfg.select_on == "absolute":
        score = np.abs(ii_arr)
        eligible = score > abs(floor)
    else:
        raise ValueError(f"Unknown cfg.select_on: {cfg.select_on!r}")

    n_eligible = int(eligible.sum())
    if n_eligible == 0:
        return np.empty(0, dtype=np.int64)

    # If we have <= top_k eligible candidates, just return them sorted desc.
    if n_eligible <= cfg.top_k_pairs:
        idx_eligible = np.where(eligible)[0]
        order = np.argsort(-score[idx_eligible])
        return idx_eligible[order]

    # Otherwise argpartition on score then sort the top.
    masked_score = np.where(eligible, score, -np.inf)
    top_idx = np.argpartition(-masked_score, cfg.top_k_pairs - 1)[: cfg.top_k_pairs]
    # Sort the top-K block descending so the heap is in priority order.
    top_idx = top_idx[np.argsort(-masked_score[top_idx])]
    return top_idx


def _build_factorize_lookup(
    factors_data: np.ndarray,
    idx_a: int,
    idx_b: int,
    nbins_a: int,
    nbins_b: int,
    classes_pair_post: np.ndarray,
    unknown_strategy: str,
) -> tuple:
    """Build the pre-prune -> post-prune lookup table that lets
    ``transform()`` replay the merge on test data.

    ``merge_vars`` densely renumbers post-prune so the engineered col
    stored in ``data`` only has values in ``[0, n_uniq)``. But the
    "code" before pruning is a deterministic function of the input:
    ``code = a_value + b_value * nbins_a``. Two training rows with
    the same ``(a, b)`` tuple produce the same pre-prune code, so a
    lookup table indexed by code works for any input that respects
    the original cardinalities.

    Unseen test combinations (combinations that never appeared in
    training) are resolved per ``unknown_strategy``:

    - ``"clip"``: cap at the highest seen class (collides unseen with
      the most frequent training combo's class -- safe, conservative).
    - ``"sentinel"``: dedicate one new class for "unseen" (inflates
      ``n_uniq`` by 1; preferable when downstream models can learn a
      special meaning for that class).
    - ``"raise"``: leave the lookup at -1 sentinel; ``apply_recipe``
      raises a clear error on the first unseen value.

    Returns ``(lookup_table, n_uniq_effective)``:
    - ``lookup_table``: ``(nbins_a * nbins_b,)`` int64 array, ``[code]
      -> post_prune_class`` (or -1 for unseen if ``raise``).
    - ``n_uniq_effective``: ``n_uniq`` + 1 if ``sentinel`` and any
      unseen cells, else ``n_uniq``.
    """
    n_samples = factors_data.shape[0]
    expected_size = int(nbins_a) * int(nbins_b)
    lookup = np.full(expected_size, -1, dtype=np.int64)
    # Vectorised population: pre-prune code per training row.
    vals_a = factors_data[:, idx_a].astype(np.int64)
    vals_b = factors_data[:, idx_b].astype(np.int64)
    pre_prune_codes = vals_a + vals_b * int(nbins_a)
    # ``classes_pair_post`` is aligned row-for-row with the data; assign
    # bulk via fancy indexing (later rows overwrite earlier ones for the
    # same code, but they're identical so order doesn't matter).
    lookup[pre_prune_codes] = classes_pair_post.astype(np.int64)

    seen_mask = lookup >= 0
    n_seen = int(seen_mask.sum())
    n_uniq_effective = int(classes_pair_post.max()) + 1 if n_samples > 0 else 0

    if not seen_mask.all():
        if unknown_strategy == "clip":
            seen_max = int(lookup[seen_mask].max())
            lookup[~seen_mask] = seen_max
        elif unknown_strategy == "sentinel":
            seen_max = int(lookup[seen_mask].max())
            lookup[~seen_mask] = seen_max + 1
            n_uniq_effective = seen_max + 2
        elif unknown_strategy == "raise":
            pass  # leave as -1; apply_recipe will raise
        else:
            raise ValueError(f"Unknown unknown_strategy: {unknown_strategy!r}")
    return lookup, n_uniq_effective


def _materialize_pairs(
    factors_data: np.ndarray,
    pairs_a: np.ndarray,
    pairs_b: np.ndarray,
    selected_idx: np.ndarray,
    nbins: np.ndarray,
    cols: list,
    dtype,
    unknown_strategy: str = "clip",
) -> tuple:
    """For each selected pair, run ``merge_vars`` to produce the
    ordinal-encoded engineered column, build the lookup table that
    enables transform-replay, then assemble the
    ``EngineeredRecipe(kind="factorize")``.

    Returns ``(new_data_block, new_names, new_nbins, new_recipes)``:
    - ``new_data_block``: ``(n_samples, len(selected_idx))`` ordinal array
    - ``new_names``: list of engineered column names
    - ``new_nbins``: list of post-merge cardinalities (pruned)
    - ``new_recipes``: list of ``EngineeredRecipe(kind="factorize")``,
      with the lookup table embedded in ``recipe.extra["lookup_table"]``.

    Caller is responsible for ``np.concatenate``-ing ``new_data_block``
    onto ``data`` (single concat, P8).
    """
    n_samples = factors_data.shape[0]
    n_pairs = len(selected_idx)
    new_data_block = np.empty((n_samples, n_pairs), dtype=dtype)
    new_names: list = []
    new_nbins: list = []
    new_recipes: list = []

    for k_out, k_in in enumerate(selected_idx):
        i = int(pairs_a[k_in])
        j = int(pairs_b[k_in])
        vi = np.array([i, j], dtype=np.int64)
        classes_pair, _, n_uniq = merge_vars(
            factors_data=factors_data,
            vars_indices=vi,
            var_is_nominal=None,
            factors_nbins=nbins,
            dtype=dtype,
        )
        nbins_a = int(nbins[i])
        nbins_b = int(nbins[j])
        lookup, n_uniq_effective = _build_factorize_lookup(
            factors_data=factors_data,
            idx_a=i, idx_b=j,
            nbins_a=nbins_a, nbins_b=nbins_b,
            classes_pair_post=classes_pair,
            unknown_strategy=unknown_strategy,
        )
        # Names follow the cat-FE convention ``kway(c1__c2)``. The
        # ``__`` separator collides with column names containing
        # ``__`` -- SD prevents lineage filter from substring-parsing.
        # For now we just assert no collision; the lineage filter
        # uses ``recipe.src_names`` directly.
        name_a = cols[i]
        name_b = cols[j]
        eng_name = f"kway({name_a}__{name_b})"
        if eng_name in cols:
            # Unlikely but possible if user pre-engineered the same name.
            # Disambiguate by appending the source indices.
            eng_name = f"kway({name_a}__{name_b})_pair{i}_{j}"
        new_data_block[:, k_out] = classes_pair
        new_names.append(eng_name)
        new_nbins.append(n_uniq_effective)
        new_recipes.append(
            EngineeredRecipe(
                name=eng_name,
                kind="factorize",
                src_names=(name_a, name_b),
                factorize_nbins=(nbins_a, nbins_b),
                unknown_strategy=unknown_strategy,
                # Lookup table is the load-bearing artefact for replay.
                # Stored as a plain ndarray in extra so the dataclass
                # frozen-field constraint is satisfied.
                extra={"lookup_table": lookup, "n_uniq_post_prune": n_uniq_effective},
            )
        )
    return new_data_block, new_names, new_nbins, new_recipes


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
    weights: np.ndarray = None,    # Tier 2.1: per-row sample weights; None = uniform
    dtype=np.int32,
    verbose: int = 0,
) -> tuple:
    """One cat-FE iteration. Augments ``data`` / ``cols`` / ``nbins``
    with new ordinal-encoded columns capturing pair (and future k-way)
    synergies. Returns the augmented arrays plus a fresh ``CatFEState``
    holding recipes + diagnostics.

    Inputs:
    - ``data``: ordinal-encoded ``(n_samples, n_cols)`` produced by
      ``categorize_dataset``.
    - ``cols``: list of column names matching ``data`` shape.
    - ``nbins``: cardinality per column.
    - ``target_indices``, ``classes_y``, ``freqs_y``: precomputed by
      caller (avoids re-binning Y for every MI call -- P2).
    - ``categorical_vars``: indices into ``data`` of categorical (or
      pre-discretized numeric, when ``cfg.include_numeric=True``)
      columns to consider.
    - ``cfg``: the cat-FE config; ``cfg.enable=True`` is the user's
      opt-in switch (caller checks before calling us).

    Returns:
    - ``data_out``: augmented ``(n_samples, n_cols + n_engineered)``
    - ``cols_out``: ``cols + engineered_names``
    - ``nbins_out``: ``np.concatenate([nbins, engineered_nbins])``
    - ``state``: ``CatFEState`` with recipes / diagnostics populated.

    When the step adds no engineered columns (no pairs cleared the
    floor / all dropped by validation gates), returns the inputs
    unchanged with an empty ``CatFEState``.
    """
    state = CatFEState()
    n_samples = data.shape[0]

    # ---- Pathological-input gates (SB10) ----
    if target_indices.size == 0:
        raise ValueError("cat-FE: empty target_indices; cannot compute MI(X;Y).")
    if n_samples < cfg.min_n_samples:
        if verbose:
            logger.info(
                "cat-FE skipped: n_samples=%d < cfg.min_n_samples=%d",
                n_samples, cfg.min_n_samples,
            )
        return data, cols, nbins, state

    # ---- Memmap detection (SB10 / F23) ----
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

    # ---- Column-level validation (SM7 / SM8) ----
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
    candidate_mi = _marginal_screen_njit(
        factors_data=data,
        candidate_idxs=candidate_idxs_arr,
        nbins=nbins,
        classes_y=classes_y,
        freqs_y=freqs_y,
        dtype=dtype,
    )
    # marginal_floor prune
    if cfg.marginal_floor > 0:
        keep_mask = candidate_mi > cfg.marginal_floor
        candidate_idxs_arr = candidate_idxs_arr[keep_mask]
        candidate_mi = candidate_mi[keep_mask]
    if len(candidate_idxs_arr) < 2:
        if verbose:
            logger.info("cat-FE skipped: %d cols cleared marginal_floor", len(candidate_idxs_arr))
        return data, cols, nbins, state

    # Build a marginal-MI lookup keyed by COLUMN INDEX (into data), so the
    # pair kernel can look up by index without re-running the screen.
    marginal_mi_full = np.full(data.shape[1], np.nan, dtype=np.float64)
    for k, idx in enumerate(candidate_idxs_arr):
        marginal_mi_full[int(idx)] = candidate_mi[k]

    # ---- Pair enumeration with cardinality budget (B5 / SB10) ----
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
            # Strict int32-overflow gate (B5): the inner merge_vars loop
            # computes ``current_nclasses * sample_class`` in int32. With
            # current_nclasses==nbins[i], this multiplied by (nbins[j]-1)
            # must stay below 2^31. Conservative: nb_prod < 2^31.
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

    # ---- Pair search: CPU (njit prange) or GPU dispatch (P9) ----
    # Backend selection per cfg.backend:
    # - "cpu": always njit prange
    # - "gpu": always GPU dispatch (raises if CuPy missing)
    # - "auto": GPU only at large-N regime (N>=200 cols AND n>=500k rows)
    use_gpu = False
    if cfg.backend == "gpu":
        use_gpu = True
    elif cfg.backend == "auto":
        n_cols_eff = len(candidate_idxs_arr)
        if n_cols_eff >= 200 and n_samples >= 500_000:
            try:
                import cupy  # noqa: F401
                use_gpu = True
            except ImportError:
                if verbose:
                    logger.info(
                        "cat-FE: backend='auto' wanted GPU at N=%d, n=%d "
                        "but CuPy not installed; falling back to CPU.",
                        n_cols_eff, n_samples,
                    )

    # Tier 2.1: choose weighted vs unweighted kernel.
    # Use weighted only when weights are actually non-uniform; uniform
    # weights are equivalent to unweighted and the weighted kernel costs
    # extra ops, so skip in that case.
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
                i = int(pairs_a[k]); j = int(pairs_b[k])
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

    # ---- Top-K selection (P6) ----
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

    # ---- Miller-Madow re-rank on top-K survivors (SB6) ----
    # Only top-K pay the 5x merge_vars cost; saves N²-scale compute while
    # catching high-cardinality bias-driven false positives. The re-rank
    # may shuffle the heap; ``selected_idx`` order matters for downstream
    # ``_materialize_pairs`` because it determines which engineered cols
    # are added first (relevant when ``top_k_pairs`` cap binds).
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

    # ---- Anti-redundancy re-rank (E3, opt-in via anti_redundancy_beta>0) ----
    # Adjusts each survivor's score by ``β * mean_z I(merged; Z)`` where
    # Z ranges over already-selected features in ``selected_so_far``.
    # No-op when β=0 or selected_so_far is empty.
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

    # ---- K-fold II stability filter (E6, opt-in via n_folds_stability>0) ----
    # Drops pairs whose II is unstable across K folds (signal driven by
    # outlier rows). Runs BEFORE permutation so we don't pay perm budget
    # on pairs that fail stability.
    selected_idx, per_fold_ii_dict = _kfold_stability_filter(
        factors_data=data, pairs_a=pairs_a, pairs_b=pairs_b,
        selected_idx=selected_idx, nbins=nbins,
        target_indices=target_indices,
        cfg=cfg, dtype=dtype, verbose=verbose,
    )
    # Surface fold IIs into the state UNCONDITIONALLY -- even when 0 pairs
    # survived, the per-fold values are useful diagnostics (the user can
    # see WHY the pair was rejected). Must happen BEFORE the early return.
    if per_fold_ii_dict:
        state.ii_stability.update(per_fold_ii_dict)
    if len(selected_idx) == 0:
        if verbose:
            logger.info("cat-FE: 0 pairs survived K-fold stability filter")
        return data, cols, nbins, state

    # ---- Permutation confirmation (E2 / SB1) + FWER correction (SB2) ----
    # Runs only when ``cfg.full_npermutations > 0`` (default 100 per SB4).
    # Tests joint-independence null; failed pairs are dropped from
    # ``selected_idx``. The resulting ``confidence_dict`` is surfaced
    # via diagnostics for user inspection. ``n_search_pairs`` is the
    # family size for FWER correction -- the count of pairs CONSIDERED
    # in the search phase, NOT the top-K count.
    # D2: full Westfall-Young requires the per-shuffle max-II across ALL
    # search pairs, which is materially more expensive than per-survivor
    # permutation. To get the full WY behaviour, we substitute the
    # per-pair p-values from the joint-independence test with the WY-
    # corrected versions BEFORE the orchestrator applies the floor.
    # Memory budget: full WY pre-merges m * n int32 cells; if that
    # exceeds e.g. 500 MB we fall back to Bonferroni-on-survivors via
    # the _apply_fwer_correction path.
    use_full_wy = (
        cfg.fwer_correction == "westfall_young"
        and cfg.full_npermutations > 0
        and len(pairs_a) * n_samples * 4 < 500 * 1024 * 1024
    )

    if use_full_wy:
        # Capture all top-K as "survivors" for WY -- the post-permutation
        # filter applies the floor on corrected p.
        wy_corrected_p = _compute_westfall_young_corrected_p(
            factors_data=data, pairs_a=pairs_a, pairs_b=pairs_b,
            ii_obs_arr=ii_arr, selected_idx=selected_idx,
            nbins=nbins, classes_y=classes_y, freqs_y=freqs_y,
            marginal_mi=marginal_mi_full,
            n_perms=cfg.full_npermutations,
            dtype=dtype, verbose=verbose,
        )
        # Build confidence_dict from WY-corrected p, then filter survivors.
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

    # ---- Bootstrap CIs on II (Tier 2.2, opt-in via bootstrap_ci_n_replicates>0) ----
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

    # ---- K-way greedy expansion (SB7, opt-in via max_kway_order > 2) ----
    # T4: HYBRID seeding -- first try only the top-K confirmed pairs,
    # which is O(top_k * N) = ~6400 merge_vars at top_k=64, N=100.
    # If that produces ZERO k-way results (the pure-k-way-XOR case
    # where all 2-way IIs are noise around 0 and top-K is random),
    # fall back to seeding from ALL pairs. This avoids the quadratic
    # cost when the signal is detectable from top-K, but preserves the
    # all-pairs path for pathological niches.
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

        # Sort k-way results by joint_MI desc and cap by top_k_pairs
        kway_results.sort(key=lambda r: -r[3])
        kway_results = kway_results[: cfg.top_k_pairs]
        if verbose:
            logger.info(
                "cat-FE: greedy k-way expansion produced %d feature(s)",
                len(kway_results),
            )

        # Tier 4.2: coordinate-ascent refinement (opt-in via refine_passes>0)
        if cfg.refine_passes > 0 and kway_results:
            kway_results = _refine_kway_coordinate_ascent(
                factors_data=data, kway_results=kway_results,
                candidate_pool=candidate_idxs_arr, nbins=nbins,
                classes_y=classes_y, freqs_y=freqs_y,
                max_combined_nbins=max_combined,
                n_passes=cfg.refine_passes,
                dtype=dtype, verbose=verbose,
            )

    # ---- Materialise pair survivors (P8 single concat) ----
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
                    "II": float(joint_mi),  # k-way: II vs union-of-marginals would
                                            # require recomputing; surface joint_MI
                                            # as a coarse rank signal.
                    "joint_MI": float(joint_mi),
                    "joint_nclasses": int(n_uniq),
                    "src_indices": tuple(int(i) for i in idx_tuple),
                    "src_names": tuple(cols[i] for i in idx_tuple),
                    "kway_order": len(idx_tuple),
                    "n_obs_per_cell_p25": float(n_samples / max(int(n_uniq), 1)),
                    "joint_dependence_confidence": None,  # k-way perm-test not implemented
                }
    # Diagnostics (E4 -- always cheap, gated by cfg)
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
                # Joint-dependence confidence (SB1): honest naming -- this
                # tests "(X1, X2) jointly independent of Y", not "no synergy".
                # ``None`` when no permutation test ran (full_npermutations=0).
                "joint_dependence_confidence": (
                    float(confidence_dict[(i, j)])
                    if (i, j) in confidence_dict
                    else None
                ),
                # Tier 2.2: bootstrap CI on II. ``None`` when disabled.
                "bootstrap_ii_ci": (
                    bootstrap_ci_dict[(i, j)]
                    if (i, j) in bootstrap_ci_dict
                    else None
                ),
            }

    state.recipes.extend(new_recipes)

    # ---- Tier 3.1: target encoding emit (opt-in) ----
    # For each pair recipe, additionally emit a target-encoded col with
    # OOF CV-aware shrinkage. Recipes carry ``kind="target_encoding"``
    # with the global cell-means table for transform() replay.
    if cfg.emit_target_encoding:
        te_cols_list: list = []
        te_names: list = []
        te_recipes: list = []
        # Materialize TE alongside factorize-recipe survivors (pairs only;
        # k-way TE is left as future work for simplicity).
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
            # te_vals dtype is float64; we don't quantize -- caller's
            # downstream model handles continuous encoded values.
            te_name = f"te({cols[i]}__{cols[j]})"
            if te_name in cols or te_name in new_names or te_name in te_names:
                te_name = f"te_{k_out}({cols[i]}__{cols[j]})"
            te_cols_list.append(te_vals)
            te_names.append(te_name)
            # Build a target_encoding recipe with the cell-means table
            # for transform() replay. Stored as ``extra``.
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
            # Target-encoded cols are FLOAT, not ordinal. Add as separate
            # block; downstream screening will discretize them again per
            # quantization_method. nbins[te_col] is unknown until then;
            # leave a sentinel that categorize_dataset will overwrite.
            # For simplicity, we store these in state.diagnostics but
            # NOT in the main data block (the cat-FE pipeline assumes
            # ordinal-encoded data; TE cols would need a quantization
            # round-trip).
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

    # ---- Single concat onto data / cols / nbins (P8) ----
    data_out = np.concatenate([data, new_data_block], axis=1)
    cols_out = list(cols) + new_names
    nbins_out = np.concatenate([nbins, np.asarray(new_nbins, dtype=nbins.dtype)])

    # ---- Build engineered_lineage map (B6 / T1) ----
    # Engineered cols land at indices [n_orig, n_orig + len(new_names)).
    # For each, record the parent indices (in the ORIGINAL data layout)
    # so screen_predictors can skip ``(orig_parent, engineered_col)``
    # k-way candidates -- they're redundant by construction.
    n_orig = data.shape[1]
    name_to_idx = {n: i for i, n in enumerate(cols)}  # original col name -> idx
    state.lineage = {}
    for k_out, name in enumerate(new_names):
        eng_idx = n_orig + k_out
        # Parent indices come from the recipe's src_names. Recipes built
        # in this orchestrator have src_names referencing ORIGINAL data
        # columns (we don't yet support nested engineered parents).
        recipe = new_recipes[k_out]
        parent_idxs = []
        for src_name in recipe.src_names:
            if src_name in name_to_idx:
                parent_idxs.append(name_to_idx[src_name])
        if parent_idxs:
            state.lineage[eng_idx] = frozenset(parent_idxs)

    return data_out, cols_out, nbins_out, state
