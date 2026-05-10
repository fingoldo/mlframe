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
    marginal_mi: np.ndarray,    # (n_cols_in_data,) -- I(X_i; Y) for ALL cols (NaN where not in candidate set)
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    dtype,
) -> tuple:
    """For each pair ``(a, b)`` compute joint MI and Jakulin II.

    Returns ``(joint_mi_arr, ii_arr, n_uniq_arr)`` -- three 1-D arrays
    of length ``len(pairs_a)`` aligned with the input order. The
    classes_pair buffers themselves are NOT returned (they would be
    O(n_samples) per pair, blowing the memory budget at 5000 pairs ×
    1 MB = 5 GB). Top-K survivors get re-merged in
    ``_materialize_pairs`` at materialization time -- one extra
    merge_vars per kept pair, traded against not paying for losers.

    Plug-in entropy throughout -- Miller-Madow correction (SB6) is
    a Phase-2 add-on that re-runs over top-K survivors with the
    correction applied to all six entropies as a unit.
    """
    n_pairs = len(pairs_a)
    joint_mi_out = np.zeros(n_pairs, dtype=np.float64)
    ii_out = np.zeros(n_pairs, dtype=np.float64)
    n_uniq_out = np.zeros(n_pairs, dtype=np.int64)
    for k in prange(n_pairs):
        i = pairs_a[k]
        j = pairs_b[k]
        # Build vars_indices = [i, j] for merge_vars
        vi = np.empty(2, dtype=np.int64)
        vi[0] = i
        vi[1] = j
        classes_pair, freqs_pair, n_uniq = merge_vars(
            factors_data=factors_data,
            vars_indices=vi,
            var_is_nominal=None,
            factors_nbins=nbins,
            dtype=dtype,
        )
        joint_mi = compute_mi_from_classes(
            classes_x=classes_pair,
            freqs_x=freqs_pair,
            classes_y=classes_y,
            freqs_y=freqs_y,
            dtype=dtype,
        )
        joint_mi_out[k] = joint_mi
        ii_out[k] = joint_mi - marginal_mi[i] - marginal_mi[j]
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


def _entropy_for_mode(freqs: np.ndarray, n_samples: int, use_mm: bool) -> float:
    """Plug-in or Miller-Madow entropy on a freq array, gated by ``use_mm``.

    ``freqs`` is normalised (sums to 1.0) per ``merge_vars`` output
    convention. Both ``entropy`` and ``entropy_miller_madow`` filter out
    zero bins internally so we don't need to pre-filter.
    """
    if use_mm:
        return entropy_miller_madow(freqs, n_samples)
    return entropy(freqs)


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

    The MM-corrected scores can shuffle the rank order: a pair where
    plug-in overstated II by 0.04 from cardinality-driven bias may drop
    below another pair's MM-stable score. The reordering is captured in
    the returned ``selected_idx_resorted``.

    When MM is disabled (cfg.use_miller_madow=False), this is a no-op
    that returns inputs unchanged.
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
        per_pair_mm = np.array([
            _should_apply_mm_for_pair(
                int(nbins[pairs_a[k]]), int(nbins[pairs_b[k]]),
                n_y_classes, n_samples,
            )
            for k in selected_idx
        ])
    if auto_gate and not per_pair_mm.any():
        return ii_arr, selected_idx

    # Compute H(Y) once (constant across pairs but doesn't actually affect
    # ranking since it appears in every II equally; included for correctness).
    h_y_pl = entropy(freqs_y)
    h_y_mm = entropy_miller_madow(freqs_y, n_samples)

    if verbose:
        n_corrected = int(per_pair_mm.sum())
        logger.info(
            "cat-FE: re-ranking %d/%d top-K pairs with Miller-Madow correction",
            n_corrected, len(selected_idx),
        )

    ii_mm_arr = ii_arr.copy()
    for j, k in enumerate(selected_idx):
        if not per_pair_mm[j]:
            continue
        ii_mm = _compute_pair_ii_mm(
            factors_data=factors_data,
            idx_a=int(pairs_a[k]), idx_b=int(pairs_b[k]),
            nbins=nbins, target_indices=target_indices,
            classes_y=classes_y, freqs_y=freqs_y,
            h_y=h_y_mm,
            use_mm=True,
            dtype=dtype,
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
        # Approximation: Bonferroni-on-survivors. Documented limitation
        # above; proper WY needs per-shuffle max-II from search phase.
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

        classes_y_safe = classes_y.copy()
        n_failed = 0
        for _ in range(n_perms):
            i_pair_p, i_x1_p, i_x2_p = _shuffle_and_compute_three_mis(
                cls_pair, fq_pair, cls_x1, fq_x1, cls_x2, fq_x2,
                classes_y_safe, freqs_y, dtype,
            )
            ii_perm = i_pair_p - i_x1_p - i_x2_p
            if ii_perm >= ii_obs:
                n_failed += 1
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

    # ---- Pair search prange kernel (P3) ----
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

    # ---- Permutation confirmation (E2 / SB1) + FWER correction (SB2) ----
    # Runs only when ``cfg.full_npermutations > 0`` (default 100 per SB4).
    # Tests joint-independence null; failed pairs are dropped from
    # ``selected_idx``. The resulting ``confidence_dict`` is surfaced
    # via diagnostics for user inspection. ``n_search_pairs`` is the
    # family size for FWER correction -- the count of pairs CONSIDERED
    # in the search phase, NOT the top-K count.
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

    # ---- Materialise survivors (P8 single concat) ----
    new_data_block, new_names, new_nbins, new_recipes = _materialize_pairs(
        factors_data=data,
        pairs_a=pairs_a, pairs_b=pairs_b,
        selected_idx=selected_idx,
        nbins=nbins,
        cols=cols,
        dtype=dtype,
        unknown_strategy=cfg.unknown_strategy,
    )
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
            }

    state.recipes.extend(new_recipes)

    # ---- Single concat onto data / cols / nbins (P8) ----
    data_out = np.concatenate([data, new_data_block], axis=1)
    cols_out = list(cols) + new_names
    nbins_out = np.concatenate([nbins, np.asarray(new_nbins, dtype=nbins.dtype)])

    return data_out, cols_out, nbins_out, state
