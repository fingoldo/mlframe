"""Miller-Madow bias-correction for pair-level Interaction Information.

Split out of ``cat_interactions.py`` to keep the parent below the 1k-line
monolith threshold. Behaviour preserved bit-for-bit; the parent re-exports
the public-looking entries so the orchestrator in
``run_cat_interaction_step`` continues to call them via the same names.

What lives here:
  - ``_entropy_for_mode`` (per-pair entropy with MM toggle)
  - ``_should_apply_mm_for_pair_analytical`` / ``_should_apply_mm_for_pair``
    (analytic + heuristic guards that decide when to flip MM on)
  - ``_compute_pair_ii_mm`` (the MM-corrected pair-II formula)
  - ``_maybe_rerank_with_mm`` (apply MM correction + re-rank a pair list)
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from .cat_fe_state import CatFEConfig
from .info_theory import entropy, entropy_miller_madow, merge_vars, weighted_class_freqs


def _freqs_for_weights(freqs: np.ndarray, classes: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    """Re-weight ``merge_vars``'s dense pruned ``freqs`` by per-row ``weights`` when given, else pass through unchanged.

    mrmr_audit_2026-07-20 B-19: binning (``merge_vars``) is weight-independent, so any already-computed
    ``classes`` array can be re-weighted post-hoc without re-running the merge -- the single primitive
    threaded through every cat-FE downstream confirmation/rerank step below.
    """
    if weights is None:
        return freqs
    return np.asarray(weighted_class_freqs(classes, weights, len(freqs)), dtype=np.float64)

logger = logging.getLogger(__name__)


# ============================================================================
# Miller-Madow II re-score
#
# The Jakulin II expansion involves SIX entropies with mixed signs:
#
#   II = H(X1,X2) + H(X1,Y) + H(X2,Y) - H(X1,X2,Y) - H(X1) - H(X2) - H(Y)
#                       (last term cancels across pairs since H(Y) is fit-wide)
#
# Plug-in entropy is biased downward by ``(k-1)/(2n)`` per entropy term, where k is the number of NON-EMPTY bins. Under the independence null, the SIGNED sum
# of these biases reduces to ``-(a-1)(b-1)(c-1)/(2n)`` -- i.e. plug-in II is biased UPWARD; Miller-Madow correction pulls it back down.
#
# Cost: 5+ ``merge_vars`` calls per pair (X1, X2, X1Y, X2Y, X1X2Y) vs the plug-in path's 1 call. To stay fast on the search loop, we apply MM ONLY to the top-K
# survivors as a re-rank step. This catches the high-cardinality false positives where bias dominates without paying the cost on every pair.
#
# References: Paninski 2003 В§4, Treves & Panzeri 1995, Roulston 1999.
# ============================================================================


def _entropy_for_mode(
    freqs: np.ndarray, n_samples: int, use_mm: bool, use_kt: bool = False,
) -> float:
    """Plug-in / Miller-Madow / Krichevsky-Trofimov entropy.

    KT smoothing adds Dirichlet(0.5) pseudocounts before plug-in entropy; less biased than plug-in for high-cardinality joints and provably asymptotically efficient
    (Krichevsky & Trofimov 1981).
    """
    if use_kt:
        # Reconstruct counts, add 0.5 to each cell, renormalize, then plug-in entropy. ``freqs`` is normalised so counts = freqs * n.
        counts = freqs * n_samples + 0.5
        K = len(counts)
        total = float(n_samples) + 0.5 * K
        probs = counts / total
        return float(entropy(probs))
    if use_mm:
        return float(entropy_miller_madow(freqs, n_samples))
    return float(entropy(freqs))


def _should_apply_mm_for_pair_analytical(
    nbins_a: int, nbins_b: int, n_y_classes: int, n_samples: int,
) -> bool:
    """Analytically-derived MM auto-gate.

    Per Paninski 2003, plug-in entropy bias is ``(K-1) / (2n)``. The signed sum across the 6 entropies of the II expansion under the independence null equals
    ``-(a-1)(b-1)(c-1) / (2n)``. We apply MM when this bias is comparable to typical synergy floors (``-3/sqrt(n)``): activate MM when
    ``(a-1)(b-1)(c-1)/(2n) >= 3/sqrt(n)``, i.e. ``(a-1)(b-1)(c-1) >= 6 * sqrt(n)``. Replaces the folklore threshold ``(a*b*c)/n > 0.05``.
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
    h_y: float,  # H(Y), precomputed once at orchestrator level
    use_mm: bool,
    dtype,
) -> float:
    """Compute Jakulin II for a single pair with optional Miller-Madow correction applied uniformly to all six entropies.

    Returns the II value. Cost: 5 ``merge_vars`` calls (X1, X2, X1+Y, X2+Y, X1+X2+Y) per call -- much heavier than the search-loop's plug-in MI, so callers MUST gate by top-K.
    """
    n_samples = factors_data.shape[0]

    # All six entropies are computed PLUG-IN here; the Miller-Madow correction is applied ONCE as a single
    # closed-form II bias term below, NOT per-entropy. Per-term MM (the pre-fix path) added ``(k_i-1)/(2n)`` with
    # each term's OWN occupied bin count k_i; those six occupied counts do NOT telescope to the analytic II bias
    # ``(a-1)(b-1)(c-1)/(2n)`` (that closed form requires the joint occupied counts to factor as products of the
    # marginals, which sparse / heavy-tailed joints violate). The leftover non-telescoping residual could be large
    # enough to FLIP the sign of a small II. Computing plug-in II and subtracting ONE consistent correction keeps
    # the six terms commensurable so the bias cancels exactly as the Jakulin/Roulston derivation intends.

    # ---- merge X1 alone (gives H(X1)) ----
    _cls_x1, freqs_x1, _ = merge_vars(
        factors_data=factors_data,
        vars_indices=np.array([idx_a], dtype=np.int64),
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    h_x1 = _entropy_for_mode(freqs_x1, n_samples, False)

    # ---- merge X2 alone (gives H(X2)) ----
    _cls_x2, freqs_x2, _ = merge_vars(
        factors_data=factors_data,
        vars_indices=np.array([idx_b], dtype=np.int64),
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    h_x2 = _entropy_for_mode(freqs_x2, n_samples, False)

    # ---- merge X1, X2 (gives H(X1, X2)) ----
    _cls_x1x2, freqs_x1x2, _ = merge_vars(
        factors_data=factors_data,
        vars_indices=np.array([idx_a, idx_b], dtype=np.int64),
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    h_x1x2 = _entropy_for_mode(freqs_x1x2, n_samples, False)

    # ---- merge X1, Y_idx ... (gives H(X1, Y)) ----
    # Concatenate idx_a with target_indices into a single vars list.
    vi_x1y = np.concatenate(([idx_a], target_indices)).astype(np.int64)
    _, freqs_x1y, _ = merge_vars(
        factors_data=factors_data,
        vars_indices=vi_x1y,
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    h_x1y = _entropy_for_mode(freqs_x1y, n_samples, False)

    # ---- merge X2, Y (gives H(X2, Y)) ----
    vi_x2y = np.concatenate(([idx_b], target_indices)).astype(np.int64)
    _, freqs_x2y, _ = merge_vars(
        factors_data=factors_data,
        vars_indices=vi_x2y,
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    h_x2y = _entropy_for_mode(freqs_x2y, n_samples, False)

    # ---- merge X1, X2, Y (gives H(X1, X2, Y)) ----
    vi_x1x2y = np.concatenate(([idx_a, idx_b], target_indices)).astype(np.int64)
    _, freqs_x1x2y, _ = merge_vars(
        factors_data=factors_data,
        vars_indices=vi_x1x2y,
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    h_x1x2y = _entropy_for_mode(freqs_x1x2y, n_samples, False)

    # II = H(X1,X2) + H(X1,Y) + H(X2,Y) - H(X1,X2,Y) - H(X1) - H(X2) - H(Y)
    ii_plugin = h_x1x2 + h_x1y + h_x2y - h_x1x2y - h_x1 - h_x2 - h_y
    if not use_mm:
        return ii_plugin
    # Single telescoped Miller-Madow II bias on OCCUPIED marginal cardinalities (matches the occupied-k convention
    # of info_theory.entropy_miller_madow / mi_miller_madow_correct). Plug-in II is biased UPWARD by
    # ``(a-1)(b-1)(c-1)/(2n)`` under the independence null; subtract it once.
    k_a = len(freqs_x1[freqs_x1 > 0])
    k_b = len(freqs_x2[freqs_x2 > 0])
    k_y = len(freqs_y[freqs_y > 0])
    if k_a <= 1 or k_b <= 1 or k_y <= 1:
        return ii_plugin
    ii_bias = (k_a - 1) * (k_b - 1) * (k_y - 1) / (2.0 * n_samples)
    return float(ii_plugin - ii_bias)


def _should_apply_mm_for_pair(
    nbins_a: int, nbins_b: int, n_y_classes: int, n_samples: int,
    threshold: float = 0.05,
) -> bool:
    """Auto-gate MM application: fire when joint cardinality / n > threshold.

    Per Paninski 2003, plug-in entropy bias scales with (k-1)/(2n). When k/n exceeds the threshold the bias is large enough to materially shift II by more than typical
    synergy floors (-3/sqrt(n)). Below the threshold the bias is in the noise and applying MM only adds compute cost.

    ``threshold=0.05`` is folklore, not derived. User can override via ``cfg.use_miller_madow=True`` to force, or ``False`` to disable.
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
    single_merge_cache_out: dict | None = None,
    weights: Optional[np.ndarray] = None,
) -> tuple:
    """If MM is enabled (cfg flag True or auto-gate fires for at least one survivor), recompute II for selected pairs with MM correction applied to all six entropies.
    Returns ``(ii_mm_arr, selected_idx_resorted)``. Hoists constant / per-column entropies (H(Y), H(X_i), H(X_i, Y)) out of the per-pair loop -- cached entropies cut MM
    cost from 5 merge_vars + 6 entropy per pair down to 2 merge_vars + 2 entropy per pair (joint terms only).

    ``single_merge_cache_out``, when given, is populated with ``{col_idx: (cls, freqs)}`` for every single-column ``merge_vars`` this call performs -- the
    SAME single-column merge ``_cat_confirm_permutation._confirm_pairs_via_permutation`` independently re-derives for the SAME survivor set right after this
    MM re-rank runs (both scan the same top-K pairs on the same factors_data/nbins/dtype). Exposed here as a caller-supplied side-channel; the two functions
    aren't wired together yet (that requires threading a shared dict through ``_cat_interactions_step.py``'s orchestrator, out of this module's scope).
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
        # Analytical MM threshold based on Paninski bias formula
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

    # KT smoothing alternative to MM (set via cfg.use_kt_smoothing). KT smooths each entropy independently (a
    # different bias model), so under KT the per-term path is used as-is; under MM the six terms are computed
    # PLUG-IN and a single telescoped occupied-k bias is subtracted per pair (see _compute_pair_ii_mm).
    use_kt = bool(getattr(cfg, "use_kt_smoothing", False))
    freqs_y_w = _freqs_for_weights(freqs_y, classes_y, weights)
    h_y_mm = _entropy_for_mode(freqs_y_w, n_samples, use_mm=not use_kt, use_kt=use_kt)
    k_y_occ = len(freqs_y_w[freqs_y_w > 0])

    # Hoist H(X_i) and H(X_i, Y) caches outside the loop. Only the columns touched by surviving pairs need to be computed.
    touched_cols: set = set()
    for j, k in enumerate(selected_idx):
        if per_pair_mm[j]:
            touched_cols.add(int(pairs_a[k]))
            touched_cols.add(int(pairs_b[k]))

    h_marginal_cache: dict = {}  # idx -> H(X_idx)
    h_marginal_y_cache: dict = {}  # idx -> H(X_idx, Y)
    k_marginal_cache: dict = {}  # idx -> occupied bin count of X_idx (for the telescoped II bias)
    for col_idx in touched_cols:
        _cls_x, freqs_x, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=np.array([col_idx], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        if single_merge_cache_out is not None:
            single_merge_cache_out[col_idx] = (_cls_x, freqs_x)
        freqs_x_w = _freqs_for_weights(freqs_x, _cls_x, weights)
        h_marginal_cache[col_idx] = _entropy_for_mode(
            freqs_x_w, n_samples, use_mm=not use_kt, use_kt=use_kt,
        )
        k_marginal_cache[col_idx] = len(freqs_x_w[freqs_x_w > 0])
        vi_xy = np.concatenate(([col_idx], target_indices)).astype(np.int64)
        _cls_xy, freqs_xy, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=vi_xy,
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        freqs_xy_w = _freqs_for_weights(freqs_xy, _cls_xy, weights)
        h_marginal_y_cache[col_idx] = _entropy_for_mode(
            freqs_xy_w, n_samples, use_mm=not use_kt, use_kt=use_kt,
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
        _cls_pair, freqs_pair, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=np.array([idx_a, idx_b], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        freqs_pair_w = _freqs_for_weights(freqs_pair, _cls_pair, weights)
        h_x1x2 = _entropy_for_mode(freqs_pair_w, n_samples, use_mm=not use_kt, use_kt=use_kt)
        vi_pair_y = np.concatenate(([idx_a, idx_b], target_indices)).astype(np.int64)
        _cls_pair_y, freqs_pair_y, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=vi_pair_y,
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        freqs_pair_y_w = _freqs_for_weights(freqs_pair_y, _cls_pair_y, weights)
        h_x1x2y = _entropy_for_mode(freqs_pair_y_w, n_samples, use_mm=not use_kt, use_kt=use_kt)
        # II = H(X1,X2) + H(X1,Y) + H(X2,Y) - H(X1,X2,Y) - H(X1) - H(X2) - H(Y)
        ii_mm = h_x1x2 + h_marginal_y_cache[idx_a] + h_marginal_y_cache[idx_b] - h_x1x2y - h_marginal_cache[idx_a] - h_marginal_cache[idx_b] - h_y_mm
        if not use_kt:
            # Single telescoped MM II bias on occupied marginal cardinalities (the six terms above are now all
            # plug-in under MM mode, so subtract the closed-form bias once instead of per-entropy -- the per-term
            # occupied-k corrections did NOT telescope and could flip the sign of a small II).
            k_a = k_marginal_cache[idx_a]
            k_b = k_marginal_cache[idx_b]
            if k_a > 1 and k_b > 1 and k_y_occ > 1:
                ii_mm -= (k_a - 1) * (k_b - 1) * (k_y_occ - 1) / (2.0 * n_samples)
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
