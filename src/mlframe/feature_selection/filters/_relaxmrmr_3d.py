"""RelaxMRMR / FJMI — 3-D MI feature selection (Vinh 2016).

Adds a 3-D conditional MI redundancy term ``I(X_k; X_j; X_i | Y)`` that
RELAXES Fleuret's conditional-independence assumption. Vinh, Zhou, Chan,
Bailey 2016 (*Pattern Recognition* 53:51-62) show this catches higher-order
redundancy that pairwise CMIM / JMIM both miss when three or more selected
features jointly explain a candidate.

Interaction information (McGill 1954) is the gap between the conditional and unconditional 3-way co-information:

    II(X; Z_1; Z_2) = I(X; Z_1; Z_2 | Y) - I(X; Z_1; Z_2)

with each co-information expanded into pairwise / joint MIs:

    I(X; Z_1; Z_2 | Y) = I(X; Z_1 | Y) + I(X; Z_2 | Y) - I(X; Z_1, Z_2 | Y)
    I(X; Z_1; Z_2)     = I(X; Z_1)     + I(X; Z_2)     - I(X; Z_1, Z_2)

II > 0 = SYNERGY (the pair (Z_1, Z_2) carries more about X once Y is fixed than it does unconditionally);
II < 0 = REDUNDANCY (the pair already explains X without help from Y). The conditional co-information alone is
NOT a synergy/redundancy signal - subtracting the unconditional co-information is what gives II its sign.

RelaxMRMR score:

    score(X_k) = I(X_k; Y)
                 - (1/|S|) * sum_{j in S} I(X_k; X_j)
                 + (alpha / C(|S|,2)) * sum_{i<j in S} II(X_k; X_i; X_j)

Adding the (signed) interaction term LOWERS the score of jointly-redundant candidates (II < 0) and RAISES
synergistic ones (II > 0). Default alpha = 1 matches Vinh 2016; higher alpha emphasises higher-order structure.

Reference: Vinh, N.X., Zhou, J., Chan, J., Bailey, J. (2016), "Can
high-order dependencies improve mutual information based feature
selection?", *Pattern Recognition* 53:51-62.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# The estimators moved to a sibling so the parallel pair loop shares them; re-exported here because callers and tests import them
# from this module.
from mlframe.feature_selection.filters._relaxmrmr_kernels import (  # noqa: F401
    _cmi_mm_njit,
    _composite_codes_njit,
    _mi_mm_njit,
)
from mlframe.feature_selection.filters._relaxmrmr_pair_loop import pair_interaction_sum

# Miller-Madow-corrected estimators for the interaction term. Plug-in MI is biased upward by roughly (occupied cells - 1) / 2n per entropy,
# and the interaction term differences MIs estimated on tables of very different sizes (the composite pair's table is K_i*K_j times wider),
# so without a correction the bias does not cancel and every candidate gets a spurious "synergy" reward. Each entropy gets its own
# (m - 1) / 2n term, applied once, and the results are NOT clamped at zero: a clamp is non-linear and would reintroduce the bias.


def relax_mrmr_score(
    x_cand: np.ndarray,
    selected_cols: list[np.ndarray],
    y: np.ndarray,
    nbins_x: int,
    nbins_selected: list[int],
    nbins_y: int,
    alpha: float = 1.0,
    min_rows_per_cell: float = 5.0,
    selected_prechecked: bool = False,
) -> float:
    """RelaxMRMR / FJMI 3-D-MI score for one candidate (Vinh 2016).

    Args:
        x_cand: 1-D integer-encoded candidate column.
        selected_cols: 1-D integer-encoded columns already selected (the conditioning set).
        y: 1-D integer-encoded target.
        nbins_x: cardinality of ``x_cand``.
        nbins_selected: cardinalities of ``selected_cols``, same order/length.
        nbins_y: cardinality of ``y``.
        alpha: weight on the 3-way interaction term (default 1.0 per Vinh 2016).
        min_rows_per_cell: a selected pair contributes to the interaction term only when its largest table,
            ``K_x * K_i * K_j * K_y`` cells for ``I(X; Z_i, Z_j | Y)``, holds at least this many rows per cell. Below that the
            estimate is dominated by sampling bias no plug-in correction removes (measured: -0.37 "interaction" on fully
            independent data at n=2000 with 10-level columns), so the pair's term is undefined and left out.
        selected_prechecked: the caller already range-checked ``y`` and ``selected_cols`` (via
            ``assert_relax_inputs_in_range``), so the per-candidate call skips re-reading columns that do not change.

    Returns: scalar score with full 3-way correction; higher = better.

    Cost: ``O(|S|^2)`` 3-D plug-in MIs per candidate; on large selected
    sets enable the dispatcher only after the per-screen filter has pruned
    the pool.
    """
    if alpha < 0.0:
        # The interaction term is already signed (synergy rewarded, redundancy penalised); a negative alpha used to be silently
        # treated as 0 (the term skipped), indistinguishable from alpha=0.
        raise ValueError(f"relax_mrmr_score: alpha must be >= 0; got {alpha!r}.")
    from ._bur_term import _mi_pair_njit  # reuse the 2-var plug-in MI kernel

    # Guard against out-of-range / -1-sentinel codes: the njit kernels index joint[x[i], y[i], z[i]] directly, so a
    # negative sentinel wraps to the last bin and an over-range code writes out of bounds (silent corruption). PID
    # hardens the same class explicitly; mirror it here.
    from ._fe_batched_mi import _assert_codes_in_range

    _assert_codes_in_range(x_cand, int(nbins_x), "relax_mrmr_score x_cand")
    if not selected_prechecked:
        # The target and the selected set are fixed for a whole greedy round, so scanning them per candidate re-reads the same |S|+1 columns
        # for every candidate. A caller that hoists them (see ``assert_relax_inputs_in_range``) checks once and says so.
        _assert_codes_in_range(y, int(nbins_y), "relax_mrmr_score y")
        for _j, _c in enumerate(selected_cols):
            _assert_codes_in_range(_c, int(nbins_selected[_j]), "relax_mrmr_score selected_col")
    # asarray, not astype: the caller's hoist already materialises int64 codes, and astype copies unconditionally, so this was |S|+2
    # full-length copies per candidate for no change of dtype.
    x_int = np.asarray(x_cand, dtype=np.int64)
    y_int = np.asarray(y, dtype=np.int64)
    K_x = int(nbins_x)
    K_y = int(nbins_y)
    n_S = len(selected_cols)
    K_sel = [int(k) for k in nbins_selected]
    # Guard BEFORE the first dense allocation (including the relevance term's own (K_x, K_y) joint and the
    # n_S == 0 early return, both of which would otherwise allocate un-capped). The 3-D I(X;Y|Z) joint is
    # (K_x, K_y, K_z) and the interaction term's composite pair joint is (K_x, K_z_i, K_z_j); guard the
    # largest of each so a high-cardinality selected set cannot OOM the dense alloc.
    from mlframe.feature_selection.filters.info_theory.shared import check_joint_cardinality

    check_joint_cardinality(K_x, K_y, what="relax_mrmr_score")
    for _K_z in K_sel:
        check_joint_cardinality(K_x, K_y, _K_z, what="relax_mrmr_score")
    if n_S > 1:
        _k_sel_max = max(K_sel)
        check_joint_cardinality(K_x, _k_sel_max, _k_sel_max, what="relax_mrmr_score")
    # Relevance I(X; Y).
    relevance = _mi_pair_njit(x_int, y_int, K_x, K_y)
    if n_S == 0:
        return float(relevance)
    # Pairwise redundancy (1/|S|) sum_j I(X; X_j): marginal MI between the candidate and each already-selected feature.
    # A candidate that duplicates a selected feature gets a large penalty; an independent one gets ~0.
    sel_int = [np.asarray(col, dtype=np.int64) for col in selected_cols]
    pair_red = 0.0
    for j in range(n_S):
        pair_red += _mi_pair_njit(x_int, sel_int[j], K_x, K_sel[j])
    pair_red /= float(n_S)
    # 3-way interaction-information correction: alpha / C(|S|,2) * sum_{i<j} II(X; Z_i; Z_j),
    # where II = I(X; Z_i; Z_j | Y) - I(X; Z_i; Z_j) and each co-information is decomposed as
    # CO_cond  = I(X; Z_i | Y) + I(X; Z_j | Y) - I(X; Z_i, Z_j | Y),
    # CO_uncond= I(X; Z_i)     + I(X; Z_j)     - I(X; Z_i, Z_j).
    # II > 0 means the pair (Z_i, Z_j) carries MORE about X once Y is fixed than unconditionally (synergy) -> reward;
    # II < 0 means the joint already explains X without Y (redundancy) -> penalty. Adding alpha*II therefore lowers the
    # score of jointly-redundant candidates and raises synergistic ones, the direction RelaxMRMR (Vinh 2016) intends.
    # The four MIs of each pair are estimated on tables of very different sizes, so they go through the Miller-Madow-corrected estimators
    # (see ``_mi_mm_njit``); the plug-in values above stay as they are for the relevance and redundancy terms.
    inter = 0.0
    if n_S >= 2 and alpha > 0.0:
        norm = float(n_S * (n_S - 1)) / 2.0
        marg_mm = np.empty(n_S, dtype=np.float64)
        cmi_y_mm = np.empty(n_S, dtype=np.float64)
        for j in range(n_S):
            marg_mm[j] = _mi_mm_njit(x_int, sel_int[j], K_x, K_sel[j])
            cmi_y_mm[j] = _cmi_mm_njit(x_int, sel_int[j], y_int, K_x, K_sel[j], K_y)
        # The pairs are independent, so the whole O(|S|^2) loop is one parallel kernel: no per-pair interpreter round trip, and each thread
        # rewrites one composite buffer instead of allocating an n-length array per pair.
        inter = pair_interaction_sum(x_int, y_int, sel_int, K_sel, K_x, K_y, cmi_y_mm, marg_mm, min_rows_per_cell)
        inter *= float(alpha) / norm
    return float(relevance - pair_red + inter)


def assert_relax_inputs_in_range(y: np.ndarray, nbins_y: int, selected_cols: Any, nbins_selected: Any) -> None:
    """Range-check the target and the selected set once, so the per-candidate score can skip re-reading columns that do not change.

    The kernels index their joint tables directly, so a negative sentinel wraps to the last bin and an over-range code writes out of bounds.
    The check is the same one the score runs; it is only moved to where the columns are materialised.
    """
    from ._fe_batched_mi import _assert_codes_in_range

    _assert_codes_in_range(y, int(nbins_y), "relax_mrmr_score y")
    for idx, col in enumerate(selected_cols):
        _assert_codes_in_range(col, int(nbins_selected[idx]), "relax_mrmr_score selected_col")

__all__ = ["assert_relax_inputs_in_range", "relax_mrmr_score"]
