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

import math

import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def _cmi_xy_given_z_njit(x: np.ndarray, y: np.ndarray, z: np.ndarray, K_x: int, K_y: int, K_z: int) -> float:
    """Plug-in I(X; Y | Z) on integer-bin inputs."""
    n = x.shape[0]
    if n <= 0:
        return 0.0
    # Joint (X, Y, Z) frequencies.
    joint = np.zeros((K_x, K_y, K_z), dtype=np.float64)
    for i in range(n):
        joint[x[i], y[i], z[i]] += 1.0
    n_f = float(n)
    # Marginalise.
    Pz = np.zeros(K_z, dtype=np.float64)
    Pxz = np.zeros((K_x, K_z), dtype=np.float64)
    Pyz = np.zeros((K_y, K_z), dtype=np.float64)
    for i in range(K_x):
        for j in range(K_y):
            for k in range(K_z):
                v = joint[i, j, k]
                Pz[k] += v
                Pxz[i, k] += v
                Pyz[j, k] += v
    cmi = 0.0
    for i in range(K_x):
        for j in range(K_y):
            for k in range(K_z):
                v = joint[i, j, k]
                if v <= 0.0 or Pxz[i, k] <= 0.0 or Pyz[j, k] <= 0.0 or Pz[k] <= 0.0:
                    continue
                p_xyz = v / n_f
                p_z = Pz[k] / n_f
                p_xz = Pxz[i, k] / n_f
                p_yz = Pyz[j, k] / n_f
                cmi += p_xyz * math.log((p_xyz * p_z) / (p_xz * p_yz))
    return max(0.0, cmi)


@njit(nogil=True, cache=True)
def _joint_mi_x_zw_given_y_njit(x: np.ndarray, z1: np.ndarray, z2: np.ndarray, y: np.ndarray, K_x: int, K_z1: int, K_z2: int, K_y: int) -> float:
    """I(X; (Z_1, Z_2) | Y) via plug-in on composite (Z_1, Z_2).

    The composite pair is the SECOND variable of the mutual information, with ``Y`` as the conditioning
    variable -- the orientation the co-information decomposition in :func:`relaxmrmr_3d_score` needs. The
    previous helper here built the same composite but placed it in the CONDITIONING slot, yielding
    ``I(X; Y | Z_1, Z_2)``: a different quantity that cannot be substituted for this one.
    """
    n = x.shape[0]
    if n <= 0:
        return 0.0
    K_zz = K_z1 * K_z2
    z_comp = np.empty(n, dtype=np.int64)
    for i in range(n):
        z_comp[i] = int(z1[i]) * K_z2 + int(z2[i])
    return float(_cmi_xy_given_z_njit(x, z_comp, y, K_x, K_zz, K_y))


@njit(nogil=True, cache=True)
def _mi_x_pair_njit(x: np.ndarray, z1: np.ndarray, z2: np.ndarray, K_x: int, K_z1: int, K_z2: int) -> float:
    """Unconditional I(X; Z_1, Z_2) via plug-in on the composite (Z_1, Z_2)."""
    n = x.shape[0]
    if n <= 0:
        return 0.0
    K_zz = K_z1 * K_z2
    joint = np.zeros((K_x, K_zz), dtype=np.float64)
    for i in range(n):
        joint[int(x[i]), int(z1[i]) * K_z2 + int(z2[i])] += 1.0
    n_f = float(n)
    Px = np.zeros(K_x, dtype=np.float64)
    Pz = np.zeros(K_zz, dtype=np.float64)
    for i in range(K_x):
        for j in range(K_zz):
            v = joint[i, j]
            Px[i] += v
            Pz[j] += v
    mi = 0.0
    for i in range(K_x):
        if Px[i] <= 0.0:
            continue
        for j in range(K_zz):
            v = joint[i, j]
            if v <= 0.0 or Pz[j] <= 0.0:
                continue
            p = v / n_f
            mi += p * math.log(p * n_f / (Px[i] * Pz[j] / n_f))
    return max(0.0, mi)


# Miller-Madow-corrected estimators for the interaction term. Plug-in MI is biased upward by roughly (occupied cells - 1) / 2n per entropy,
# and the interaction term differences MIs estimated on tables of very different sizes (the composite pair's table is K_i*K_j times wider),
# so without a correction the bias does not cancel and every candidate gets a spurious "synergy" reward. Each entropy gets its own
# (m - 1) / 2n term, applied once, and the results are NOT clamped at zero: a clamp is non-linear and would reintroduce the bias.
@njit(nogil=True, cache=True)
def _mi_mm_njit(a: np.ndarray, b: np.ndarray, K_a: int, K_b: int) -> float:
    """Miller-Madow-corrected plug-in I(A; B) on integer codes, unclamped."""
    n = a.shape[0]
    if n <= 0:
        return 0.0
    joint = np.zeros((K_a, K_b), dtype=np.float64)
    for i in range(n):
        joint[a[i], b[i]] += 1.0
    Pa = joint.sum(axis=1)
    Pb = joint.sum(axis=0)
    n_f = float(n)
    mi = 0.0
    m_ab = 0
    for i in range(K_a):
        for j in range(K_b):
            v = joint[i, j]
            if v > 0.0:
                m_ab += 1
                mi += (v / n_f) * math.log(v * n_f / (Pa[i] * Pb[j]))
    m_a = 0
    for i in range(K_a):
        if Pa[i] > 0.0:
            m_a += 1
    m_b = 0
    for j in range(K_b):
        if Pb[j] > 0.0:
            m_b += 1
    return mi - (m_ab - m_a - m_b + 1) / (2.0 * n_f)


@njit(nogil=True, cache=True)
def _cmi_mm_njit(x: np.ndarray, y: np.ndarray, z: np.ndarray, K_x: int, K_y: int, K_z: int) -> float:
    """Miller-Madow-corrected plug-in I(X; Y | Z) on integer codes, unclamped."""
    n = x.shape[0]
    if n <= 0:
        return 0.0
    joint = np.zeros((K_x, K_y, K_z), dtype=np.float64)
    for i in range(n):
        joint[x[i], y[i], z[i]] += 1.0
    Pz = np.zeros(K_z, dtype=np.float64)
    Pxz = np.zeros((K_x, K_z), dtype=np.float64)
    Pyz = np.zeros((K_y, K_z), dtype=np.float64)
    m_xyz = 0
    for i in range(K_x):
        for j in range(K_y):
            for k in range(K_z):
                v = joint[i, j, k]
                if v > 0.0:
                    m_xyz += 1
                Pz[k] += v
                Pxz[i, k] += v
                Pyz[j, k] += v
    n_f = float(n)
    cmi = 0.0
    for i in range(K_x):
        for j in range(K_y):
            for k in range(K_z):
                v = joint[i, j, k]
                if v > 0.0:
                    cmi += (v / n_f) * math.log((v * Pz[k]) / (Pxz[i, k] * Pyz[j, k]))
    m_z = 0
    for k in range(K_z):
        if Pz[k] > 0.0:
            m_z += 1
    m_xz = 0
    for i in range(K_x):
        for k in range(K_z):
            if Pxz[i, k] > 0.0:
                m_xz += 1
    m_yz = 0
    for j in range(K_y):
        for k in range(K_z):
            if Pyz[j, k] > 0.0:
                m_yz += 1
    return cmi - (m_xyz - m_xz - m_yz + m_z) / (2.0 * n_f)


@njit(nogil=True, cache=True)
def _composite_codes_njit(z1: np.ndarray, z2: np.ndarray, K_z2: int) -> np.ndarray:
    """Integer code of the pair (Z_1, Z_2)."""
    n = z1.shape[0]
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        out[i] = int(z1[i]) * K_z2 + int(z2[i])
    return out


def relax_mrmr_score(
    x_cand: np.ndarray,
    selected_cols: list[np.ndarray],
    y: np.ndarray,
    nbins_x: int,
    nbins_selected: list[int],
    nbins_y: int,
    alpha: float = 1.0,
    min_rows_per_cell: float = 5.0,
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

    Returns: scalar score with full 3-way correction; higher = better.

    Cost: ``O(|S|^2)`` 3-D plug-in MIs per candidate; on large selected
    sets enable the dispatcher only after the per-screen filter has pruned
    the pool.
    """
    from ._bur_term import _mi_pair_njit  # reuse the 2-var plug-in MI kernel
    # Guard against out-of-range / -1-sentinel codes: the njit kernels index joint[x[i], y[i], z[i]] directly, so a
    # negative sentinel wraps to the last bin and an over-range code writes out of bounds (silent corruption). PID
    # hardens the same class explicitly; mirror it here.
    from ._fe_batched_mi import _assert_codes_in_range

    _assert_codes_in_range(x_cand, int(nbins_x), "relax_mrmr_score x_cand")
    _assert_codes_in_range(y, int(nbins_y), "relax_mrmr_score y")
    for _j, _c in enumerate(selected_cols):
        _assert_codes_in_range(_c, int(nbins_selected[_j]), "relax_mrmr_score selected_col")
    x_int = x_cand.astype(np.int64)
    y_int = y.astype(np.int64)
    K_x = int(nbins_x)
    K_y = int(nbins_y)
    n_S = len(selected_cols)
    K_sel = [int(k) for k in nbins_selected]
    # Guard BEFORE the first dense allocation (including the relevance term's own (K_x, K_y) joint and the
    # n_S == 0 early return, both of which would otherwise allocate un-capped). The 3-D I(X;Y|Z) joint is
    # (K_x, K_y, K_z) and the interaction term's composite pair joint is (K_x, K_z_i, K_z_j); guard the
    # largest of each so a high-cardinality selected set cannot OOM the dense alloc.
    from .info_theory._batch_kernels import check_joint_cardinality

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
    sel_int = [col.astype(np.int64) for col in selected_cols]
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
        n_rows = float(x_int.shape[0])
        for i in range(n_S):
            for j in range(i + 1, n_S):
                K_i = K_sel[i]
                K_j = K_sel[j]
                if n_rows < float(min_rows_per_cell) * K_x * K_i * K_j * K_y:
                    continue  # undersampled composite table: this pair's interaction term is not estimable
                z_pair = _composite_codes_njit(sel_int[i], sel_int[j], K_j)
                cmi_ij = _cmi_mm_njit(x_int, z_pair, y_int, K_x, K_i * K_j, K_y)
                co_cond = cmi_y_mm[i] + cmi_y_mm[j] - cmi_ij
                mi_x_zz = _mi_mm_njit(x_int, z_pair, K_x, K_i * K_j)
                co_uncond = marg_mm[i] + marg_mm[j] - mi_x_zz
                inter += co_cond - co_uncond
        inter *= float(alpha) / norm
    return float(relevance - pair_red + inter)


__all__ = ["relax_mrmr_score"]
