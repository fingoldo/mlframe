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
NOT a synergy/redundancy signal -- subtracting the unconditional co-information is what gives II its sign.

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
def _joint_cmi_xy_given_zw_njit(x: np.ndarray, y: np.ndarray, z1: np.ndarray, z2: np.ndarray, K_x: int, K_y: int, K_z1: int, K_z2: int) -> float:
    """I(X; Y | Z_1, Z_2) via plug-in on composite (Z_1, Z_2). Treats the
    pair (Z_1, Z_2) as a single conditioning variable of size K_z1*K_z2."""
    n = x.shape[0]
    if n <= 0:
        return 0.0
    K_zz = K_z1 * K_z2
    z_comp = np.empty(n, dtype=np.int64)
    for i in range(n):
        z_comp[i] = int(z1[i]) * K_z2 + int(z2[i])
    return _cmi_xy_given_z_njit(x, y, z_comp, K_x, K_y, K_zz)


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


def relax_mrmr_score(
    x_cand: np.ndarray, selected_cols: list[np.ndarray], y: np.ndarray, nbins_x: int, nbins_selected: list[int], nbins_y: int, alpha: float = 1.0
) -> float:
    """RelaxMRMR / FJMI 3-D-MI score for one candidate (Vinh 2016).

    Args:
        x_cand, selected_cols, y, nbins_*: same shape as ``jmim_score``.
        alpha: weight on the 3-way interaction term (default 1.0 per Vinh 2016).

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
    # Relevance I(X; Y).
    relevance = _mi_pair_njit(x_int, y_int, K_x, K_y)
    n_S = len(selected_cols)
    if n_S == 0:
        return float(relevance)
    # Pairwise redundancy (1/|S|) sum_j I(X; X_j): marginal MI between the candidate and each already-selected feature.
    # A candidate that duplicates a selected feature gets a large penalty; an independent one gets ~0.
    sel_int = [col.astype(np.int64) for col in selected_cols]
    K_sel = [int(k) for k in nbins_selected]
    pair_red = 0.0
    marg_mi = np.empty(n_S, dtype=np.float64)  # I(X; X_j)
    cmi_given_y = np.empty(n_S, dtype=np.float64)  # I(X; X_j | Y)
    for j in range(n_S):
        K_z = K_sel[j]
        marg_mi[j] = _mi_pair_njit(x_int, sel_int[j], K_x, K_z)
        pair_red += marg_mi[j]
        cmi_given_y[j] = _cmi_xy_given_z_njit(x_int, y_int, sel_int[j], K_x, K_y, K_z)
    pair_red /= float(n_S)
    # 3-way interaction-information correction: alpha / C(|S|,2) * sum_{i<j} II(X; Z_i; Z_j),
    # where II = I(X; Z_i; Z_j | Y) - I(X; Z_i; Z_j) and each co-information is decomposed as
    # CO_cond  = I(X; Z_i | Y) + I(X; Z_j | Y) - I(X; Z_i, Z_j | Y),
    # CO_uncond= I(X; Z_i)     + I(X; Z_j)     - I(X; Z_i, Z_j).
    # II > 0 means the pair (Z_i, Z_j) carries MORE about X once Y is fixed than unconditionally (synergy) -> reward;
    # II < 0 means the joint already explains X without Y (redundancy) -> penalty. Adding alpha*II therefore lowers the
    # score of jointly-redundant candidates and raises synergistic ones, the direction RelaxMRMR (Vinh 2016) intends.
    inter = 0.0
    if n_S >= 2 and alpha > 0.0:
        norm = float(n_S * (n_S - 1)) / 2.0
        for i in range(n_S):
            for j in range(i + 1, n_S):
                col_i = sel_int[i]
                col_j = sel_int[j]
                K_i = K_sel[i]
                K_j = K_sel[j]
                cmi_ij = _joint_cmi_xy_given_zw_njit(x_int, y_int, col_i, col_j, K_x, K_y, K_i, K_j)
                co_cond = cmi_given_y[i] + cmi_given_y[j] - cmi_ij
                mi_x_zz = _mi_x_pair_njit(x_int, col_i, col_j, K_x, K_i, K_j)
                co_uncond = marg_mi[i] + marg_mi[j] - mi_x_zz
                inter += co_cond - co_uncond
        inter *= float(alpha) / norm
    return float(relevance - pair_red + inter)


__all__ = ["relax_mrmr_score"]
