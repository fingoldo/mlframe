"""RelaxMRMR / FJMI — 3-D MI feature selection (Vinh 2016).

Adds a 3-D conditional MI redundancy term ``I(X_k; X_j; X_i | Y)`` that
RELAXES Fleuret's conditional-independence assumption. Vinh, Zhou, Chan,
Bailey 2016 (*Pattern Recognition* 53:51-62) show this catches higher-order
redundancy that pairwise CMIM / JMIM both miss when three or more selected
features jointly explain a candidate.

Decomposition (interaction information, McGill 1954):

    I(X; Z_1; Z_2 | Y) = I(X; Z_1 | Y) + I(X; Z_2 | Y) - I(X; Z_1, Z_2 | Y)

The third term is the 3-way joint CMI; subtracting the two pairwise CMIs
yields the SYNERGY/REDUNDANCY interaction term. Positive = synergistic
(jointly carries more), negative = redundant (jointly carries less than sum).

RelaxMRMR score:

    score(X_k) = I(X_k; Y)
                 - (1/|S|) * sum_{j in S} I(X_k; X_j)
                 + (alpha / |S|(|S|-1)) * sum_{i<j in S} I(X_k; X_i; X_j | Y)

Default alpha = 1 matches Vinh 2016. Higher alpha emphasises higher-order
redundancy detection.

Reference: Vinh, N.X., Zhou, J., Chan, J., Bailey, J. (2016), "Can
high-order dependencies improve mutual information based feature
selection?", *Pattern Recognition* 53:51-62.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def _cmi_xy_given_z_njit(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                          K_x: int, K_y: int, K_z: int) -> float:
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
                if v <= 0.0 or Pxz[i, k] <= 0.0 or Pyz[j, k] <= 0.0 \
                        or Pz[k] <= 0.0:
                    continue
                p_xyz = v / n_f
                p_z = Pz[k] / n_f
                p_xz = Pxz[i, k] / n_f
                p_yz = Pyz[j, k] / n_f
                cmi += p_xyz * math.log((p_xyz * p_z) / (p_xz * p_yz))
    return max(0.0, cmi)


@njit(nogil=True, cache=True)
def _joint_cmi_xy_given_zw_njit(x: np.ndarray, y: np.ndarray,
                                  z1: np.ndarray, z2: np.ndarray,
                                  K_x: int, K_y: int,
                                  K_z1: int, K_z2: int) -> float:
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


def relax_mrmr_score(x_cand: np.ndarray, selected_cols: list[np.ndarray],
                      y: np.ndarray, nbins_x: int,
                      nbins_selected: list[int], nbins_y: int,
                      alpha: float = 1.0) -> float:
    """RelaxMRMR / FJMI 3-D-MI score for one candidate (Vinh 2016).

    Args:
        x_cand, selected_cols, y, nbins_*: same shape as ``jmim_score``.
        alpha: weight on the 3-way interaction term (default 1.0 per Vinh 2016).

    Returns: scalar score with full 3-way correction; higher = better.

    Cost: ``O(|S|^2)`` 3-D plug-in MIs per candidate; on large selected
    sets enable the dispatcher only after the per-screen filter has pruned
    the pool.
    """
    from ._jmim_scorer import _joint_mi_3d_njit  # reuse plug-in MI
    x_int = x_cand.astype(np.int64)
    y_int = y.astype(np.int64)
    K_x = int(nbins_x)
    K_y = int(nbins_y)
    # I(X; Y)
    relevance = _joint_mi_3d_njit(
        x_int, np.zeros(x_int.size, dtype=np.int64), y_int,
        K_x, 1, K_y,
    )
    n_S = len(selected_cols)
    if n_S == 0:
        return float(relevance)
    # Pairwise CMI term: (1/|S|) sum_j I(X; X_j) (treated as redundancy proxy
    # per Vinh 2016 eq. 4; using marginal MI keeps cost manageable).
    pair_red = 0.0
    cmis_given_y = np.empty(n_S, dtype=np.float64)
    for j, col_j in enumerate(selected_cols):
        K_z = int(nbins_selected[j])
        pair_red += _joint_mi_3d_njit(
            x_int, col_j.astype(np.int64), np.zeros(x_int.size, dtype=np.int64),
            K_x, K_z, 1,
        )
        cmis_given_y[j] = _cmi_xy_given_z_njit(
            x_int, y_int, col_j.astype(np.int64), K_x, K_y, K_z,
        )
    pair_red /= float(n_S)
    # 3-D interaction term: alpha / |S|(|S|-1) sum_{i<j} I(X; Z_i; Z_j | Y).
    inter = 0.0
    if n_S >= 2 and alpha > 0.0:
        norm = float(n_S * (n_S - 1)) / 2.0
        for i in range(n_S):
            for j in range(i + 1, n_S):
                col_i = selected_cols[i].astype(np.int64)
                col_j = selected_cols[j].astype(np.int64)
                K_i = int(nbins_selected[i])
                K_j = int(nbins_selected[j])
                # I(X; Z_i | Y) + I(X; Z_j | Y) - I(X; Z_i, Z_j | Y).
                cmi_ij = _joint_cmi_xy_given_zw_njit(
                    x_int, y_int, col_i, col_j,
                    K_x, K_y, K_i, K_j,
                )
                inter += cmis_given_y[i] + cmis_given_y[j] - cmi_ij
        inter *= float(alpha) / norm
    return float(relevance - pair_red + inter)


__all__ = ["relax_mrmr_score"]
