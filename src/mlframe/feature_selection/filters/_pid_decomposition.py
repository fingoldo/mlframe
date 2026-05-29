"""Partial Information Decomposition (PID) — Bertschinger I_ccs approximation.

Decomposes ``I({X_1, X_2}; Y)`` into FOUR additive components:

    I({X_1, X_2}; Y) = Unique(X_1; Y \\ X_2)
                     + Unique(X_2; Y \\ X_1)
                     + Redundant(X_1, X_2; Y)
                     + Synergistic(X_1, X_2; Y)

These can identify features whose value comes from SYNERGY (jointly carry
info that neither alone does -- classic XOR) so the MRMR redundancy step
doesn't discard them. Equally, redundant pairs get correctly flagged.

This module implements the I_ccs (common change in surprisal) approximation
from Ince 2017 ("Measuring Multivariate Redundant Information with
Pointwise Common Change in Surprisal", *Entropy* 19(7):318) -- simpler than
BROJA-PID's LP solve, with closely-matching results on the standard PID
gates (AND, OR, XOR, COPY) per Ince 2017 Table 1.

Algorithm (Williams-Beer 2010 lattice; Ince 2017 I_ccs):
  1. R = I_ccs (redundant info) — pointwise common change in surprisal.
  2. U_1 = I(X_1; Y) - R
  3. U_2 = I(X_2; Y) - R
  4. S = I({X_1, X_2}; Y) - U_1 - U_2 - R

Reference:
  - Williams, P., Beer, R. (2010), "Nonnegative Decomposition of Multivariate
    Information", arXiv:1004.2515.
  - Ince, R. (2017), "Measuring Multivariate Redundant Information with
    Pointwise Common Change in Surprisal", *Entropy* 19(7):318. arXiv:1602.05063
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def _pointwise_log_ratio(p_joint: float, p_marg_a: float,
                          p_marg_b: float) -> float:
    """Pointwise log ratio: log(p_joint / (p_a * p_b)) for the I_ccs
    redundancy decomposition.
    """
    if p_joint <= 0.0 or p_marg_a <= 0.0 or p_marg_b <= 0.0:
        return 0.0
    return math.log(p_joint / (p_marg_a * p_marg_b))


@njit(nogil=True, cache=True)
def _i_ccs_redundancy_njit(joint_xyz: np.ndarray) -> float:
    """Ince 2017 I_ccs redundant information for the (X_1, X_2; Y) PID.

    Args:
        joint_xyz: (K_x1, K_x2, K_y) joint count tensor.

    Returns: redundant information in nats; floored at 0.
    """
    K_x1, K_x2, K_y = joint_xyz.shape
    N = 0.0
    for i in range(K_x1):
        for j in range(K_x2):
            for k in range(K_y):
                N += joint_xyz[i, j, k]
    if N <= 0.0:
        return 0.0
    # Marginals.
    p_y = np.zeros(K_y, dtype=np.float64)
    p_x1 = np.zeros(K_x1, dtype=np.float64)
    p_x2 = np.zeros(K_x2, dtype=np.float64)
    p_x1y = np.zeros((K_x1, K_y), dtype=np.float64)
    p_x2y = np.zeros((K_x2, K_y), dtype=np.float64)
    for i in range(K_x1):
        for j in range(K_x2):
            for k in range(K_y):
                v = joint_xyz[i, j, k] / N
                p_y[k] += v
                p_x1[i] += v
                p_x2[j] += v
                p_x1y[i, k] += v
                p_x2y[j, k] += v
    r_ccs = 0.0
    for i in range(K_x1):
        if p_x1[i] <= 0.0:
            continue
        for j in range(K_x2):
            if p_x2[j] <= 0.0:
                continue
            for k in range(K_y):
                if p_y[k] <= 0.0 or p_x1y[i, k] <= 0.0 or p_x2y[j, k] <= 0.0:
                    continue
                # pointwise change in surprisal for (x_1, y) and (x_2, y).
                dh_x1 = math.log(p_x1y[i, k] / (p_x1[i] * p_y[k]))
                dh_x2 = math.log(p_x2y[j, k] / (p_x2[j] * p_y[k]))
                # I_ccs: contribution only when both have same sign and
                # magnitude min(|dh_x1|, |dh_x2|), then sign.
                if dh_x1 * dh_x2 <= 0.0:
                    continue
                common = min(abs(dh_x1), abs(dh_x2))
                signed = common if dh_x1 > 0.0 else -common
                # Joint observation probability for the contribution weight.
                # Sum over the implicit third coordinate -> use p(x1, x2, y).
                p_xyz = joint_xyz[i, j, k] / N
                r_ccs += p_xyz * signed
    return max(0.0, r_ccs)


@njit(nogil=True, cache=True)
def _mi_x_pair_y_njit(joint: np.ndarray) -> float:
    """I({X_1, X_2}; Y) via plug-in on the 3-D joint tensor."""
    K_x1, K_x2, K_y = joint.shape
    N = 0.0
    for i in range(K_x1):
        for j in range(K_x2):
            for k in range(K_y):
                N += joint[i, j, k]
    if N <= 0.0:
        return 0.0
    # Marginal p(y).
    p_y = np.zeros(K_y, dtype=np.float64)
    # Composite (X_1, X_2) marginal.
    p_xx = np.zeros((K_x1, K_x2), dtype=np.float64)
    for i in range(K_x1):
        for j in range(K_x2):
            for k in range(K_y):
                v = joint[i, j, k] / N
                p_y[k] += v
                p_xx[i, j] += v
    mi = 0.0
    for i in range(K_x1):
        for j in range(K_x2):
            if p_xx[i, j] <= 0.0:
                continue
            for k in range(K_y):
                p_joint = joint[i, j, k] / N
                if p_joint <= 0.0 or p_y[k] <= 0.0:
                    continue
                mi += p_joint * math.log(p_joint / (p_xx[i, j] * p_y[k]))
    return max(0.0, mi)


@njit(nogil=True, cache=True)
def _mi_xy_njit(p_xy: np.ndarray) -> float:
    K_x, K_y = p_xy.shape
    # p_xy is already normalised joint.
    p_x = np.zeros(K_x, dtype=np.float64)
    p_y = np.zeros(K_y, dtype=np.float64)
    for i in range(K_x):
        for j in range(K_y):
            v = p_xy[i, j]
            p_x[i] += v
            p_y[j] += v
    mi = 0.0
    for i in range(K_x):
        if p_x[i] <= 0.0:
            continue
        for j in range(K_y):
            v = p_xy[i, j]
            if v <= 0.0 or p_y[j] <= 0.0:
                continue
            mi += v * math.log(v / (p_x[i] * p_y[j]))
    return max(0.0, mi)


def pid_decomposition(x1: np.ndarray, x2: np.ndarray, y: np.ndarray,
                       K_x1: int, K_x2: int, K_y: int) -> dict:
    """Williams-Beer / Ince I_ccs Partial Information Decomposition.

    Args:
        x1, x2, y: 1-D integer-encoded arrays.
        K_x1, K_x2, K_y: cardinalities.

    Returns: dict with keys ``redundant``, ``unique_x1``, ``unique_x2``,
        ``synergistic``, ``total`` -- all in nats.
    """
    x1 = np.asarray(x1, dtype=np.int64).ravel()
    x2 = np.asarray(x2, dtype=np.int64).ravel()
    y = np.asarray(y, dtype=np.int64).ravel()
    n = x1.size
    if n == 0:
        return {"redundant": 0.0, "unique_x1": 0.0, "unique_x2": 0.0,
                "synergistic": 0.0, "total": 0.0}
    joint = np.zeros((int(K_x1), int(K_x2), int(K_y)), dtype=np.float64)
    for i in range(n):
        joint[x1[i], x2[i], y[i]] += 1.0
    # I_ccs redundancy.
    r = float(_i_ccs_redundancy_njit(joint))
    # I(X_1; Y) and I(X_2; Y) via marginal joint.
    p_x1y = joint.sum(axis=1) / float(n)
    p_x2y = joint.sum(axis=0) / float(n)
    mi_x1_y = float(_mi_xy_njit(p_x1y))
    mi_x2_y = float(_mi_xy_njit(p_x2y))
    # Total: I({X_1, X_2}; Y).
    total = float(_mi_x_pair_y_njit(joint))
    # Unique components.
    u1 = max(0.0, mi_x1_y - r)
    u2 = max(0.0, mi_x2_y - r)
    synergy = max(0.0, total - u1 - u2 - r)
    return {
        "redundant": r,
        "unique_x1": u1,
        "unique_x2": u2,
        "synergistic": synergy,
        "total": total,
    }


__all__ = ["pid_decomposition"]
