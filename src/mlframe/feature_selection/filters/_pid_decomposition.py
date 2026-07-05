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
def _pointwise_log_ratio(p_joint: float, p_marg_a: float, p_marg_b: float) -> float:
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
def _occupied_counts_2d(p_xy: np.ndarray):
    """Occupied (non-empty) marginal bin counts (k_x, k_y) of a normalised 2-D joint."""
    K_x, K_y = p_xy.shape
    kx = 0
    for i in range(K_x):
        s = 0.0
        for j in range(K_y):
            s += p_xy[i, j]
        if s > 0.0:
            kx += 1
    ky = 0
    for j in range(K_y):
        s = 0.0
        for i in range(K_x):
            s += p_xy[i, j]
        if s > 0.0:
            ky += 1
    return kx, ky


@njit(nogil=True, cache=True)
def _mm_mi_correct(mi_plugin: float, k_x: int, k_y: int, n: int) -> float:
    """Miller-Madow MI bias correction ``I_mm = I_plugin - (k_x-1)(k_y-1)/(2n)`` on OCCUPIED bin counts.

    Mirrors :func:`info_theory._entropy_kernels.mi_miller_madow_correct`. Pass-through when either side is
    degenerate (``k <= 1``) so the plug-in value is untouched.
    """
    if k_x <= 1 or k_y <= 1 or n <= 0:
        return mi_plugin
    return mi_plugin - (k_x - 1) * (k_y - 1) / (2.0 * n)


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


def pid_decomposition(x1: np.ndarray, x2: np.ndarray, y: np.ndarray, K_x1: int, K_x2: int, K_y: int) -> dict:
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
        return {"redundant": 0.0, "unique_x1": 0.0, "unique_x2": 0.0, "synergistic": 0.0, "total": 0.0}
    # 2026-05-30 Wave 9.1 fix (loop iter 24): validate index ranges
    # explicitly. Pre-fix the joint-tabulation loop accepted negative
    # values silently because numpy negative-indexing wraps to the last
    # bin: x1[i]=-1 -> joint[K_x1-1, ...] += 1. Upper-bound overflow
    # correctly raised IndexError (asymmetric handling - the smoking
    # gun). Downstream NaN-sentinel callers (e.g. discretizers that
    # use -1 for NaN per iter-11 convention) would silently corrupt
    # PID output. Mirror the explicit-raise pattern from iter 13
    # (factorize raise) and iter 22 (target_encoding raise).
    if (x1 < 0).any() or (x2 < 0).any() or (y < 0).any():
        raise ValueError(
            "pid_decomposition: negative integer indices not allowed; "
            "drop or recode NaN sentinels before calling. "
            f"Got x1.min()={int(x1.min())}, x2.min()={int(x2.min())}, "
            f"y.min()={int(y.min())}."
        )
    if int(x1.max()) >= int(K_x1) or int(x2.max()) >= int(K_x2) or int(y.max()) >= int(K_y):
        raise ValueError(
            "pid_decomposition: index exceeds declared cardinality. "
            f"Got x1.max()={int(x1.max())} K_x1={K_x1}, "
            f"x2.max()={int(x2.max())} K_x2={K_x2}, "
            f"y.max()={int(y.max())} K_y={K_y}."
        )
    K_x2_i = int(K_x2)
    K_y_i = int(K_y)
    flat_idx = (x1 * K_x2_i + x2) * K_y_i + y
    joint = np.bincount(flat_idx, minlength=int(K_x1) * K_x2_i * K_y_i).reshape(int(K_x1), K_x2_i, K_y_i).astype(np.float64)
    # I_ccs redundancy.
    r = float(_i_ccs_redundancy_njit(joint))
    # I(X_1; Y) and I(X_2; Y) via marginal joint.
    p_x1y = joint.sum(axis=1) / float(n)
    p_x2y = joint.sum(axis=0) / float(n)
    mi_x1_y = float(_mi_xy_njit(p_x1y))
    mi_x2_y = float(_mi_xy_njit(p_x2y))
    # Total: I({X_1, X_2}; Y).
    total = float(_mi_x_pair_y_njit(joint))

    # Miller-Madow bias-correct each plug-in MI on its OCCUPIED bin counts before the synergy subtraction.
    # Synergy = total - U1 - U2 - R is a difference of plug-in MIs, and the ``total`` term is I over the COMPOSITE
    # (X1,X2) source -- a joint whose occupied cardinality is ~K_x1*K_x2, far over-binned vs the 2-D marginal MIs.
    # The 3-D over-binning inflates ``total``'s positive plug-in bias more than U1/U2/R's, so on an INDEPENDENT
    # high-cardinality pair at small n the raw difference reports spurious positive synergy. Correcting each MI on
    # its own occupied k removes the asymmetry (-> 0 as n -> inf, so large-n PID is untouched).
    p_xx = joint.sum(axis=2) / float(n)
    k_x1, k_y1 = _occupied_counts_2d(p_x1y)
    k_x2, k_y2 = _occupied_counts_2d(p_x2y)
    k_xx = int((p_xx > 0.0).sum())  # occupied composite (X1,X2)-source CELLS (not marginal rows) -- the true source cardinality
    # y-cardinality for the ``total = I({X1,X2};Y)`` MM-correction. ``p_x1y = joint.sum(axis=1)`` has ALREADY summed over X2, so its
    # y-marginal occupancy equals the FULL composite y-marginal occupancy ``(joint.sum(axis=(0,1)) > 0).sum()`` -- a y-bin is occupied
    # in ``p_x1y`` iff some (x1, x2) row carries it. Reusing the X1-marginal here is therefore NOT an X1-only undercount: it is the exact
    # composite occupied-y count (verified equal for all joints), so a y-bin X2 occupies but X1 does not is still counted.
    _, k_yj = _occupied_counts_2d(p_x1y)
    mi_x1_y = max(0.0, _mm_mi_correct(mi_x1_y, k_x1, k_y1, n))
    mi_x2_y = max(0.0, _mm_mi_correct(mi_x2_y, k_x2, k_y2, n))
    total = max(0.0, _mm_mi_correct(total, k_xx, k_yj, n))
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
