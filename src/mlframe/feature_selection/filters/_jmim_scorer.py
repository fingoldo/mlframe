"""JMIM scorer — Joint Mutual Information Maximisation (Bennasar 2015).

Drop-in alternative to Fleuret's CMIM ``min_k I(X_k; Y | Z_j)`` aggregator.
JMIM replaces the conditional MI with the joint:

    J_JMIM(X_k) = min_{X_j in S} I(X_k, X_j ; Y)

where the joint MI is computed on the 3-D joint histogram ``p(X_k, X_j, Y)``.

Why it beats CMIM on multi-collinear groups: when ``{Z_1, Z_2, Z_3}`` are
noisy reflections of a latent ``z``, CMIM's ``min_k I(X; Y | Z_k)`` collapses
exactly when there is most redundancy to clean up (each conditional is roughly
``I(X; Y | z)``, so the min is uninformative). JMIM's joint formulation
preserves the synergistic information that CMIM rejects.

Reference: Bennasar, M., Hua, Y., Setchi, R. (2015), "Feature selection
using Joint Mutual Information Maximisation", *Expert Systems with
Applications* 42(22):8520-8532. Brown 2012 ([JMLR](https://jmlr.org/papers/v13/brown12a.html))
unifies CMIM/JMI/DISR as special cases of conditional likelihood
maximisation; JMIM picks the minimum over already-selected.

Implementation notes (per README.md methodology):
  - Uses an njit 3-D joint histogram + entropy aggregator.
  - Plug-in MI inside the histogram for tight integer-bin compatibility.
  - Returns a single scalar per (candidate, selected_set) pair so the
    existing Fleuret driver can dispatch on it without architectural change.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def _joint_mi_3d_njit(x: np.ndarray, z: np.ndarray, y: np.ndarray,
                       K_x: int, K_z: int, K_y: int) -> float:
    """I((X, Z); Y) via plug-in on the 3-D joint count cube.

    Treats (x_i, z_i) as a flat composite category, then collapses to a
    2-D (composite, y) joint and computes plug-in MI.
    """
    n = x.shape[0]
    if n <= 0:
        return 0.0
    K_xz = K_x * K_z
    joint = np.zeros((K_xz, K_y), dtype=np.float64)
    for i in range(n):
        composite = int(x[i]) * K_z + int(z[i])
        joint[composite, int(y[i])] += 1.0
    n_f = float(n)
    Px = np.zeros(K_xz, dtype=np.float64)
    Py = np.zeros(K_y, dtype=np.float64)
    for i in range(K_xz):
        for j in range(K_y):
            v = joint[i, j]
            Px[i] += v
            Py[j] += v
    mi = 0.0
    for i in range(K_xz):
        if Px[i] <= 0.0:
            continue
        for j in range(K_y):
            v = joint[i, j]
            if v <= 0.0 or Py[j] <= 0.0:
                continue
            p = v / n_f
            mi += p * math.log(p * n_f / (Px[i] * Py[j] / n_f))
    return max(0.0, mi)


def jmim_score(x_cand: np.ndarray, selected_cols: list[np.ndarray],
                y: np.ndarray, nbins_x: int, nbins_selected: list[int],
                nbins_y: int) -> float:
    """JMIM redundancy-controlled score for one candidate (Bennasar 2015).

    Args:
        x_cand: 1-D integer-encoded bin codes of the candidate column.
        selected_cols: list of 1-D integer arrays for each already-selected
            feature.
        y: 1-D integer-encoded target.
        nbins_x: bin count of the candidate.
        nbins_selected: bin count of each selected column.
        nbins_y: target bin count.

    Returns:
        ``min_{j in S} I(X_cand, X_j ; Y)`` in nats. When ``S`` is empty,
        returns the simple ``I(X_cand; Y)`` (no redundancy to enforce yet).
    """
    # Guard against out-of-range / -1-sentinel codes: the njit kernel uses each code DIRECTLY as a flat histogram
    # offset (composite = x*K_z + z), so a negative sentinel wraps to the last bin and an over-range code writes out
    # of bounds -- silent histogram corruption. PID hardens the same class explicitly; mirror it here.
    from ._fe_batched_mi import _assert_codes_in_range

    _assert_codes_in_range(x_cand, int(nbins_x), "jmim_score x_cand")
    _assert_codes_in_range(y, int(nbins_y), "jmim_score y")
    for _j, _c in enumerate(selected_cols):
        _assert_codes_in_range(_c, int(nbins_selected[_j]), "jmim_score selected_col")
    if not selected_cols:
        # Fall back to plug-in I(X; Y) for the first feature.
        return _joint_mi_3d_njit(
            x_cand.astype(np.int64),
            np.zeros(x_cand.size, dtype=np.int64),
            y.astype(np.int64),
            int(nbins_x), 1, int(nbins_y),
        )
    K_x = int(nbins_x)
    K_y = int(nbins_y)
    best = float("inf")
    x_int = x_cand.astype(np.int64)
    y_int = y.astype(np.int64)
    for j, col_j in enumerate(selected_cols):
        K_z = int(nbins_selected[j])
        mi = _joint_mi_3d_njit(x_int, col_j.astype(np.int64), y_int,
                                K_x, K_z, K_y)
        if mi < best:
            best = float(mi)
    return max(0.0, best)


__all__ = ["jmim_score"]
