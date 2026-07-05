"""MRwMR-BUR — Boosting Unique Relevance (Gao 2022, arXiv:2212.06143).

Adds an explicit "unique relevance" term to the standard MRMR objective:

    score_BUR(X_k) = score_MRMR(X_k) + lambda * U(X_k | S)

where ``U(X_k | S)`` is the relevance of ``X_k`` to ``Y`` that CANNOT be
explained by any feature already in ``S``:

    U(X_k | S) = I(X_k; Y) - max_{X_j in S} I(X_k; X_j)

Intuition: a feature that is strongly relevant to ``Y`` AND uncorrelated with
the selected set gets a bonus. Gao 2022 reports +2-3% accuracy with 25-30%
fewer features selected on UCI benchmark suites.

The BUR term composes orthogonally with Fleuret / JMIM / SU — it is an
ADDITIVE bonus on the MRMR score, not a replacement.

Reference: Gao, Y., Yu, S., Li, B., Wang, Z. (2022), "MRwMR-BUR: Mutual
Information based Feature Selection via Maximizing Relevance with Minimum
Redundancy and Boosting Unique Relevance".
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def _mi_pair_njit(x: np.ndarray, y: np.ndarray, K_x: int, K_y: int) -> float:
    """Plug-in I(X; Y) on integer-bin inputs."""
    n = x.shape[0]
    if n <= 0:
        return 0.0
    joint = np.zeros((K_x, K_y), dtype=np.float64)
    for i in range(n):
        joint[x[i], y[i]] += 1.0
    n_f = float(n)
    Px = np.zeros(K_x, dtype=np.float64)
    Py = np.zeros(K_y, dtype=np.float64)
    for i in range(K_x):
        for j in range(K_y):
            v = joint[i, j]
            Px[i] += v
            Py[j] += v
    mi = 0.0
    for i in range(K_x):
        if Px[i] <= 0.0:
            continue
        for j in range(K_y):
            v = joint[i, j]
            if v <= 0.0 or Py[j] <= 0.0:
                continue
            p = v / n_f
            mi += p * math.log(p * n_f / (Px[i] * Py[j] / n_f))
    return max(0.0, mi)


def bur_term(x_cand: np.ndarray, selected_cols: list[np.ndarray], y: np.ndarray, nbins_x: int, nbins_selected: list[int], nbins_y: int) -> float:
    """Unique-relevance bonus for ``X_cand`` given the selected set.

    Computes ``I(X_cand; Y) - max_{j in S} I(X_cand; X_j)``. The first term
    is the marginal relevance; the second is the strongest correlation of
    ``X_cand`` to any already-selected feature. The difference is the
    portion of ``X_cand``'s relevance that the selected set CANNOT explain
    via marginal correlation -- the additive bonus.

    When ``S`` is empty, returns ``I(X_cand; Y)`` (no selected features to
    explain anything yet).

    Floored at 0 -- a feature whose marginal-y MI is less than its
    correlation with a selected feature gets zero bonus, not a penalty.
    """
    # Guard against out-of-range / -1-sentinel codes: the njit kernel indexes joint[x[i], y[i]] directly, so a
    # negative sentinel wraps to the last bin and an over-range code writes out of bounds (silent corruption). PID
    # hardens the same class explicitly; mirror it here.
    from ._fe_batched_mi import _assert_codes_in_range

    _assert_codes_in_range(x_cand, int(nbins_x), "bur_term x_cand")
    _assert_codes_in_range(y, int(nbins_y), "bur_term y")
    for _j, _c in enumerate(selected_cols):
        _assert_codes_in_range(_c, int(nbins_selected[_j]), "bur_term selected_col")
    x_int = x_cand.astype(np.int64)
    y_int = y.astype(np.int64)
    K_x = int(nbins_x)
    K_y = int(nbins_y)
    mi_xy = _mi_pair_njit(x_int, y_int, K_x, K_y)
    if not selected_cols:
        return float(mi_xy)
    max_corr = 0.0
    for j, col_j in enumerate(selected_cols):
        K_z = int(nbins_selected[j])
        mi_xz = _mi_pair_njit(x_int, col_j.astype(np.int64), K_x, K_z)
        if mi_xz > max_corr:
            max_corr = float(mi_xz)
    return float(max(0.0, mi_xy - max_corr))


__all__ = ["bur_term"]
