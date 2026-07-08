"""Chao-Shen entropy correction (2026-05-30 Wave 8).

Better small-sample / sparse-contingency entropy estimator than Miller-Madow.
Per the Pawluszek-Filipiak 2025 *Information* 16(9):724 empirical study,
Chao-Shen and James-Stein shrinkage estimators outperform Miller-Madow on
sparse contingency tables that arise when bin count rises with the
``nbins_strategy='fd'`` / ``'qs'`` defaults.

Algorithm (Chao & Shen 2003, *Environmental and Ecological Statistics* 10:429):

  1. Estimate the unseen-symbol coverage: ``C_hat = 1 - f_1 / N`` where
     ``f_1`` is the count of singletons (categories observed exactly once).
  2. Coverage-adjusted relative frequencies: ``p_i = C_hat * (n_i / N)``.
  3. Sample-coverage entropy:
        H_CS = - sum_i [p_i * log(p_i) / (1 - (1 - p_i)^N)]
     where the denominator corrects for the truncation bias of empirical
     frequencies (sample inclusion probability term).

Returns the entropy in nats. Floor at 0 for finite-sample negative noise.

Reference: Chao, A., & Shen, T.-J. (2003). "Nonparametric estimation of
Shannon's index of diversity when there are unseen species in sample."
*Environmental and Ecological Statistics* 10, 429-443.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def chao_shen_entropy_from_counts(counts: np.ndarray, coverage: float = -1.0) -> float:
    """Chao-Shen entropy estimator on a 1-D integer count array.

    Args:
        counts: int64 array of category counts. Sum > 0; len = K categories.
        coverage: if ``>= 0``, use this externally-supplied Good-Turing coverage ``C_hat`` instead of estimating
            it from ``counts``. Required by :func:`_joint_chao_shen_mi_njit` so the marginal and joint entropies
            share ONE coverage basis -- a per-table coverage makes the three terms incommensurable and turns the
            ``H_x+H_y-H_xy`` subtraction into a non-valid coverage-corrected MI. Default ``-1`` = estimate locally.

    Returns:
        H_CS in nats; floored at 0.
    """
    N = 0
    f_1 = 0  # singletons count
    K = counts.shape[0]
    for k in range(K):
        c = counts[k]
        if c <= 0:
            continue
        N += int(c)
        if c == 1:
            f_1 += 1
    if N <= 0:
        return 0.0
    # Coverage estimate (Good-Turing): externally supplied when the caller needs a shared basis across tables.
    if coverage >= 0.0:
        C_hat = coverage
    else:
        C_hat = 1.0 - float(f_1) / float(N)
    # If all observations are singletons, C_hat -> 0; fall back to plug-in.
    if C_hat <= 1e-12:
        N_f = float(N)
        h = 0.0
        for k in range(K):
            c = counts[k]
            if c > 0:
                p = float(c) / N_f
                h -= p * math.log(p)
        return max(0.0, h)
    N_f = float(N)
    h_cs = 0.0
    for k in range(K):
        c = counts[k]
        if c <= 0:
            continue
        p_emp = float(c) / N_f
        p_adj = C_hat * p_emp
        if p_adj <= 0.0:
            continue
        # Stable 1-(1-p_adj)^N via -expm1(N*log1p(-p_adj)): the naive form catastrophically cancels
        # for small p_adj*N (rare category, large N — exactly this estimator's target regime), silently
        # tripping the <=1e-12 guard and dropping the category.
        denom = -math.expm1(N * math.log1p(-p_adj))
        if denom <= 0.0:
            continue
        h_cs -= p_adj * math.log(p_adj) / denom
    return max(0.0, h_cs)


def chao_shen_entropy(x: np.ndarray) -> float:
    """Chao-Shen entropy of a 1-D integer-encoded sample. Convenience wrapper
    that bincounts ``x`` and dispatches to ``chao_shen_entropy_from_counts``."""
    x = np.asarray(x).ravel()
    if x.size == 0:
        return 0.0
    if x.dtype.kind not in "iub":
        # Quantile-bin float input to 10 quantiles before Chao-Shen; matches
        # the dispatch policy of ``_plug_in_mi``.
        q = np.quantile(x.astype(np.float64), np.linspace(0, 1, 11))
        q = np.unique(q)
        if q.size < 2:
            return 0.0
        x = np.searchsorted(q[1:-1], x.astype(np.float64), side="right")
    counts = np.bincount(x.astype(np.int64))
    return float(chao_shen_entropy_from_counts(counts.astype(np.int64)))


@njit(nogil=True, cache=True)
def _joint_chao_shen_mi_njit(joint: np.ndarray) -> float:
    """Chao-Shen MI from a joint count matrix (K_x, K_y)."""
    K_x, K_y = joint.shape
    if K_x < 1 or K_y < 1:
        return 0.0
    # Marginal counts.
    row_sums = np.zeros(K_x, dtype=np.int64)
    col_sums = np.zeros(K_y, dtype=np.int64)
    for i in range(K_x):
        for j in range(K_y):
            v = int(joint[i, j])
            if v > 0:
                row_sums[i] += v
                col_sums[j] += v
    # Flatten joint to 1-D counts for entropy.
    flat = joint.ravel().astype(np.int64)
    # Shared coverage basis: estimate ONE Good-Turing coverage from the joint table (the finest partition, whose
    # singleton structure dominates the sparse-contingency bias the CS correction targets) and apply it to all
    # three entropy terms. A per-term coverage (the pre-fix path) gave H_x/H_y/H_xy three different rescalings, so
    # ``H_x+H_y-H_xy`` was not a coverage-consistent MI -- the mismatched bases left a deterministic residual that
    # inflated MI on sparse joints. With a single C_hat the rescalings are commensurable and largely cancel.
    N = 0
    f_1 = 0
    for t in range(flat.shape[0]):
        c = int(flat[t])
        if c > 0:
            N += c
            if c == 1:
                f_1 += 1
    if N <= 0:
        return 0.0
    C_hat = 1.0 - float(f_1) / float(N)
    if C_hat <= 1e-12:
        C_hat = -1.0  # all-singleton joint: let each term fall back to local plug-in
    H_xy = chao_shen_entropy_from_counts(flat, C_hat)
    H_x = chao_shen_entropy_from_counts(row_sums, C_hat)
    H_y = chao_shen_entropy_from_counts(col_sums, C_hat)
    mi = H_x + H_y - H_xy
    return float(max(0.0, mi))


def chao_shen_mi(x_binned: np.ndarray, y: np.ndarray) -> float:
    """Plug-in-style MI using Chao-Shen marginal/joint entropies.

    Drop-in replacement for ``_adaptive_nbins._plug_in_mi`` when
    ``mi_correction='chao_shen'`` is active. Tighter bias control than
    Miller-Madow on small-sample / sparse-contingency tables.
    """
    x_binned = np.asarray(x_binned).ravel()
    y = np.asarray(y).ravel()
    if x_binned.size == 0:
        return 0.0
    # Coerce y to int.
    if y.dtype.kind not in "iub":
        q = np.quantile(y.astype(np.float64), np.linspace(0, 1, 11))
        q = np.unique(q)
        if q.size < 2:
            return 0.0
        y_b = np.searchsorted(q[1:-1], y.astype(np.float64), side="right").astype(np.int64)
    else:
        y_b = y.astype(np.int64)
    x_b = x_binned.astype(np.int64)
    # Drop entries with negative codes (e.g. NaN sentinels): they would make x_b*K_y+y_b negative and crash bincount.
    if x_b.size:
        _valid = (x_b >= 0) & (y_b >= 0)
        if not _valid.all():
            x_b = x_b[_valid]
            y_b = y_b[_valid]
    K_x = int(x_b.max()) + 1 if x_b.size else 1
    K_y = int(y_b.max()) + 1 if y_b.size else 1
    if K_x < 1 or K_y < 1:
        return 0.0
    joint = np.bincount(x_b * K_y + y_b, minlength=K_x * K_y).reshape(K_x, K_y).astype(np.int64)
    return float(_joint_chao_shen_mi_njit(joint))


__all__ = [
    "chao_shen_entropy",
    "chao_shen_entropy_from_counts",
    "chao_shen_mi",
]
