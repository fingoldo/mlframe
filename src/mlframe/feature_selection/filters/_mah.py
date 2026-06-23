"""Multidimensional Adaptive Histogram (MAH) MI estimator (2026-05-29).

Python port of Marx, Yang, van Leeuwen (2021), "Estimating Conditional Mutual
Information for Discrete-Continuous Mixtures using Multi-Dimensional Adaptive
Histograms", SIAM Int. Conf. Data Mining (SDM 2021). arXiv:2101.05009.

Algorithm overview (marginal 1-D MI variant; the paper's main contribution
is the multi-d CMI version, but the 1-D marginal estimator inherits the same
stochastic-complexity-of-histogram bin-merge logic):

  1. Initial fine quantile binning of X and Y (max K bins each).
  2. Greedy bin merging on the joint table guided by REFINED MDL (Normalised
     Maximum Likelihood, NML) regret over a multinomial code:

         L_NML(joint | M) = -log Sum_{n_1..n_K} prod_i n_i! / n!  (Shtarkov)

     equivalently the regret-of-regret term has a closed form via the
     multinomial NML regret (Mononen-Myllymaeki 2008):

         R(K, N) ~ ((K-1)/2) * log(N / (2 * pi)) + log Gamma(K/2)

     The estimator MERGES bins whenever the NML code shortens (the regret
     reduction outweighs the empirical-likelihood reduction).

  3. After merging, the joint counts give plug-in MI in nats, but the SCI
     score (stochastic complexity index) is what the paper uses for
     hypothesis-test calibration:

         SCI(X; Y) = L_NML(X | flat marginal Y) - L_NML(X | refined joint)

     Under H_0 (X indep Y) the expected SCI is 0; large SCI signals real
     dependence. We return SCI / N -- a regret-normalised proxy for I(X; Y)
     in nats per sample, with the same 0-under-independence property.

Reference: https://arxiv.org/abs/2101.05009
R code (canonical): https://github.com/ylincen/CMI-adaptive-hist
Background: Marx & Vreeken (2019), "Testing Conditional Independence on
Discrete Data using Stochastic Complexity", AISTATS 2019, arXiv:1903.04829.

Implementation notes (per README.md optimization methodology):
  * njit-compiled NML regret + greedy merge inner loop.
  * Vectorised initial quantile binning via numpy.
  * No external deps beyond numpy/scipy/numba.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def _nml_regret(K: int, N: int) -> float:
    """Mononen-Myllymaki (2008) closed-form approximation of the multinomial
    NML regret for K bins on N samples:

        R(K, N) ~ ((K-1)/2) * log(N / (2 * pi)) + log Gamma(K/2) - (K/2) * log Gamma(1/2)

    Returns the regret in nats. R(1, N) = 0 by convention.
    """
    if K <= 1 or N <= 0:
        return 0.0
    half_K = float(K) / 2.0
    return (
        (K - 1) * 0.5 * math.log(max(N, 1) / (2.0 * math.pi))
        + math.lgamma(half_K)
        - half_K * math.lgamma(0.5)
    )


@njit(nogil=True, cache=True)
def _stochastic_complexity(counts: np.ndarray) -> float:
    """Stochastic complexity (NML code length) of a multinomial:

        L_NML(counts) = -log prod_i (n_i / N)^{n_i}  +  R(K, N)
                     = -sum_i n_i log(n_i / N)        +  R(K, N)
                     =  N * H_plug(counts)            +  R(K, N)
    in nats. Used as the MDL "code length" in Marx 2019/2021 sec. 3.
    """
    N = 0
    K_nonzero = 0
    for c in counts:
        N += int(c)
        if c > 0:
            K_nonzero += 1
    if N <= 0 or K_nonzero <= 1:
        return 0.0
    h = 0.0
    N_f = float(N)
    for c in counts:
        if c > 0:
            p = float(c) / N_f
            h -= p * math.log(p)
    return N * h + _nml_regret(K_nonzero, N)


@njit(nogil=True, cache=True)
def _greedy_merge_bins(joint: np.ndarray) -> np.ndarray:
    """Greedy column- and row-merging of a (K_x, K_y) joint table guided by
    the JOINT NML code length (Marx 2021 sec. 3.2): merge when the regret
    reduction outweighs the log-likelihood increase. Operates on the full
    flat joint matrix, not the marginal -- so merging is data-dependent
    (preserves the joint structure even when the marginal would otherwise
    collapse).
    """
    cur = joint.copy()
    while cur.shape[1] > 2:
        best_delta = 0.0
        best_i = -1
        sci_now = _stochastic_complexity(cur.ravel())
        for i in range(cur.shape[1] - 1):
            # Build candidate merged matrix by adding columns i and i+1.
            new_shape_c = cur.shape[1] - 1
            cand = np.empty((cur.shape[0], new_shape_c), dtype=np.float64)
            for r in range(cur.shape[0]):
                for c in range(new_shape_c):
                    if c < i:
                        cand[r, c] = cur[r, c]
                    elif c == i:
                        cand[r, c] = cur[r, i] + cur[r, i + 1]
                    else:
                        cand[r, c] = cur[r, c + 1]
            sci_merged = _stochastic_complexity(cand.ravel())
            delta = sci_now - sci_merged
            if delta > best_delta:
                best_delta = delta
                best_i = i
        if best_i < 0:
            break
        new = np.empty((cur.shape[0], cur.shape[1] - 1), dtype=np.float64)
        for r in range(cur.shape[0]):
            for c in range(cur.shape[1] - 1):
                if c < best_i:
                    new[r, c] = cur[r, c]
                elif c == best_i:
                    new[r, c] = cur[r, best_i] + cur[r, best_i + 1]
                else:
                    new[r, c] = cur[r, c + 1]
        cur = new
    while cur.shape[0] > 2:
        sci_now = _stochastic_complexity(cur.ravel())
        best_delta = 0.0
        best_i = -1
        for i in range(cur.shape[0] - 1):
            new_shape_r = cur.shape[0] - 1
            cand = np.empty((new_shape_r, cur.shape[1]), dtype=np.float64)
            for r in range(new_shape_r):
                for c in range(cur.shape[1]):
                    if r < i:
                        cand[r, c] = cur[r, c]
                    elif r == i:
                        cand[r, c] = cur[i, c] + cur[i + 1, c]
                    else:
                        cand[r, c] = cur[r + 1, c]
            sci_merged = _stochastic_complexity(cand.ravel())
            delta = sci_now - sci_merged
            if delta > best_delta:
                best_delta = delta
                best_i = i
        if best_i < 0:
            break
        new = np.empty((cur.shape[0] - 1, cur.shape[1]), dtype=np.float64)
        for r in range(cur.shape[0] - 1):
            for c in range(cur.shape[1]):
                if r < best_i:
                    new[r, c] = cur[r, c]
                elif r == best_i:
                    new[r, c] = cur[best_i, c] + cur[best_i + 1, c]
                else:
                    new[r, c] = cur[r + 1, c]
        cur = new
    return cur


@njit(nogil=True, cache=True)
def _mi_from_joint(joint: np.ndarray) -> float:
    """Plug-in MI in nats from a joint count matrix."""
    K_x, K_y = joint.shape
    N = joint.sum()
    if N <= 0:
        return 0.0
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    mi = 0.0
    for i in range(K_x):
        if px[i] <= 0:
            continue
        for j in range(K_y):
            if joint[i, j] <= 0 or py[j] <= 0:
                continue
            p_ij = joint[i, j] / N
            mi += p_ij * math.log(p_ij * N / (px[i] * py[j] / N))
    return mi


def _greedy_merge_with_history(joint: np.ndarray):
    """Identical SCI-greedy merge to ``_greedy_merge_bins`` but ALSO records
    which row/col positions got merged at each step.

    Returns:
        merged (K_x', K_y'): the post-merge joint table.
        row_merges (list[int]): at each step the CURRENT-matrix row index
            that merged with index+1.
        col_merges (list[int]): same for columns.
    """
    cur = joint.copy()
    col_merges = []
    row_merges = []
    while cur.shape[1] > 2:
        sci_now = _stochastic_complexity(cur.ravel())
        best_delta = 0.0
        best_i = -1
        for i in range(cur.shape[1] - 1):
            cand = np.empty((cur.shape[0], cur.shape[1] - 1), dtype=np.float64)
            for r in range(cur.shape[0]):
                for c in range(cur.shape[1] - 1):
                    if c < i:
                        cand[r, c] = cur[r, c]
                    elif c == i:
                        cand[r, c] = cur[r, i] + cur[r, i + 1]
                    else:
                        cand[r, c] = cur[r, c + 1]
            sci_merged = _stochastic_complexity(cand.ravel())
            delta = sci_now - sci_merged
            if delta > best_delta:
                best_delta = delta
                best_i = i
        if best_i < 0:
            break
        col_merges.append(best_i)
        new = np.empty((cur.shape[0], cur.shape[1] - 1), dtype=np.float64)
        for r in range(cur.shape[0]):
            for c in range(cur.shape[1] - 1):
                if c < best_i:
                    new[r, c] = cur[r, c]
                elif c == best_i:
                    new[r, c] = cur[r, best_i] + cur[r, best_i + 1]
                else:
                    new[r, c] = cur[r, c + 1]
        cur = new
    while cur.shape[0] > 2:
        sci_now = _stochastic_complexity(cur.ravel())
        best_delta = 0.0
        best_i = -1
        for i in range(cur.shape[0] - 1):
            cand = np.empty((cur.shape[0] - 1, cur.shape[1]), dtype=np.float64)
            for r in range(cur.shape[0] - 1):
                for c in range(cur.shape[1]):
                    if r < i:
                        cand[r, c] = cur[r, c]
                    elif r == i:
                        cand[r, c] = cur[i, c] + cur[i + 1, c]
                    else:
                        cand[r, c] = cur[r + 1, c]
            sci_merged = _stochastic_complexity(cand.ravel())
            delta = sci_now - sci_merged
            if delta > best_delta:
                best_delta = delta
                best_i = i
        if best_i < 0:
            break
        row_merges.append(best_i)
        new = np.empty((cur.shape[0] - 1, cur.shape[1]), dtype=np.float64)
        for r in range(cur.shape[0] - 1):
            for c in range(cur.shape[1]):
                if r < best_i:
                    new[r, c] = cur[r, c]
                elif r == best_i:
                    new[r, c] = cur[best_i, c] + cur[best_i + 1, c]
                else:
                    new[r, c] = cur[r + 1, c]
        cur = new
    return cur, row_merges, col_merges


def _apply_merges_to_edges(initial_inner_edges: np.ndarray,
                            row_merges) -> np.ndarray:
    """Walk through the row-merge sequence and drop the inner edges that
    got swallowed by a merge.

    ``initial_inner_edges`` has length ``K - 1`` (one inner edge between each
    pair of adjacent quantile bins). Each merge at current-matrix position ``i``
    removes the edge between the ``i``-th and ``(i+1)``-th currently-remaining
    bin -- which corresponds to a specific position in the dropping list.
    """
    # Track ranges of original-bin indices per current bin.
    K = initial_inner_edges.size + 1
    ranges = [[k, k + 1] for k in range(K)]
    for pos in row_merges:
        if pos + 1 >= len(ranges):
            continue
        # Merge ranges[pos] and ranges[pos+1].
        ranges[pos][1] = ranges[pos + 1][1]
        del ranges[pos + 1]
    # Remaining inner edges: original index = ranges[i].end for i in 0..len-2.
    remaining_positions = [ranges[i][1] - 1 for i in range(len(ranges) - 1)]
    if not remaining_positions:
        return np.array([], dtype=np.float64)
    return initial_inner_edges[np.asarray(remaining_positions, dtype=np.int64)]


def mah_bin_edges(x: np.ndarray, y: np.ndarray, *,
                   initial_k: int = 16) -> np.ndarray:
    """Multidimensional Adaptive Histogram bin edges for X (Marx 2021).

    Wave 7 (2026-05-29): exposed as a Family-1 ``nbins_strategy`` option.
    Returns INNER bin edges suitable for ``np.searchsorted(edges, x, side='right')``.

    Pipeline:
      1. Equal-frequency K-bin x; label-encode (or K-bin) y.
      2. Build (K_x, K_y) joint count matrix.
      3. Greedy 2D bin merging guided by SCI (Marx 2021 sec. 3.2).
      4. Drop the X-axis inner edges that got swallowed by row-merges.

    Args:
        x: 1-D continuous.
        y: 1-D supervised target (discrete preferred per the paper's
            discrete-continuous regime; continuous y is quantile-binned to K).
        initial_k: starting bin count. Marx 2021 uses K=16; greedy merge
            adapts down.

    Reference: Marx, Yang, van Leeuwen (2021), SDM 2021. arXiv:2101.05009.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y).ravel()
    n = x.size
    if n < 16:
        return np.array([], dtype=np.float64)
    K = int(initial_k)
    qx = np.quantile(x, np.linspace(0, 1, K + 1))
    qx_unique = np.unique(qx)
    if qx_unique.size < 3:
        return np.array([], dtype=np.float64)
    initial_inner = qx_unique[1:-1].astype(np.float64)
    xb = np.searchsorted(initial_inner, x, side="right").astype(np.int64)
    # Discrete y label-encode, else K-quantile-bin.
    if y.dtype.kind in "iub" or np.unique(y).size <= K:
        uniq_y = np.unique(y)
        yb = np.searchsorted(uniq_y, y).astype(np.int64)
    else:
        qy = np.quantile(y.astype(np.float64), np.linspace(0, 1, K + 1))
        qy_unique = np.unique(qy)
        if qy_unique.size < 3:
            return initial_inner
        yb = np.searchsorted(qy_unique[1:-1], y.astype(np.float64),
                              side="right").astype(np.int64)
    K_x = int(xb.max()) + 1
    K_y = int(yb.max()) + 1
    if K_x < 2 or K_y < 2:
        return initial_inner
    joint = np.bincount(xb * K_y + yb, minlength=K_x * K_y).reshape(K_x, K_y).astype(np.float64)
    _, row_merges, _ = _greedy_merge_with_history(joint)
    # Map row merges to drop the swallowed X edges.
    return _apply_merges_to_edges(initial_inner, row_merges)


def mah_mi(x: np.ndarray, y: np.ndarray, *,
            initial_k: int = 16,
            return_sci: bool = False) -> float:
    """MAH/SCI mutual information estimator (Marx 2021, SDM). See module
    docstring for the algorithm. Returns I(X; Y) in nats; for ``return_sci=True``
    returns the SCI difference (regret-normalised hypothesis test statistic).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y).ravel()
    n = x.size
    if n < 16:
        return 0.0
    K = int(initial_k)
    qx = np.quantile(x, np.linspace(0, 1, K + 1))
    qx_unique = np.unique(qx)
    if qx_unique.size < 3:
        return 0.0
    xb = np.searchsorted(qx_unique[1:-1], x, side="right").astype(np.int64)
    if y.dtype.kind in "iub" or np.unique(y).size <= K:
        uniq_y = np.unique(y)
        yb = np.searchsorted(uniq_y, y).astype(np.int64)
    else:
        qy = np.quantile(y.astype(np.float64), np.linspace(0, 1, K + 1))
        qy_unique = np.unique(qy)
        if qy_unique.size < 3:
            return 0.0
        yb = np.searchsorted(qy_unique[1:-1], y.astype(np.float64),
                              side="right").astype(np.int64)
    K_x = int(xb.max()) + 1
    K_y = int(yb.max()) + 1
    if K_x < 2 or K_y < 2:
        return 0.0
    joint = np.bincount(xb * K_y + yb, minlength=K_x * K_y).reshape(K_x, K_y).astype(np.float64)
    merged, _, _ = _greedy_merge_with_history(joint)
    if return_sci:
        sci_joint = _stochastic_complexity(merged.ravel())
        marg_x = merged.sum(axis=1)
        marg_y = merged.sum(axis=0)
        sci_marg = _stochastic_complexity(marg_x) + _stochastic_complexity(marg_y)
        return max(0.0, float(sci_marg - sci_joint) / float(n))
    return max(0.0, float(_mi_from_joint(merged)))


__all__ = ["mah_mi", "mah_bin_edges"]
