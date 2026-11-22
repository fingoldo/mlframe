"""Kraskov (2004) k-NN mutual information between one-dimensional continuous columns and a continuous target, in numba.

sklearn's ``mutual_info_regression`` fits a ``NearestNeighbors`` tree on the (x, y) plane and two ``KDTree`` radius counters per column. With a single feature the plane search is a walk over the points sorted by x that stops
once the x gap exceeds the k-th best Chebyshev distance, and the two marginal counts are walks over the sorted values, so this kernel reproduces the same estimate (same scaling, same Kraskov noise drawn from the
same ``RandomState`` in the same order, same ``nextafter`` radius, same ``<=`` counting) without trees or joblib. The sibling for a discrete target is ``_ksg_cd_fast``.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ._ksg_cd_fast import _column_mean_abs, _column_std, _psi_table, _row_tiles
from ._numba_utils_shared import radix_order_f64


@njit(cache=True, nogil=True)
def _knn_radii(xs, ys, order, k):
    """For each point (in x order) the Chebyshev distance to its k-th nearest neighbour in the (x, y) plane, lowered by one ulp (sklearn's ``nextafter`` radius), stored by the point's original index."""
    n = xs.shape[0]
    radius_by_point = np.empty(n)
    best = np.empty(k)
    for a in range(n):
        for t in range(k):
            best[t] = np.inf
        xa = xs[a]
        ya = ys[a]
        b = a + 1
        while b < n:
            dx = abs(xs[b] - xa)
            if dx > best[k - 1]:
                break
            d = abs(ys[b] - ya)
            if dx > d:
                d = dx
            if d < best[k - 1]:
                t = k - 1
                while t > 0 and best[t - 1] > d:
                    best[t] = best[t - 1]
                    t -= 1
                best[t] = d
            b += 1
        b = a - 1
        while b >= 0:
            dx = abs(xs[b] - xa)
            if dx > best[k - 1]:
                break
            d = abs(ys[b] - ya)
            if dx > d:
                d = dx
            if d < best[k - 1]:
                t = k - 1
                while t > 0 and best[t - 1] > d:
                    best[t] = best[t - 1]
                    t -= 1
                best[t] = d
            b -= 1
        radius_by_point[order[a]] = np.nextafter(best[k - 1], 0.0)
    return radius_by_point


@njit(cache=True, nogil=True)
def _marginal_digamma_sum(sorted_vals, order, radius_by_point, psi):
    """Sum over the points of ``digamma(count)``, ``count`` the number of values within ``radius`` of the point's own value (itself included): a walk outward in the sorted values, the points taken in sorted
    order (``order[c]`` is the original index of the c-th smallest value) so the values and the walks stay in cache."""
    n = sorted_vals.shape[0]
    total = 0.0
    for c in range(n):
        r = radius_by_point[order[c]]
        v = sorted_vals[c]
        cnt = 1
        b = c + 1
        while b < n and abs(sorted_vals[b] - v) <= r:
            cnt += 1
            b += 1
        b = c - 1
        while b >= 0 and abs(sorted_vals[b] - v) <= r:
            cnt += 1
            b -= 1
        total += psi[cnt]
    return total


@njit(cache=True, nogil=True)
def _mi_cc_column(x, y, y_order, ysorted, n_neighbors, psi):
    """Kraskov MI (nats, floored at 0) of the noisy columns ``x`` and ``y`` (``y_order`` / ``ysorted``: the sort order of ``y`` and its sorted values)."""
    n = x.shape[0]
    k = n_neighbors
    if n <= k:
        return 0.0
    order = radix_order_f64(x)
    xs = np.empty(n)
    ys = np.empty(n)
    for a in range(n):
        xs[a] = x[order[a]]
        ys[a] = y[order[a]]
    radius_by_point = _knn_radii(xs, ys, order, k)
    sum_x = _marginal_digamma_sum(xs, order, radius_by_point, psi)
    sum_y = _marginal_digamma_sum(ysorted, y_order, radius_by_point, psi)
    mi = psi[n] + psi[k] - sum_x / n - sum_y / n
    return mi if mi > 0.0 else 0.0


@njit(parallel=True, cache=True, nogil=True)
def _mi_cc_columns(X, y, n_neighbors):
    """Per-column Kraskov MI for the columns of ``X`` (already scaled and noised) against ``y`` (likewise)."""
    n, n_cols = X.shape
    psi = _psi_table(n)
    y_order = radix_order_f64(y)
    ysorted = np.empty(n)
    for b in range(n):
        ysorted[b] = y[y_order[b]]
    out = np.empty(n_cols)
    for j in prange(n_cols):
        out[j] = _mi_cc_column(np.ascontiguousarray(X[:, j]), y, y_order, ysorted, n_neighbors, psi)
    return out


def ksg_mi_continuous_target(X: np.ndarray, y: np.ndarray, *, n_neighbors: int, random_state: int) -> np.ndarray:
    """Drop-in for ``mutual_info_regression(X, y, discrete_features=False)``: double MI per column, nats."""
    # Intentional float64: reproduces sklearn mutual_info_regression, which scales and adds the noise in double precision
    X = np.array(X, dtype=np.float64, copy=True)
    y = np.array(np.asarray(y).ravel(), dtype=np.float64, copy=True)
    n = X.shape[0]
    std = _column_std(X)
    std[std == 0.0] = 1.0
    X /= std
    means = np.maximum(1.0, _column_mean_abs(X))
    rng = np.random.RandomState(int(random_state))
    # the (n, p) draw comes first, then the target's, exactly as sklearn orders its stream; ``standard_normal`` fills C order from one sequential stream, so row tiles drawn one after the other
    # are the values of a single (n, p) draw, and neither the draw nor its product with the scale is ever held whole (two more copies of the block at the 500000-row memory peak)
    noise_scale = 1e-10 * means
    for r0, r1 in _row_tiles(n, X.shape[1]):
        X[r0:r1] += noise_scale * rng.standard_normal(size=(r1 - r0, X.shape[1]))
    ystd = y.std()
    y /= ystd if ystd != 0.0 else 1.0
    y += 1e-10 * max(1.0, float(np.mean(np.abs(y)))) * rng.standard_normal(size=n)
    return np.asarray(_mi_cc_columns(np.ascontiguousarray(X), np.ascontiguousarray(y), int(n_neighbors)), dtype=np.float64)


def ksg_mi_continuous_target_columns(X: np.ndarray, y: np.ndarray, *, n_neighbors: int, random_state: int) -> np.ndarray:
    """``ksg_mi_continuous_target`` of every column of ``X`` in one call, bit-identical to calling it once per column.

    The one-column call draws its Kraskov noise from a fresh ``RandomState(random_state)``: ``n`` values for the column, then ``n`` for the target. Every column therefore gets the same two draws, so they
    are drawn once here and broadcast, where a single ``(n, p)`` draw would give each column other values.
    """
    # Intentional float64: reproduces sklearn mutual_info_regression, which scales and adds the noise in double precision
    X = np.array(X, dtype=np.float64, copy=True)
    y = np.array(np.asarray(y).ravel(), dtype=np.float64, copy=True)
    n = X.shape[0]
    rng = np.random.RandomState(int(random_state))
    draw_x = np.asarray(rng.standard_normal(size=n))
    draw_y = np.asarray(rng.standard_normal(size=n))
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    X /= std
    means = np.maximum(np.full(X.shape[1], 1.0), np.mean(np.abs(X), axis=0))
    X += (1e-10 * means) * draw_x[:, None]
    ystd = y.std()
    y /= ystd if ystd != 0.0 else 1.0
    y += 1e-10 * max(1.0, float(np.mean(np.abs(y)))) * draw_y
    return np.asarray(_mi_cc_columns(np.ascontiguousarray(X), np.ascontiguousarray(y), int(n_neighbors)), dtype=np.float64)


@njit(parallel=True, cache=True, nogil=True)
def _knn_cheb2d_kernel(xs, ys, order, k1, dist_out, idx_out):
    """For every point (visited in x order) its ``k1`` nearest points in the Chebyshev metric, itself included, as ascending distances and ORIGINAL row indices."""
    n = xs.shape[0]
    for a in prange(n):
        bd = np.full(k1, np.inf)
        bi = np.full(k1, -1, dtype=np.int64)
        xa = xs[a]
        ya = ys[a]
        bd[0] = 0.0
        bi[0] = order[a]
        for direction in range(2):
            b = a + 1 if direction == 0 else a - 1
            while 0 <= b < n:
                dx = abs(xs[b] - xa)
                if dx > bd[k1 - 1]:
                    break
                d = abs(ys[b] - ya)
                if dx > d:
                    d = dx
                if d < bd[k1 - 1]:
                    t = k1 - 1
                    while t > 0 and bd[t - 1] > d:
                        bd[t] = bd[t - 1]
                        bi[t] = bi[t - 1]
                        t -= 1
                    bd[t] = d
                    bi[t] = order[b]
                b = b + 1 if direction == 0 else b - 1
        for t in range(k1):
            dist_out[order[a], t] = bd[t]
            idx_out[order[a], t] = bi[t]


def knn_chebyshev_2d(points: np.ndarray, k: int) -> "tuple[np.ndarray, np.ndarray]":
    """``(distances, indices)`` of the ``k + 1`` nearest points of every row of the ``(n, 2)`` array ``points`` in the Chebyshev metric (the row itself comes first), the answer of
    ``sklearn.neighbors.KDTree(points, metric="chebyshev").query(points, k=k + 1)`` for points without ties."""
    # Intentional float64: the neighbour distances feed digamma sums, so the points are held in double precision
    P = np.asarray(points, dtype=np.float64)
    n = P.shape[0]
    k1 = min(int(k) + 1, n)
    order = radix_order_f64(np.ascontiguousarray(P[:, 0])).astype(np.int64)
    xs = np.ascontiguousarray(P[order, 0])
    ys = np.ascontiguousarray(P[order, 1])
    dist = np.empty((n, k1))
    idx = np.empty((n, k1), dtype=np.int64)
    _knn_cheb2d_kernel(xs, ys, order, k1, dist, idx)
    return dist, idx
