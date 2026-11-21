"""Kraskov (2004) k-NN mutual information between one-dimensional continuous columns and a continuous target, in numba.

sklearn's ``mutual_info_regression`` fits a ``NearestNeighbors`` tree on the (x, y) plane and two ``KDTree`` radius counters per column. With a single feature the plane search is a walk over the points sorted by x that stops
once the x gap exceeds the k-th best Chebyshev distance, and the two marginal counts are walks over the sorted values, so this kernel reproduces the same estimate (same scaling, same Kraskov noise drawn from the
same ``RandomState`` in the same order, same ``nextafter`` radius, same ``<=`` counting) without trees or joblib. The sibling for a discrete target is ``_ksg_cd_fast``.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ._ksg_cd_fast import _psi_table
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
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    X /= std
    means = np.maximum(1.0, np.mean(np.abs(X), axis=0))
    rng = np.random.RandomState(int(random_state))
    X += 1e-10 * means * rng.standard_normal(size=X.shape)  # the (n, p) draw comes first, then the target's, exactly as sklearn orders its stream
    ystd = y.std()
    y /= ystd if ystd != 0.0 else 1.0
    y += 1e-10 * max(1.0, float(np.mean(np.abs(y)))) * rng.standard_normal(size=n)
    return np.asarray(_mi_cc_columns(np.ascontiguousarray(X), np.ascontiguousarray(y), int(n_neighbors)), dtype=np.float64)
