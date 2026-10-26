"""Ross (2014) continuous-versus-discrete k-NN mutual information for one-dimensional columns, in numba.

sklearn's ``mutual_info_classif`` builds a KD-tree per class and per column; with a single continuous feature the neighbour search is a walk over the sorted values, so this kernel
reproduces the same estimate (same scaling, same Kraskov noise drawn from the same ``RandomState``, same ``nextafter`` radius) without trees or joblib.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _psi_table(n):
    """digamma at integers 1..n via psi(m+1) = psi(m) + 1/m, psi(1) = -Euler gamma."""
    out = np.zeros(n + 2)
    out[1] = -0.5772156649015329
    for m in range(1, n + 1):
        out[m + 1] = out[m] + 1.0 / m
    return out


@njit(cache=True)
def _mi_cd_column(x, d, n_classes, n_neighbors, psi):
    """Ross MI (nats) between the noisy continuous column ``x`` and integer labels ``d`` in 0..n_classes-1."""
    n = x.shape[0]
    counts = np.zeros(n_classes, dtype=np.int64)
    for i in range(n):
        counts[d[i]] += 1
    order = np.argsort(x)
    xs = np.empty(n)
    ds = np.empty(n, dtype=np.int64)
    n_kept = 0
    for j in range(n):
        i = order[j]
        if counts[d[i]] > 1:
            xs[n_kept] = x[i]
            ds[n_kept] = d[i]
            n_kept += 1
    if n_kept == 0:
        return 0.0
    base = np.zeros(n_classes + 1, dtype=np.int64)
    for c in range(n_classes):
        base[c + 1] = base[c] + (counts[c] if counts[c] > 1 else 0)
    fill = np.zeros(n_classes, dtype=np.int64)
    cls_vals = np.empty(n_kept)
    pos = np.empty(n_kept, dtype=np.int64)
    for j in range(n_kept):
        c = ds[j]
        cls_vals[base[c] + fill[c]] = xs[j]
        pos[j] = fill[c]
        fill[c] += 1
    sum_k = 0.0
    sum_cnt = 0.0
    sum_m = 0.0
    xk = xs[:n_kept]
    for j in range(n_kept):
        c = ds[j]
        cnt = counts[c]
        k = min(n_neighbors, cnt - 1)
        b = base[c]
        i = pos[j]
        v = xs[j]
        lo = i - 1
        hi = i + 1
        r = 0.0
        for _ in range(k):
            dl = v - cls_vals[b + lo] if lo >= 0 else np.inf
            dh = cls_vals[b + hi] - v if hi < cnt else np.inf
            if dl <= dh:
                r = dl
                lo -= 1
            else:
                r = dh
                hi += 1
        # Points strictly closer than the k-th neighbour (sklearn queries radius nextafter(r, 0)); the searchsorted bounds can round onto the neighbour itself, so trim by exact distance.
        lo_m = np.searchsorted(xk, v - r, side="left")
        hi_m = np.searchsorted(xk, v + r, side="right")
        while lo_m < hi_m and v - xk[lo_m] >= r:
            lo_m += 1
        while hi_m > lo_m and xk[hi_m - 1] - v >= r:
            hi_m -= 1
        m = hi_m - lo_m
        sum_k += psi[k]
        sum_cnt += psi[cnt]
        sum_m += psi[m]
    mi = psi[n_kept] + sum_k / n_kept - sum_cnt / n_kept - sum_m / n_kept
    return mi if mi > 0.0 else 0.0


@njit(parallel=True, cache=True)
def _mi_cd_columns(X, d, n_classes, n_neighbors):
    """Per-column Ross MI for the columns of ``X`` (already scaled and noised)."""
    n_cols = X.shape[1]
    psi = _psi_table(X.shape[0])
    out = np.empty(n_cols)
    for j in prange(n_cols):
        out[j] = _mi_cd_column(np.ascontiguousarray(X[:, j]), d, n_classes, n_neighbors, psi)
    return out


def ksg_mi_discrete_target(X: np.ndarray, y: np.ndarray, *, n_neighbors: int, random_state: int) -> np.ndarray:
    """Drop-in for ``mutual_info_classif(X, y, discrete_features=False)``: float64 MI per column, nats."""
    X = np.array(X, dtype=np.float64, copy=True)
    n = X.shape[0]
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    X /= std
    means = np.maximum(1.0, np.mean(np.abs(X), axis=0))
    rng = np.random.RandomState(int(random_state))
    X += 1e-10 * means * rng.standard_normal(size=(n, X.shape[1]))
    _, d = np.unique(np.asarray(y).ravel(), return_inverse=True)
    d = np.ascontiguousarray(d, dtype=np.int64)
    return cast(np.ndarray, _mi_cd_columns(np.ascontiguousarray(X), d, int(d.max()) + 1 if d.size else 0, int(n_neighbors)))
