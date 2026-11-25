"""Ross (2014) continuous-versus-discrete k-NN mutual information for one-dimensional columns, in numba.

sklearn's ``mutual_info_classif`` builds a KD-tree per class and per column; with a single continuous feature the neighbour search is a walk over the sorted values, so this kernel
reproduces the same estimate (same scaling, same Kraskov noise drawn from the same ``RandomState``, same ``nextafter`` radius) without trees or joblib.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numba import njit, prange

from mlframe._numba_parallel_guard import parallel_kernel_entry
from ._numba_utils import _psi_table  # shared njit helper, re-exported for callers of this module
from ._numba_utils_shared import radix_order_f64


@njit(cache=True, nogil=True)
def _mi_cd_column(x, d, n_classes, n_neighbors, psi):
    """Ross MI (nats) between the noisy continuous column ``x`` and integer labels ``d`` in 0..n_classes-1."""
    n = x.shape[0]
    counts = np.zeros(n_classes, dtype=np.int64)
    for i in range(n):
        counts[d[i]] += 1
    order = radix_order_f64(x)  # stable LSD radix order of the (finite, noised) column: replaces the comparison argsort
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
        # The bounds are those of ``searchsorted(xk, v - r, "left")`` / ``searchsorted(xk, v + r, "right")``, found by walking out from the point's own sorted position ``j``
        # (``xk[j] == v`` lies inside both): the radius spans a few neighbours, so the walk takes a few steps where each binary search took about 15 cache-missing ones.
        lo_key = v - r
        hi_key = v + r
        lo_m = j
        while lo_m > 0 and xk[lo_m - 1] >= lo_key:
            lo_m -= 1
        hi_m = j + 1
        while hi_m < n_kept and xk[hi_m] <= hi_key:
            hi_m += 1
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


@njit(parallel=True, cache=True, nogil=True)
def _mi_cd_columns(X, d, n_classes, n_neighbors):
    """Per-column Ross MI for the columns of ``X`` (already scaled and noised)."""
    n_cols = X.shape[1]
    psi = _psi_table(X.shape[0])
    out = np.empty(n_cols)
    for j in prange(n_cols):
        out[j] = _mi_cd_column(np.ascontiguousarray(X[:, j]), d, n_classes, n_neighbors, psi)
    return out


def _mi_cd_columns_auto(X: np.ndarray, d: np.ndarray, n_classes: int, n_neighbors: int) -> np.ndarray:
    """``_mi_cd_columns`` on the device when ``_ksg_cd_gpu`` takes the block, else the host kernel."""
    from ._ksg_cd_gpu import mi_cd_columns_device, use_device

    if use_device(int(X.shape[0])):
        got = mi_cd_columns_device(X, d, n_classes, n_neighbors)
        if got is not None:
            return got
    with parallel_kernel_entry():
        return cast(np.ndarray, _mi_cd_columns(X, d, n_classes, n_neighbors))


_COLUMNS_BLOCK_BYTES = 128 * 1024 * 1024  # host bytes of the double working copy of one block of columns in ksg_mi_discrete_target_columns
_NOISE_TILE_ELEMENTS = 4_000_000  # elements of one tile of the Kraskov noise (32 MB in double)


def _row_tiles(n: int, p: int) -> "list[tuple[int, int]]":
    """Row ranges of about ``_NOISE_TILE_ELEMENTS`` cells, so a temporary of one tile replaces one of the whole (n, p) block."""
    step = max(1, int(_NOISE_TILE_ELEMENTS // max(1, p)))
    return [(r0, min(n, r0 + step)) for r0 in range(0, n, step)]


def _column_std(X: np.ndarray) -> np.ndarray:
    """Population std of every column of ``X``, as ``X.std(axis=0)`` but through row tiles: that call holds a temporary of the whole block (443 MB at 500000 x 110 double) while the
    noise and the next stage's blocks are also resident."""
    n, p = X.shape
    total = np.zeros(p)
    for r0, r1 in _row_tiles(n, p):
        total += X[r0:r1].sum(axis=0)
    mean = total / n
    sq = np.zeros(p)
    for r0, r1 in _row_tiles(n, p):
        sq += np.square(X[r0:r1] - mean).sum(axis=0)
    return np.asarray(np.sqrt(sq / n))


def _column_mean_abs(X: np.ndarray) -> np.ndarray:
    """``np.mean(np.abs(X), axis=0)`` through row tiles (the plain form makes an absolute-value copy of the whole block)."""
    n, p = X.shape
    total = np.zeros(p)
    for r0, r1 in _row_tiles(n, p):
        total += np.abs(X[r0:r1]).sum(axis=0)
    return np.asarray(total / n)


def ksg_mi_discrete_target(X: np.ndarray, y: np.ndarray, *, n_neighbors: int, random_state: int) -> np.ndarray:
    """Drop-in for ``mutual_info_classif(X, y, discrete_features=False)``: double MI per column, nats."""
    n, p_all = X.shape
    step = max(1, int(_COLUMNS_BLOCK_BYTES // max(1, 8 * n)))
    if step < p_all:
        # A block too wide for one float64 copy (the copy, the noise tile and the sorted columns of 100000 x 1000 are several GB): the columns are independent, so they go through in blocks, each scaled and
        # jittered by itself with the noise drawn from ``RandomState(random_state + block index)``. The jitter is a tie-breaker of 1e-10 and the MI of a column does not depend on which stream it came from, but
        # the values are no longer the ones sklearn's single (n, p) draw gives (accepted because the selection stays the same); the unblocked path below keeps the sklearn stream.
        out = np.empty(p_all, dtype=np.float64)  # Intentional float64: MI statistic vector
        for b, c0 in enumerate(range(0, p_all, step)):
            c1 = min(p_all, c0 + step)
            out[c0:c1] = ksg_mi_discrete_target(np.ascontiguousarray(X[:, c0:c1]), y, n_neighbors=n_neighbors, random_state=int(random_state) + b)
        return out
    # Intentional float64: reproduces sklearn mutual_info_classif, which scales and adds noise in double
    X = np.array(X, dtype=np.float64, copy=True)
    std = _column_std(X)
    std[std == 0.0] = 1.0
    X /= std
    means = np.maximum(1.0, _column_mean_abs(X))
    rng = np.random.RandomState(int(random_state))
    # the noise block is drawn and added in row tiles: ``RandomState.standard_normal`` fills C order from one sequential stream, so tiles drawn one after the other are the values of a single (n, p) draw, and
    # ``(1e-10 * means) * noise`` is the same two products per cell; the (n, p) float64 draw and its product (two more copies of the block, ~1.4 GB at 30000 x 3018) are never held
    noise_scale = 1e-10 * means
    tile_rows = max(1, int(_NOISE_TILE_ELEMENTS // max(1, X.shape[1])))
    for r0 in range(0, n, tile_rows):
        r1 = min(n, r0 + tile_rows)
        X[r0:r1] += noise_scale * rng.standard_normal(size=(r1 - r0, X.shape[1]))
    _, d = np.unique(np.asarray(y).ravel(), return_inverse=True)
    d = np.ascontiguousarray(d, dtype=np.int64)
    return _mi_cd_columns_auto(np.ascontiguousarray(X), d, int(d.max()) + 1 if d.size else 0, int(n_neighbors))


@njit(cache=True, nogil=True)
def _prepare_columns(x: np.ndarray, draw: np.ndarray) -> np.ndarray:
    """The columns of ``x`` scaled to unit population std (a constant column by 1) with the noise ``1e-10 * max(1, mean|x|) * draw`` added, in double precision.

    The same arithmetic in the same order as :func:`_prepare_columns_numpy` (sequential sums over the rows, one division by the std, one product of the noise scale and the draw), so the result is bit-identical; the
    passes run over the rows with the columns innermost, where the numpy form made about eight passes with temporaries of the block's size (33 ms against 10 ms at 20000 x 100)."""
    n, p = x.shape
    out = np.empty((n, p), dtype=np.float64)  # Intentional float64: the scaled columns and noise are double precision, as sklearn does
    s = np.zeros(p)
    for i in range(n):
        for j in range(p):
            s[j] += x[i, j]
    mean = s / n
    v = np.zeros(p)
    for i in range(n):
        for j in range(p):
            d = x[i, j] - mean[j]
            v[j] += d * d
    std = np.sqrt(v / n)
    for j in range(p):
        if std[j] == 0.0:
            std[j] = 1.0
    m = np.zeros(p)
    for i in range(n):
        for j in range(p):
            xi = x[i, j] / std[j]
            out[i, j] = xi
            m[j] += abs(xi)
    c = np.empty(p)
    for j in range(p):
        c[j] = 1e-10 * max(1.0, m[j] / n)
    for i in range(n):
        di = draw[i]
        for j in range(p):
            out[i, j] += c[j] * di
    return out


def _prepare_columns_numpy(X: np.ndarray, one_draw: np.ndarray) -> np.ndarray:
    """The numpy form of :func:`_prepare_columns` (the reference it is checked against)."""
    # Intentional float64: reproduces sklearn mutual_info_classif, which scales and adds the noise in double precision
    X = np.array(X, dtype=np.float64, copy=True)
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    X /= std
    means = np.maximum(1.0, np.mean(np.abs(X), axis=0))
    X += 1e-10 * means * one_draw[:, None]
    return X


def ksg_mi_discrete_target_columns(X: np.ndarray, y: np.ndarray, *, n_neighbors: int, random_state: int) -> np.ndarray:
    """``ksg_mi_discrete_target`` of every column of ``X`` in one call, bit-identical to calling it per column.

    The per-column call draws its Kraskov noise as one ``(n, 1)`` block from ``RandomState(random_state)``. A single ``(n, p)`` draw would give each column
    other values, so every column here draws from its own fresh ``RandomState(random_state)``, the same ``n`` values the one-column call draws.
    """
    n = X.shape[0]
    # every column draws from a fresh ``RandomState(random_state)``, so all columns get the SAME n values: one draw, broadcast
    # Intentional float64: the noise is drawn in double precision, as the per-column call draws it
    one_draw = np.asarray(np.random.RandomState(int(random_state)).standard_normal(size=n), dtype=np.float64)
    # Intentional float64: reproduces sklearn mutual_info_classif, which scales and adds the noise in double precision
    _, d = np.unique(np.asarray(y).ravel(), return_inverse=True)
    d = np.ascontiguousarray(d, dtype=np.int64)
    n_classes = int(d.max()) + 1 if d.size else 0
    # the columns are independent (scaling, noise and the MI are all per column), so they go through in blocks whose float64 copy stays within ``_COLUMNS_BLOCK_BYTES``: at 30000 x 3018 the whole copy was 725 MB, and
    # several of these calls run at once on the stage threads
    p = int(X.shape[1])
    step = max(1, int(_COLUMNS_BLOCK_BYTES // max(1, 8 * n)))
    if step >= p:
        return _mi_cd_columns_auto(np.ascontiguousarray(_prepare_columns(np.ascontiguousarray(X), one_draw)), d, n_classes, int(n_neighbors))
    out = np.empty(p, dtype=np.float64)  # Intentional float64: MI statistic vector
    for c0 in range(0, p, step):
        c1 = min(p, c0 + step)
        block = _prepare_columns(np.ascontiguousarray(X[:, c0:c1]), one_draw)
        out[c0:c1] = _mi_cd_columns_auto(np.ascontiguousarray(block), d, n_classes, int(n_neighbors))
    return out
