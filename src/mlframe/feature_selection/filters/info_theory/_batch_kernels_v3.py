"""Column-major batched FE-candidate MI kernel with the observed-MI and permutation noise gate (v3).

Same contract and bit-identical output as :func:`_batch_kernels.batch_mi_with_noise_gate_v2` (same dense codes, observed MI, noise-gate
verdict and shuffle stream). The only change is memory layout. ``disc_2d`` is C-ordered ``(n, K)``, so reading one candidate column
strides by ``K`` elements and writing its dense codes into a ``(n, K)`` buffer strides the same way: at the canonical 2000 x 7668 chunk
every column touches ~n cache lines twice. Here the input is transposed once (blocked, parallel) to ``(K, n)``, the dense codes are written
to a ``(K, n)`` buffer, so each column is read and written contiguously, and the permutation gate reads its contiguous code row directly.

Measured on the captured 2000 x 7668 int8 call (min of 5): the observed-MI pass drops from ~173 ms to ~19 ms (transpose ~10 ms), with
output identical to v2 bit for bit.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(parallel=True, nogil=True, cache=True)
def _transpose_blocked(src: np.ndarray, dst: np.ndarray) -> None:
    """``dst[k, r] = src[r, k]`` in 64 x 64 tiles, parallel over column tiles (each tile writes its own destination rows)."""
    n = src.shape[0]
    K = src.shape[1]
    B = 64
    n_tiles = (K + B - 1) // B
    for t in prange(n_tiles):
        k0 = t * B
        k1 = min(K, k0 + B)
        for r0 in range(0, n, B):
            r1 = min(n, r0 + B)
            for k in range(k0, k1):
                for r in range(r0, r1):
                    dst[k, r] = src[r, k]


@njit(nogil=True, cache=True)
def _perm_failcount_codes(
    col_codes: np.ndarray,
    freqs_k: np.ndarray,
    K_x: int,
    locals_mat: np.ndarray,
    freqs_y: np.ndarray,
    n: int,
    npermutations: int,
    original_mi_k: float,
    use_su: bool,
    stop_at: int,
) -> int:
    """Permutation fail-count for one column whose dense codes are the contiguous row ``col_codes``. Bit-identical to
    :func:`_batch_kernels._perm_failcount_col` (same joint increments and accumulation order); ``stop_at > 0`` returns as soon as that
    many permutations reached the observed MI, which cannot change the caller's ``nfailed >= max_failed`` verdict."""
    K_y = freqs_y.shape[0]
    # Flat joint (row-major, a * K_y + b): one contiguous zero per permutation and one index per row, same increments in the same order.
    joint = np.zeros(K_x * K_y, dtype=np.int64)
    # Two independent histograms (even / odd rows) break the increment dependency chain when many rows land in the same cell; the
    # counts are integers, so summing the two lanes is exact.
    joint2 = np.zeros(K_x * K_y, dtype=np.int64)
    inv_n = 1.0 / n
    su_denom = 0.0
    if use_su:
        h_x = 0.0
        for a in range(K_x):
            p = freqs_k[a]
            if p > 0:
                h_x -= p * math.log(p)
        h_y = 0.0
        for b in range(K_y):
            p = freqs_y[b]
            if p > 0:
                h_y -= p * math.log(p)
        su_denom = h_x + h_y
    nfk = 0
    for i in range(npermutations):
        joint[:] = 0
        joint2[:] = 0
        lm = locals_mat[i]
        for r in range(0, n - 1, 2):
            joint[col_codes[r] * K_y + lm[r]] += 1
            joint2[col_codes[r + 1] * K_y + lm[r + 1]] += 1
        if n % 2 == 1:
            joint[col_codes[n - 1] * K_y + lm[n - 1]] += 1
        for c in range(K_x * K_y):
            joint[c] += joint2[c]
        mi_perm = 0.0
        for a in range(K_x):
            prob_x = freqs_k[a]
            for b in range(K_y):
                jc = joint[a * K_y + b]
                if jc != 0:
                    jf = jc * inv_n
                    mi_perm += jf * math.log(jf / (prob_x * freqs_y[b]))
        if use_su:
            if mi_perm < 0.0:
                mi_perm = 0.0
            mi_perm = 0.0 if su_denom <= 1e-12 else 2.0 * mi_perm / su_denom
        if mi_perm >= original_mi_k:
            nfk += 1
            if stop_at > 0 and nfk >= stop_at:
                return nfk
    return nfk


@njit(nogil=True, cache=True)
def _observed_column(disc_t, k, nb_k, classes_y, freqs_y, n, use_su, classes_dense_t, freqs_dense, kx, original_mi):
    """Observed MI of column ``k`` (dense codes written to ``classes_dense_t[k]``, ``freqs_dense[k]`` and ``kx[k]`` set)."""
    K_y = freqs_y.shape[0]
    inv_n = 1.0 / n
    counts = np.zeros(nb_k, dtype=np.int64)
    for r in range(n):
        counts[disc_t[k, r]] += 1
    lookup = np.empty(nb_k, dtype=np.int64)
    nzeros = 0
    for c in range(nb_k):
        if counts[c] == 0:
            nzeros += 1
        lookup[c] = c - nzeros
    K_x = nb_k - nzeros
    kx[k] = K_x
    wpos = 0
    for c in range(nb_k):
        if counts[c] != 0:
            freqs_dense[k, wpos] = counts[c] / n
            wpos += 1
    joint = np.zeros((K_x, K_y), dtype=np.int64)
    for r in range(n):
        dc = lookup[disc_t[k, r]]
        classes_dense_t[k, r] = dc
        joint[dc, classes_y[r]] += 1
    mi_xy = 0.0
    for i in range(K_x):
        prob_x = freqs_dense[k, i]
        for j in range(K_y):
            jc = joint[i, j]
            if jc != 0:
                prob_y = freqs_y[j]
                jf = jc * inv_n
                mi_xy += jf * math.log(jf / (prob_x * prob_y))
    if not use_su:
        original_mi[k] = mi_xy
        return
    h_x = 0.0
    for i in range(K_x):
        p = freqs_dense[k, i]
        if p > 0:
            h_x -= p * math.log(p)
    h_y = 0.0
    for j in range(K_y):
        p = freqs_y[j]
        if p > 0:
            h_y -= p * math.log(p)
    denom = h_x + h_y
    if denom <= 1e-12:
        original_mi[k] = 0.0
    else:
        if mi_xy < 0.0:
            mi_xy = 0.0
        original_mi[k] = 2.0 * mi_xy / denom


@njit(nogil=True, cache=True)
def _permutation_locals(classes_y_safe, base_seed, npermutations):
    """Shuffled y-vectors for permutations ``0..npermutations-1`` of ``base_seed`` (the shared LCG Fisher-Yates stream)."""
    ny = classes_y_safe.shape[0]
    locals_mat = np.empty((npermutations, ny), dtype=classes_y_safe.dtype)
    for i in range(npermutations):
        state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(i + 1)
        local = classes_y_safe.copy()
        for j in range(ny - 1, 0, -1):
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            kk = int(state >> np.uint64(33)) % (j + 1)
            tmp = local[j]
            local[j] = local[kk]
            local[kk] = tmp
        locals_mat[i] = local
    return locals_mat


@njit(parallel=True, nogil=True, cache=True)
def batch_mi_with_noise_gate_v3(
    disc_2d: np.ndarray,
    factors_nbins: np.ndarray,
    classes_y: np.ndarray,
    classes_y_safe: np.ndarray,
    freqs_y: np.ndarray,
    npermutations: int,
    base_seed: np.uint64,
    min_nonzero_confidence: float,
    use_su: bool,
    dtype: type = np.int32,
    classes_dtype: type = np.int16,
) -> np.ndarray:
    """Batched FE-candidate MI with the permutation noise gate, column-major (see module docstring). Returns the gated MI per column,
    identical to ``batch_mi_with_noise_gate_v2`` for the same arguments."""
    n = disc_2d.shape[0]
    K = disc_2d.shape[1]
    fe_mi = np.zeros(K, dtype=np.float64)
    if K == 0 or n == 0:
        return fe_mi

    max_nbins = 1
    for k in range(K):
        nb_k = int(factors_nbins[k])
        if nb_k > max_nbins:
            max_nbins = nb_k
    # Same capacity guard as v2: a dense code that cannot be represented by classes_dtype would wrap silently.
    if max_nbins > 1 and classes_dtype(max_nbins - 1) != max_nbins - 1:
        raise ValueError("batch_mi_with_noise_gate: factors_nbins exceeds classes_dtype capacity; widen classes_dtype (int32).")

    disc_t = np.empty((K, n), dtype=disc_2d.dtype)
    _transpose_blocked(disc_2d, disc_t)
    classes_dense_t = np.empty((K, n), dtype=classes_dtype)
    freqs_dense = np.zeros((K, max_nbins), dtype=np.float64)
    kx = np.zeros(K, dtype=np.int64)
    original_mi = np.zeros(K, dtype=np.float64)
    for k in prange(K):
        _observed_column(disc_t, k, int(factors_nbins[k]), classes_y, freqs_y, n, use_su, classes_dense_t, freqs_dense, kx, original_mi)

    if npermutations <= 0:
        for k in range(K):
            fe_mi[k] = original_mi[k]
        return fe_mi

    max_failed = int(npermutations * (1.0 - min_nonzero_confidence))
    if max_failed <= 1:
        max_failed = 1
    nfailed = np.zeros(K, dtype=np.int64)
    locals_mat = _permutation_locals(classes_y_safe, base_seed, npermutations)
    for k in prange(K):
        if original_mi[k] <= 0.0:
            continue
        nfailed[k] = _perm_failcount_codes(
            classes_dense_t[k],
            freqs_dense[k],
            int(kx[k]),
            locals_mat,
            freqs_y,
            n,
            npermutations,
            original_mi[k],
            use_su,
            max_failed,
        )
    for k in range(K):
        om = original_mi[k]
        if om > 0.0 and nfailed[k] >= max_failed:
            fe_mi[k] = 0.0
        else:
            fe_mi[k] = om
    return fe_mi
