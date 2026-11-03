"""Shared @njit helpers of the MDLP, adaptive-bins, group-statistics and factorisation kernels.

These are imported across several filters submodules, so they live with the ``_numba_utils`` family. Their former
modules re-export them, so every existing import path keeps resolving to the same compiled function.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from ._numba_utils_shuffle import bounded, next_state, seed_state


@njit(nogil=True, cache=True)
def _plug_in_mi_njit(x_binned: np.ndarray, y_b: np.ndarray, K_x: int, K_y: int, miller_madow: bool) -> float:
    """njit core: plug-in MI from already-integer-encoded bins, optional Miller-Madow.

    Miller-Madow (Miller 1955; Madow 1948) bias term: ``(K_x - 1) * (K_y - 1) / (2 * N)``.
    Subtracted post-accumulation, floored at 0 so honest no-signal MI is exactly 0.
    """
    n = x_binned.shape[0]
    if n == 0:
        return 0.0
    joint = np.zeros((K_x, K_y), dtype=np.float64)  # Intentional float64: joint-count accumulator for the MI statistic
    for i in range(n):
        joint[x_binned[i], y_b[i]] += 1.0
    n_f = float(n)
    Px = np.zeros(K_x, dtype=np.float64)  # Intentional float64: marginal-count accumulator for the MI statistic
    Py = np.zeros(K_y, dtype=np.float64)  # Intentional float64: marginal-count accumulator for the MI statistic
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
    if miller_madow:
        bias = (K_x - 1) * (K_y - 1) / (2.0 * n_f)
        mi -= bias
        if mi < 0.0:
            mi = 0.0
    return mi


_FAC_ARRAY_CAP = 16_000_000


@njit(cache=True)
def _factorize_dense_njit(joint: np.ndarray) -> tuple:
    """Factorize an int64 array to dense first-seen ids in one O(n) pass.

    Replaces ``np.unique(joint, return_inverse=True)``'s O(n log n) sort. The
    per-fold joint is bounded (``old_dense(0..mult-1) + c*mult``), so when the
    max id keeps the ``seen`` buffer small we use a direct-array counting pass
    (array indexing, no hashing - ~10x over the typed.Dict form, ~17x over
    np.unique on the common low-cardinality group/cat joints). A typed.Dict
    fallback guards the rare cartesian-blow-up (high-card col x large running
    class count) so the lookup buffer never explodes.

    Ids are assigned FIRST-SEEN, not sorted - semantically equivalent for every
    consumer: the joint feeds only plug-in entropy (count-based, label-
    permutation-invariant) and further renumbering, and the next
    ``joint + c*mult`` step is a bijection regardless of the 0..k-1 permutation.
    nclasses + the induced partition are identical to the numpy form (verified).
    """
    n = joint.size
    if n == 0:
        return joint, 0
    jmax = 0
    jmin = 0
    for i in range(n):
        v = joint[i]
        if v > jmax:
            jmax = v
        if v < jmin:
            jmin = v
    inv = np.empty(n, dtype=np.int64)
    nc = 0
    # A direct table over the id range costs a memset and a cache miss per element once the range is far above the row count; beyond 16 n it is cheaper to hash.
    if jmin >= 0 and jmax < _FAC_ARRAY_CAP and jmax < 16 * n + 1024:
        # Direct-array counting path (fast common case).
        seen = np.full(jmax + 1, -1, dtype=np.int64)
        for i in range(n):
            v = joint[i]
            s = seen[v]
            if s >= 0:
                inv[i] = s
            else:
                seen[v] = nc
                inv[i] = nc
                nc += 1
    else:
        # Open-addressing hash table (power-of-two size, at most half full, Fibonacci hashing, linear probing) for a wide id range or negative ids; ids are first-seen, as in the direct table.
        bits = 4
        while (1 << bits) < 2 * n:
            bits += 1
        mask = np.int64((1 << bits) - 1)
        shift = np.uint64(64 - bits)
        keys = np.empty(1 << bits, dtype=np.int64)
        slot_id = np.full(1 << bits, -1, dtype=np.int64)
        for i in range(n):
            v = joint[i]
            h = np.int64((np.uint64(v) * np.uint64(11400714819323198485)) >> shift)
            while slot_id[h] >= 0 and keys[h] != v:
                h = (h + 1) & mask
            if slot_id[h] >= 0:
                inv[i] = slot_id[h]
            else:
                keys[h] = v
                slot_id[h] = nc
                inv[i] = nc
                nc += 1
    return inv, nc


@njit(cache=True)
def group_mean_std_count(inverse: np.ndarray, x: np.ndarray, n_groups: int):
    """``(mean, std, count)`` per group code ``0..n_groups-1``: mean and std in the dtype of ``x`` (NaN for a group without a non-NaN row, std NaN below two rows), count as int64."""
    dtype = x.dtype
    sumx = np.zeros(n_groups, dtype=dtype)
    comp = np.zeros(n_groups, dtype=dtype)
    cnt = np.zeros(n_groups, dtype=np.int64)
    run_mean = np.zeros(n_groups, dtype=np.float64)  # Intentional float64: pandas accumulates the variance in float64 whatever the column dtype
    m2 = np.zeros(n_groups, dtype=np.float64)  # Intentional float64: same accumulator
    for i in range(x.shape[0]):
        v = x[i]
        if v == v:
            lab = inverse[i]
            cnt[lab] += 1
            y = v - comp[lab]
            t = sumx[lab] + y
            comp[lab] = t - sumx[lab] - y
            sumx[lab] = t
            old = run_mean[lab]
            run_mean[lab] = old + (v - old) / cnt[lab]
            m2[lab] = m2[lab] + (v - run_mean[lab]) * (v - old)
    mean = np.empty(n_groups, dtype=dtype)
    std = np.empty(n_groups, dtype=dtype)
    for g in range(n_groups):
        if cnt[g] == 0:
            mean[g] = np.nan
            std[g] = np.nan
        else:
            mean[g] = sumx[g] / cnt[g]
            if cnt[g] > 1:
                std[g] = np.sqrt(m2[g] / (cnt[g] - 1))
            else:
                std[g] = np.nan
    return mean, std, cnt


@njit(cache=True)
def seeded_permutation(n: int, seed: int) -> np.ndarray:
    """The Fisher-Yates shuffle of ``range(n)`` seeded ``seed``: the permutation ``b`` of the stream of ``permuted_labels_seeded`` is ``seeded_permutation(n, base_seed + b)``."""
    x = seed_state(seed)
    perm = np.arange(n)
    for i in range(n - 1, 0, -1):
        x = next_state(x)
        j = bounded(x, i + 1)
        t = perm[i]
        perm[i] = perm[j]
        perm[j] = t
    return perm


@njit(parallel=True, cache=True)
def _presort_columns(raw_cont: np.ndarray):
    """Sort order of every column of the (compact, continuous-only) ``raw_cont``, int32, one row per column: 4 bytes per cell, the sorted
    values are gathered per call instead of stored."""
    n = raw_cont.shape[0]
    k = raw_cont.shape[1]
    order = np.zeros((k, n), dtype=np.int32)
    for j in prange(k):
        o = np.argsort(raw_cont[:, j], kind="mergesort")
        for i in range(n):
            order[j, i] = o[i]
    return order


_GATHER_ROW_TILE = 512


@njit(cache=True)
def gather_columns_to_rows(views, out: np.ndarray) -> None:
    """``out[j] = views[j]`` for a typed list of equal-length 1-D arrays (any strides), copied in tiles of rows.

    The columns of a pandas frame built from a C-ordered ``(n, p)`` array are strided views 4p bytes apart; copying them one at a time reads a cache line per
    value. Walking a tile of rows across every column reads the neighbouring columns of each line together (108 columns x 100000 rows: 0.025 s against 0.14 s
    for ``np.stack``)."""
    k = len(views)
    n = out.shape[1]
    for i0 in range(0, n, _GATHER_ROW_TILE):
        i1 = min(n, i0 + _GATHER_ROW_TILE)
        for j in range(k):
            v = views[j]
            for i in range(i0, i1):
                out[j, i] = v[i]


@njit(cache=True)
def radix_order_f32(col: np.ndarray) -> np.ndarray:
    """An order that sorts the float32 vector ``col`` ascending (equal values adjacent; stable), from a 4-pass LSD radix sort of the order-preserving uint32 key
    of each value (``-0.0`` keyed as ``+0.0``, which compare equal). NaN is not handled: the callers pass finite values. Replaces a comparison argsort where only
    the order and the tie runs matter."""
    n = col.shape[0]
    bits = col.view(np.uint32)
    va = np.empty(n, dtype=np.uint32)
    ra = np.empty(n, dtype=np.int64)
    vb = np.empty(n, dtype=np.uint32)
    rb = np.empty(n, dtype=np.int64)
    for t in range(n):
        u = bits[t]
        if u == np.uint32(0x80000000):
            u = np.uint32(0)
        va[t] = ~u if (u & np.uint32(0x80000000)) else (u | np.uint32(0x80000000))
        ra[t] = t
    cnt = np.empty(257, dtype=np.int64)
    for shift in (0, 8, 16, 24):
        cnt[:] = 0
        for t in range(n):
            cnt[((va[t] >> np.uint32(shift)) & np.uint32(0xFF)) + 1] += 1
        for dgt in range(256):
            cnt[dgt + 1] += cnt[dgt]
        for t in range(n):
            dgt = (va[t] >> np.uint32(shift)) & np.uint32(0xFF)
            q = cnt[dgt]
            cnt[dgt] = q + 1
            vb[q] = va[t]
            rb[q] = ra[t]
        va, vb = vb, va
        ra, rb = rb, ra
    return ra


@njit(cache=True)
def radix_order_f64(col: np.ndarray) -> np.ndarray:
    """:func:`radix_order_f32` for a float64 vector: a stable 8-pass LSD radix sort of the order-preserving uint64 keys (``-0.0`` keyed as ``+0.0``); finite input only."""
    n = col.shape[0]
    bits = col.view(np.uint64)
    va = np.empty(n, dtype=np.uint64)
    ra = np.empty(n, dtype=np.int64)
    vb = np.empty(n, dtype=np.uint64)
    rb = np.empty(n, dtype=np.int64)
    sign = np.uint64(0x8000000000000000)
    for t in range(n):
        u = bits[t]
        if u == sign:
            u = np.uint64(0)
        va[t] = ~u if (u & sign) else (u | sign)
        ra[t] = t
    cnt = np.empty(257, dtype=np.int64)
    for shift in (0, 8, 16, 24, 32, 40, 48, 56):
        cnt[:] = 0
        for t in range(n):
            cnt[((va[t] >> np.uint64(shift)) & np.uint64(0xFF)) + 1] += 1
        for dgt in range(256):
            cnt[dgt + 1] += cnt[dgt]
        for t in range(n):
            dgt = (va[t] >> np.uint64(shift)) & np.uint64(0xFF)
            q = cnt[dgt]
            cnt[dgt] = q + 1
            vb[q] = va[t]
            rb[q] = ra[t]
        va, vb = vb, va
        ra, rb = rb, ra
    return ra
