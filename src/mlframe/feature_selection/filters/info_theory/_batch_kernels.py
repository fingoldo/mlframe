"""Parallel batched MI kernels: per-pair MI over an array of variable-index pairs, and the FE-candidate MI + permutation noise-gate."""
from __future__ import annotations

import math
import os
from typing import Callable, cast

import numpy as np
from numba import njit, prange

# Cardinality cap for the per-pair / per-triple joint histogram. ``nbins`` derives from ``data.max(axis=0)+1`` and can be huge for high-cardinality
# categoricals (hash IDs, zip codes), so an ungated ``nb_a*nb_b`` (pair) or ``nb_a*nb_b*nb_c`` (triple) product silently OOMs the per-iteration buffer
# and can overflow the index product. A joint with more cells than there are rows is degenerate (most cells empty, MI dominated by sampling noise), so a
# pair/triple whose RAW cardinality exceeds this cap is skipped and scored MI=0.0 -- the no-information sentinel the FE noise-gate already treats as
# uninformative. 64M int64 cells == 512 MiB per worker thread is the default ceiling; override via ``MLFRAME_BATCH_JOINT_CARD_CAP`` (read by the Python
# callers and threaded down) for hosts that want a different RAM budget. The triple kernel dense-renumbers occupied cells so its WORKING histogram is
# bounded by ``n``, but the ``remap`` table it must allocate is sized by the raw product -- the same OOM hazard, gated identically.
MAX_JOINT_CARDINALITY = 64_000_000


@njit(parallel=True, nogil=True, cache=True)
def batch_pair_mi_prange(
    factors_data: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
) -> np.ndarray:
    """Vectorised batch MI computation over an array of (a, b) variable-index pairs.

    Replaces the ``joblib.Parallel(delayed(compute_pairs_mis)(...))`` outer loop in
    :meth:`MRMR._run_fe_step` with a single numba ``prange`` over all pairs. Each
    iteration:

      1. Builds the joint class encoding for ``(factors_data[:, a], factors_data[:, b])``
         in-place into a thread-local buffer (no Python objects, no dict lookups,
         GIL released throughout).
      2. Counts joint-class frequencies.
      3. Calls :func:`compute_mi_from_classes` with the just-computed marginal.

    Pre-fix (iter-371 1M cb multiclass + MRMR + binary=medium) used joblib loky
    with per-worker mmap of the entire ``factors_data`` array; threading (Layer 1)
    already removed the copy cost, this kernel removes the joblib threadpool
    dispatch overhead AND the Python wrapper GIL contention between numba calls.
    Pair MI now scales bit-for-bit linearly across physical cores.

    Notes:
      * No permutation testing is run here. Callers that need a confidence gate
        should fall back to per-pair ``mi_direct`` afterwards for the surviving
        candidates -- the bench harness shows that's a small fraction of pairs.
      * ``factors_data`` MUST be pre-binned (categorical / ordinal int dtype);
        ``categorize_dataset`` produces this format. Passing a float matrix
        gives undefined results.
      * The joint encoding uses ``a * nbins[b] + b`` which is monotone in
        ``(a, b)`` -- the same convention :func:`merge_vars` uses, so this
        kernel and the legacy path produce numerically identical MIs on the
        same inputs (verified by ``test_batch_pair_mi_prange_matches_merge_vars``).
    """
    n_samples = factors_data.shape[0]
    n_pairs = pair_a.shape[0]
    out = np.empty(n_pairs, dtype=np.float64)
    n_classes_y = freqs_y.shape[0]

    # Wave 47 (2026-05-20): empty factors_data divides by zero in `1/n_samples`
    # inside the per-pair MI inner loop; return zeros (no-information baseline).
    if n_samples == 0:
        out[:] = 0.0
        return out

    for p in prange(n_pairs):
        a = pair_a[p]
        b = pair_b[p]
        nb_a = int(nbins[a])
        nb_b = int(nbins[b])

        # Skip-and-sentinel a pathologically high-cardinality pair: allocating a (joint_card, n_classes_y) histogram for a huge
        # nb_a*nb_b would OOM the worker. MI=0.0 is the no-information value the FE noise-gate treats as uninformative.
        # Test the cap via division BEFORE multiplying: nb_a*nb_b is int64 and would wrap silently on billion-scale
        # cardinalities, producing a small/negative joint_card that passes the cap and then either mis-sizes the
        # histogram (silent corruption) or asks np.zeros for a negative dimension (hard error).
        if nb_a <= 0 or nb_b <= 0 or nb_a > MAX_JOINT_CARDINALITY // nb_b:
            out[p] = 0.0
            continue
        joint_card = nb_a * nb_b

        # Thread-local buffers. numba allocates these per-prange-iteration on the
        # worker thread's stack so there's no cross-thread aliasing.
        joint_counts = np.zeros((joint_card, n_classes_y), dtype=np.int64)
        freqs_x_int = np.zeros(joint_card, dtype=np.int64)

        for i in range(n_samples):
            va = int(factors_data[i, a])
            vb = int(factors_data[i, b])
            cls_x = va * nb_b + vb
            cls_y = int(classes_y[i])
            joint_counts[cls_x, cls_y] += 1
            freqs_x_int[cls_x] += 1

        # Direct MI computation (inlined compute_mi_from_classes core to avoid a
        # cross-kernel call that would require classes_x to be materialised as a
        # 1-D array).
        total = 0.0
        inv_n = 1.0 / n_samples
        for i in range(joint_card):
            fx = freqs_x_int[i]
            if fx == 0:
                continue
            prob_x = fx * inv_n
            for j in range(n_classes_y):
                jc = joint_counts[i, j]
                if jc == 0:
                    continue
                jf = jc * inv_n
                prob_y = freqs_y[j]
                if prob_y > 0.0:
                    total += jf * math.log(jf / (prob_x * prob_y))
        out[p] = total

    return out


@njit(parallel=True, nogil=True, cache=True)
def batch_pair_mi_perm_batched(
    factors_data: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    nbins: np.ndarray,
    y_perms: np.ndarray,
    freqs_y: np.ndarray,
) -> np.ndarray:
    """Permutation-batched pair MI for the maxT null: MI of every pair vs EACH of the ``K`` permuted targets in
    ``y_perms`` (shape ``(K, n)``), returned as ``(K, n_pairs)``. Bit-identical to calling
    :func:`batch_pair_mi_prange` once per shuffle, but the pair-joint encoding ``cls_x = va*nb_b+vb`` and the x-joint
    marginal ``freqs_x`` are INVARIANT under a y-permutation (they depend only on the fixed X columns), so they are
    computed ONCE per pair and reused across all K shuffles; only the ``(joint, y_perm)`` contingency + MI sum re-run
    per shuffle. ~1.71x over the per-shuffle re-encode, exact (max|d|=0)."""
    n = factors_data.shape[0]
    n_pairs = pair_a.shape[0]
    K = y_perms.shape[0]
    n_classes_y = freqs_y.shape[0]
    out = np.empty((K, n_pairs), dtype=np.float64)
    if n == 0:
        out[:, :] = 0.0
        return out
    inv_n = 1.0 / n
    for p in prange(n_pairs):
        a = pair_a[p]
        b = pair_b[p]
        nb_a = int(nbins[a])
        nb_b = int(nbins[b])
        if nb_a <= 0 or nb_b <= 0 or nb_a > MAX_JOINT_CARDINALITY // nb_b:
            for k in range(K):
                out[k, p] = 0.0
            continue
        joint_card = nb_a * nb_b
        cls_x = np.empty(n, dtype=np.int64)  # invariant pair-joint code per row -- built once, reused across shuffles
        freqs_x_int = np.zeros(joint_card, dtype=np.int64)
        for i in range(n):
            c = int(factors_data[i, a]) * nb_b + int(factors_data[i, b])
            cls_x[i] = c
            freqs_x_int[c] += 1
        # Hoist the (joint_card x k_y) histogram alloc out of the K-loop: allocate ONCE per pair and
        # zero-and-reuse across the K shuffles (was K*n_pairs allocs -> n_pairs). ~1.03x, bit-identical
        # (max|d|=0). The scatter itself stays memory-bandwidth-bound -- see bench_pair_maxt_kernel_hoist.py.
        joint_counts = np.empty((joint_card, n_classes_y), dtype=np.int64)
        for k in range(K):
            joint_counts[:, :] = 0
            for i in range(n):
                joint_counts[cls_x[i], int(y_perms[k, i])] += 1
            total = 0.0
            for ci in range(joint_card):
                fx = freqs_x_int[ci]
                if fx == 0:
                    continue
                prob_x = fx * inv_n
                for cj in range(n_classes_y):
                    jc = joint_counts[ci, cj]
                    if jc == 0:
                        continue
                    jf = jc * inv_n
                    prob_y = freqs_y[cj]
                    if prob_y > 0.0:
                        total += jf * math.log(jf / (prob_x * prob_y))
            out[k, p] = total
    return out


@njit(parallel=True, nogil=True, cache=True)
def batch_triple_mi_prange(
    factors_data: np.ndarray,
    triple_a: np.ndarray,
    triple_b: np.ndarray,
    triple_c: np.ndarray,
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
) -> np.ndarray:
    """Vectorised batch JOINT MI ``I((x_a, x_b, x_c); y)`` over an array of (a, b, c) variable-index triples.

    The order-3 twin of :func:`batch_pair_mi_prange` -- the per-shuffle kernel the
    order-3 Westfall-Young maxT permutation-null floor
    (:func:`pooled_triple_permutation_null_joint_mi_floor`) runs to bound the best-of-pool
    chance-max 3-way joint MI, and the same plug-in estimator a 3-way candidate's observed
    joint MI is scored with, so the floor is on the exact scale of the values it gates.

    The naive 3-way joint cardinality ``nb_a*nb_b*nb_c`` blows up the per-triple histogram
    (10^3 = 1000 cells at nbins=10, far beyond the ``n`` distinct rows actually present and
    a memory hazard at higher nbins). This kernel DENSE-RENUMBERS the 3-way joint to its
    OCCUPIED classes (cardinality <= n) before the MI reduction, exactly as ``merge_vars``
    prunes empty joint cells:

      1. Per row build the raw 3-way code ``ra = (va*nb_b + vb)*nb_c + vc`` into a thread-local
         buffer (monotone in (a, b, c), same encoding convention as ``merge_vars``).
      2. Dense-renumber the raw codes via a hash-free direct-address remap table of size
         ``nb_a*nb_b*nb_c`` (zeroed once per triple): first pass marks occupied raw codes,
         a prefix-style scan assigns each occupied raw code a dense id in ``[0, n_dense)``,
         ``n_dense <= n``. (The remap table is the only allocation that scales with the raw
         cardinality; for nbins ~ 10 it is a 1000-int8 scratch, trivially stack/pool-allocated.)
      3. Accumulate the dense joint-with-y counts ``(dense_x, y)`` and the dense-x marginal,
         then the standard plug-in ``sum jf * log(jf / (px * py))`` MI.

    BIT-CONSISTENT with the pair kernel's MI reduction (same ``jf * log(jf / (px * py))``
    accumulation, same float64 ``inv_n`` normalisation), differing only in the 3-way dense
    code construction. ``factors_data`` MUST be the ordinal-encoded (int) screening matrix;
    a float matrix gives undefined results. Returns ``float64[n_triples]``; an empty
    ``factors_data`` yields all-zero (no-information baseline), matching the pair kernel.
    """
    n_samples = factors_data.shape[0]
    n_triples = triple_a.shape[0]
    out = np.empty(n_triples, dtype=np.float64)
    n_classes_y = freqs_y.shape[0]

    if n_samples == 0:
        out[:] = 0.0
        return out

    for p in prange(n_triples):
        a = triple_a[p]
        b = triple_b[p]
        c = triple_c[p]
        nb_a = int(nbins[a])
        nb_b = int(nbins[b])
        nb_c = int(nbins[c])

        # Skip-and-sentinel a pathologically high-cardinality triple: the dense-renumber bounds the WORKING histogram to n_dense<=n, but the
        # direct-address ``remap`` table below is sized by the raw product and is the OOM hazard. MI=0.0 is the gate's uninformative sentinel.
        # Test the cap via staged division BEFORE multiplying: nb_a*nb_b*nb_c is int64 and wraps silently on
        # billion-scale cardinalities, defeating the cap and asking np.full for a negative/overflowed dimension.
        if nb_a <= 0 or nb_b <= 0 or nb_c <= 0 or nb_b > MAX_JOINT_CARDINALITY // nb_c or nb_a > MAX_JOINT_CARDINALITY // (nb_b * nb_c):
            out[p] = 0.0
            continue
        raw_card = nb_a * nb_b * nb_c

        # Thread-local per-row raw 3-way codes + a direct-address remap table.
        raw_codes = np.empty(n_samples, dtype=np.int64)
        remap = np.full(raw_card, -1, dtype=np.int64)  # raw code -> dense id (-1 = unseen)
        n_dense = 0
        for i in range(n_samples):
            va = int(factors_data[i, a])
            vb = int(factors_data[i, b])
            vc = int(factors_data[i, c])
            rc = (va * nb_b + vb) * nb_c + vc
            raw_codes[i] = rc
            if remap[rc] == -1:
                remap[rc] = n_dense
                n_dense += 1

        # Dense joint-with-y counts + dense-x marginal (cardinality n_dense <= n).
        joint_counts = np.zeros((n_dense, n_classes_y), dtype=np.int64)
        freqs_x_int = np.zeros(n_dense, dtype=np.int64)
        for i in range(n_samples):
            dx = remap[raw_codes[i]]
            cy = int(classes_y[i])
            joint_counts[dx, cy] += 1
            freqs_x_int[dx] += 1

        total = 0.0
        inv_n = 1.0 / n_samples
        for i in range(n_dense):
            fx = freqs_x_int[i]
            if fx == 0:
                continue
            prob_x = fx * inv_n
            for j in range(n_classes_y):
                jc = joint_counts[i, j]
                if jc == 0:
                    continue
                jf = jc * inv_n
                prob_y = freqs_y[j]
                if prob_y > 0.0:
                    total += jf * math.log(jf / (prob_x * prob_y))
        out[p] = total

    return out


@njit(nogil=True, cache=True)
def _perm_failcount_col(
    classes_dense: np.ndarray, k: int, freqs_dense: np.ndarray, K_x: int,
    locals_mat: np.ndarray, freqs_y: np.ndarray, n: int, npermutations: int,
    original_mi_k: float, use_su: bool, dtype,
) -> int:
    """Permutation fail-count for ONE densified column ``k`` against all ``npermutations`` pre-shuffled
    ``locals_mat[i]`` y-vectors. BIT-IDENTICAL to looping ``_relevance_from_dense(..., locals_mat[i], ...)``
    and counting ``mi_perm >= original_mi_k`` -- same joint-count increments, same ``jf*log(jf/(px*py))``
    accumulation order -- but (F1/F2, 2026-06-22, cProfile-driven): the column's dense codes are copied to a
    CONTIGUOUS buffer ONCE (unit-stride vs the strided classes_dense[r,k] gather re-read per perm) and ONE
    ``joint`` histogram is reused across perms (re-zeroed) instead of a fresh np.zeros per (perm,col). Called
    from a prange over k, so each thread owns its column -> no cross-thread races (nfk is returned, not shared).
    """
    K_y = len(freqs_y)
    col_codes = np.empty(n, dtype=classes_dense.dtype)
    for r in range(n):
        col_codes[r] = classes_dense[r, k]
    # int64 counter: a joint cell can reach ``n``; int32 (default ``dtype``) wraps negative above ~2.1e9 rows -> NaN MI.
    joint = np.zeros((K_x, K_y), dtype=np.int64)
    inv_n = 1.0 / n
    # SU normalisation denom is perm-invariant (depends only on freqs_dense[k] + freqs_y) -> hoist (same float).
    su_denom = 0.0
    if use_su:
        h_x = 0.0
        for a in range(K_x):
            p = freqs_dense[k, a]
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
        for a in range(K_x):
            for b in range(K_y):
                joint[a, b] = 0
        for r in range(n):
            joint[col_codes[r], locals_mat[i, r]] += 1
        mi_perm = 0.0
        for a in range(K_x):
            prob_x = freqs_dense[k, a]
            for b in range(K_y):
                jc = joint[a, b]
                if jc != 0:
                    jf = jc * inv_n
                    mi_perm += jf * math.log(jf / (prob_x * freqs_y[b]))
        if use_su:
            if mi_perm < 0.0:
                mi_perm = 0.0
            mi_perm = 0.0 if su_denom <= 1e-12 else 2.0 * mi_perm / su_denom
        if mi_perm >= original_mi_k:
            nfk += 1
    return nfk


@njit(parallel=True, nogil=True, cache=True)
def batch_mi_with_noise_gate(
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
    """Batched FE-candidate MI + permutation noise-gate, BIT-IDENTICAL to a per-column
    ``mi_direct`` loop on the default FE path (``parallelism='outer'``, ``n_workers=1``,
    which routes to ``parallel_mi_prange``, ``base_seed=0``).

    For each candidate column ``k`` of ``disc_2d[:, k]`` (ordinal-encoded into
    ``factors_nbins[k]`` bins) this returns ``fe_mi[k]`` -- the plug-in (or SU) MI of the
    densified column against ``y``, ZEROED when the permutation noise-gate rejects it.

    The bit-identity hinges on the per-permutation shuffle in ``parallel_mi_prange`` being
    seeded ONLY by ``(base_seed, i)`` -- never by the candidate's ``classes_x``. So every
    candidate is tested against the SAME ``npermutations`` shuffles of ``y``. This kernel
    exploits that: for each permutation ``i`` it shuffles ``classes_y_safe`` ONCE with the
    identical LCG/Fisher-Yates, then scores ALL ``K`` columns against that single shuffled
    ``y`` -- amortising both the shuffle and the per-permutation MI across the batch.

    Rejection contract (matches ``mi_direct`` non-null path exactly):
      * ``max_failed = max(int(npermutations * (1 - min_nonzero_confidence)), 1)`` when
        ``npermutations > 0``.
      * ``fe_mi[k] = 0.0`` iff ``original_mi[k] > 0`` AND ``npermutations > 0`` AND
        ``nfailed[k] >= max_failed`` (where ``nfailed[k]`` counts permutations whose
        ``mi_perm >= original_mi[k]``). Otherwise ``fe_mi[k] = original_mi[k]``.

    NOTE: ``original_mi[k]`` is computed against ``classes_y`` + ``freqs_y`` (mirroring
    ``mi_direct``'s observed-MI line), while the permutations shuffle ``classes_y_safe``
    (mirroring the ``classes_y_safe if not None else classes_y`` arg ``parallel_mi_prange``
    receives). In normal FE operation those two arrays are value-equal copies.

    Densified columns are materialised into flat padded buffers (``classes_dense`` of
    shape ``(n, K)`` and ``freqs_dense`` of shape ``(K, max_nbins)`` with per-column lengths
    ``kx``) rather than a typed list of arrays, so the parallel-reduction columns stay
    contiguous and prange-safe.
    """
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

    # Per-column densified codes + dense marginals, replicating ``merge_vars`` for a single
    # variable: histogram, prune empty bins, renumber to a dense 0..Kx-1 range.
    # OPT-B (2026-06-07): ``classes_dense`` holds DENSE codes in ``[0, n_dense) <= n_bins``, so it
    # is sized by the narrow ``classes_dtype`` (int8 on the FE path where n_bins ~10) instead of
    # the int32 ``dtype``. This is the SAME (n, K) shape as ``disc_2d`` -- on the scene 2407x64152
    # chunk that is 147 MiB int8 vs 589 MiB int32 (the allocation that OOM'd RAM-tight hosts), and
    # it halves/quarters the bandwidth of the strided ``classes_dense[r, k]`` gathers re-read once
    # per permutation in ``_relevance_from_dense``. BIT-IDENTICAL: the codes are non-negative
    # ordinals only ever READ as histogram indices (``joint_counts[classes_dense[r,k], ...]``), so
    # the narrower storage width does not change a single count; ``joint_counts`` itself stays at
    # the wide ``dtype`` (the actual counter). Default ``classes_dtype=int16`` (2026-06-20): the dense
    # codes are in ``[0, n_dense) <= n_bins``, so int16 holds any realistic nbins (<32768) and HALVES
    # this (n, K) buffer vs the old int32 default for EVERY caller (not just those that opted into int8),
    # value-identical. Callers with a known-narrower disc dtype (int8) still pass it and win further.
    classes_dense: np.ndarray = np.zeros((n, K), dtype=classes_dtype)
    freqs_dense = np.zeros((K, max_nbins), dtype=np.float64)
    kx = np.zeros(K, dtype=np.int64)
    original_mi = np.zeros(K, dtype=np.float64)

    for k in prange(K):
        nb_k = int(factors_nbins[k])
        counts = np.zeros(nb_k, dtype=np.int64)
        col = disc_2d[:, k]
        for r in range(n):
            counts[col[r]] += 1
        # Dense remap: lookup_table[c] = c - (#empty bins below c).
        lookup = np.empty(nb_k, dtype=np.int64)
        nzeros = 0
        for c in range(nb_k):
            if counts[c] == 0:
                nzeros += 1
            lookup[c] = c - nzeros
        n_dense = nb_k - nzeros
        kx[k] = n_dense
        # Write dense codes + dense marginals.
        wpos = 0
        for c in range(nb_k):
            if counts[c] != 0:
                # ``counts[c] / n`` (NOT ``counts[c] * (1/n)``) to be bit-identical to
                # ``merge_vars``' ``freqs / len(factors_data)`` array division.
                freqs_dense[k, wpos] = counts[c] / n
                wpos += 1
        for r in range(n):
            classes_dense[r, k] = lookup[col[r]]

        original_mi[k] = _relevance_from_dense(
            use_su, classes_dense, k, freqs_dense, n_dense, classes_y, freqs_y, dtype,
        )

    if npermutations <= 0:
        for k in range(K):
            fe_mi[k] = original_mi[k]
        return fe_mi

    max_failed = int(npermutations * (1.0 - min_nonzero_confidence))
    if max_failed <= 1:
        max_failed = 1

    nfailed = np.zeros(K, dtype=np.int64)
    ny = classes_y_safe.shape[0]

    # F1 (2026-06-22, cProfile-driven): precompute ALL npermutations shuffled y-vectors ONCE (serial; the
    # EXACT same (base_seed, i)-seeded Fisher-Yates as parallel_mi_prange, just hoisted out of the scoring
    # loop), then prange over COLUMNS with a serial perm-inner loop -- instead of a serial perm-outer loop
    # re-entering an inner prange npermutations times. This removes npermutations-1 fork/join barriers and
    # keeps all cores saturated regardless of K. Bit-identical: the shuffle arithmetic/order is unchanged;
    # nfailed[k] is an integer count of mi_perm>=original_mi[k], so reordering the (i,k) iteration leaves
    # each per-column count identical; each prange thread owns its own k (writes nfailed[k] privately).
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

    for k in prange(K):
        if original_mi[k] <= 0.0:
            continue
        nfailed[k] = _perm_failcount_col(
            classes_dense, k, freqs_dense, int(kx[k]), locals_mat, freqs_y, n,
            npermutations, original_mi[k], use_su, dtype,
        )

    for k in range(K):
        om = original_mi[k]
        if om > 0.0 and nfailed[k] >= max_failed:
            fe_mi[k] = 0.0
        else:
            fe_mi[k] = om

    return fe_mi


@njit(parallel=True, nogil=True, cache=True)
def batch_mi_with_noise_gate_v2(
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
    """FUSED-OBSERVED-MI twin of :func:`batch_mi_with_noise_gate` (F2, 2026-06-22). BIT-IDENTICAL output
    (same dense codes, same observed MI, same permutation noise-gate, same shuffle stream) -- the ONLY
    change is that the per-column observed-MI pass writes the dense codes AND accumulates the joint with y
    in ONE n-row loop (``_densify_and_relevance_fused``) instead of writing the dense codes then re-reading
    them in a SECOND strided pass (the legacy ``classes_dense[r,k]=...`` loop + ``_relevance_from_dense``).
    The observed-MI sweep is the dominant CPU FE cost at the canonical 30k-subsample chunk (measured ~5.8s
    of the ~7s per chunk EVEN with npermutations=0), so collapsing its double column-pass to a single pass
    is the largest selection-safe lever. The permutation path below is verbatim from the original kernel
    (it re-reads ``classes_dense`` per shuffle, which the fused pass still fully populates). Dispatched via
    :func:`select_batch_mi_kernel` (KTC-gated); the original kernel remains the fallback / small-size path."""
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

    classes_dense: np.ndarray = np.zeros((n, K), dtype=classes_dtype)
    freqs_dense = np.zeros((K, max_nbins), dtype=np.float64)
    kx = np.zeros(K, dtype=np.int64)
    original_mi = np.zeros(K, dtype=np.float64)

    for k in prange(K):
        nb_k = int(factors_nbins[k])
        counts = np.zeros(nb_k, dtype=np.int64)
        col = disc_2d[:, k]
        for r in range(n):
            counts[col[r]] += 1
        lookup = np.empty(nb_k, dtype=np.int64)
        nzeros = 0
        for c in range(nb_k):
            if counts[c] == 0:
                nzeros += 1
            lookup[c] = c - nzeros
        n_dense = nb_k - nzeros
        kx[k] = n_dense
        wpos = 0
        for c in range(nb_k):
            if counts[c] != 0:
                freqs_dense[k, wpos] = counts[c] / n
                wpos += 1
        # FUSED: write classes_dense[:, k] AND accumulate the observed-MI joint in ONE pass.
        original_mi[k] = _densify_and_relevance_fused(
            use_su, col, lookup, classes_dense, k, freqs_dense, n_dense, classes_y, freqs_y, dtype,
        )

    if npermutations <= 0:
        for k in range(K):
            fe_mi[k] = original_mi[k]
        return fe_mi

    max_failed = int(npermutations * (1.0 - min_nonzero_confidence))
    if max_failed <= 1:
        max_failed = 1

    nfailed = np.zeros(K, dtype=np.int64)
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

    for k in prange(K):
        if original_mi[k] <= 0.0:
            continue
        nfailed[k] = _perm_failcount_col(
            classes_dense, k, freqs_dense, int(kx[k]), locals_mat, freqs_y, n,
            npermutations, original_mi[k], use_su, dtype,
        )

    for k in range(K):
        om = original_mi[k]
        if om > 0.0 and nfailed[k] >= max_failed:
            fe_mi[k] = 0.0
        else:
            fe_mi[k] = om

    return fe_mi


@njit(nogil=True, cache=True)
def _densify_and_relevance_fused(
    use_su: bool,
    col: np.ndarray,
    lookup: np.ndarray,
    classes_dense: np.ndarray,
    k: int,
    freqs_dense: np.ndarray,
    K_x: int,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    dtype,
) -> float:
    """FUSED dense-code WRITE + observed-MI of column ``k`` in ONE pass over the n rows (F2, 2026-06-22,
    cProfile-driven). The legacy path made TWO strided full-column passes per candidate: one to write
    ``classes_dense[r,k] = lookup[col[r]]`` and a SECOND inside ``_relevance_from_dense`` to re-read
    ``classes_dense[r,k]`` + ``classes_y[r]`` into the joint histogram. At the canonical 30k-subsample
    K~3888 chunk that observed-MI sweep is the dominant CPU FE cost (~5.8s/chunk even with npermutations=0,
    i.e. the permutation shuffles are NOT the cost -- this densify/relevance double-pass is). Fusing them
    accumulates the (dense_x, y) joint WHILE writing the dense code, so the n-row column is touched once
    here instead of twice. BIT-IDENTICAL to ``classes_dense[r,k]=lookup[col[r]]`` followed by
    ``_relevance_from_dense(...)``: the dense codes written are the same values, the joint counts are the
    same increments, and the ``jf*log(jf/(px*py))`` reduction order over the (K_x, K_y) histogram is
    unchanged. ``classes_dense`` is still fully written (the permutation path re-reads it per shuffle)."""
    n = col.shape[0]
    K_y = len(freqs_y)
    # int64 counter: a joint cell can reach ``n``; int32 (default ``dtype``) wraps negative above ~2.1e9 rows -> NaN MI.
    joint_counts = np.zeros((K_x, K_y), dtype=np.int64)
    for r in range(n):
        dc = lookup[col[r]]
        classes_dense[r, k] = dc
        joint_counts[dc, classes_y[r]] += 1
    inv_n = 1.0 / n

    mi_xy = 0.0
    for i in range(K_x):
        prob_x = freqs_dense[k, i]
        for j in range(K_y):
            jc = joint_counts[i, j]
            if jc != 0:
                prob_y = freqs_y[j]
                jf = jc * inv_n
                mi_xy += jf * math.log(jf / (prob_x * prob_y))

    if not use_su:
        return mi_xy

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
        return 0.0
    # Floor the plug-in numerator at 0: float round-off on near-deterministic columns can leave ``mi_xy`` slightly negative -> tiny negative SU treated as valid low relevance instead of 0.
    if mi_xy < 0.0:
        mi_xy = 0.0
    return 2.0 * mi_xy / denom


@njit(nogil=True, cache=True)
def _relevance_from_dense(
    use_su: bool,
    classes_dense: np.ndarray,
    k: int,
    freqs_dense: np.ndarray,
    K_x: int,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    dtype,
) -> float:
    """MI (or SU) of densified column ``k`` against ``y`` -- inlined twin of
    ``compute_relevance_score`` reading the column from the padded ``classes_dense`` /
    ``freqs_dense`` buffers. Numerically identical to calling
    ``compute_relevance_score(use_su, classes_dense[:, k], freqs_dense[k, :K_x], ...)``:
    same indexed joint-count pass, same ``jf * log(jf / (px * py))`` accumulation order.
    """
    n = classes_dense.shape[0]
    K_y = len(freqs_y)
    # bench-attempt-rejected (2026-06-07): REUSE a thread-local joint_counts scratch
    # (indexed by numba.get_thread_id()) instead of this per-call ``np.zeros`` to remove
    # the K*(nperm+1) allocations (Q5). BYTE-IDENTICAL but a NET LOSS: 0.58x (K=300) ..
    # 0.77x (K=4000) on the scene-like CPU gate. numba pool/stack-allocates this tiny
    # (K_x x K_y ~ 10x2) array essentially for free; a shared (nthreads, max_Kx*K_y)
    # scratch adds get_thread_id() + a manual zeroing loop + flat-index arithmetic +
    # cross-thread false sharing that all cost MORE than the elided alloc. Keep the
    # per-call np.zeros. (proto D:/Temp/q5_scratch_proto.py)
    # int64 counter: a joint cell can reach ``n``; int32 (default ``dtype``) wraps negative above ~2.1e9 rows -> NaN MI.
    joint_counts = np.zeros((K_x, K_y), dtype=np.int64)
    for r in range(n):
        joint_counts[classes_dense[r, k], classes_y[r]] += 1
    inv_n = 1.0 / n

    mi_xy = 0.0
    for i in range(K_x):
        prob_x = freqs_dense[k, i]
        for j in range(K_y):
            jc = joint_counts[i, j]
            if jc != 0:
                prob_y = freqs_y[j]
                jf = jc * inv_n
                mi_xy += jf * math.log(jf / (prob_x * prob_y))

    if not use_su:
        return mi_xy

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
        return 0.0
    # Floor the plug-in numerator at 0: float round-off on near-deterministic columns can leave ``mi_xy`` slightly negative -> tiny negative SU treated as valid low relevance instead of 0.
    if mi_xy < 0.0:
        mi_xy = 0.0
    return 2.0 * mi_xy / denom


# ---- batched FE-candidate MI kernel selector (F2 fused observed-MI dispatch, 2026-06-22) -------------
# ``batch_mi_with_noise_gate_v2`` fuses the per-column dense-code write with the observed-MI joint
# accumulation (one n-row pass instead of two), measured 1.18-1.21x faster than the original kernel at the
# canonical 30k-subsample K~3888 chunk and BIT-IDENTICAL (maxdiff 0.0 on the observed MI AND the
# noise-gated output, both npermutations=0 and =25). It is structurally never slower (it removes a full
# strided column re-read), so it is the default. Per the repo "keep all kernel versions; dispatch by
# size via kernel_tuning_cache" rule the ORIGINAL kernel is retained and the choice is routed per-host
# through the KTC (axes n_rows x n_cols); the v2 fused path is the measurement-backed fallback for an
# un-tuned host (the safe faster default here). ``MLFRAME_BATCH_MI_KERNEL=v1|v2`` force-overrides.
_BATCH_MI_KERNEL_CODE_VERSION = "batch_mi_noise_gate_kernel-v2-fused-2026-06-22"


def _batch_mi_kernel_fallback_choice(n_rows: int, n_cols: int) -> str:
    """Pre-sweep default kernel for the batched FE-candidate MI: the fused v2 (measured uniformly faster
    + bit-identical). The KTC sweep can later refine per-host if a pathological size ever regresses."""
    return "v2"


def select_batch_mi_kernel(n_rows: int, n_cols: int) -> Callable:
    """Return the batched FE-candidate MI kernel (``batch_mi_with_noise_gate`` v1 or the fused v2) for this
    (n_rows, n_cols) on this host. Routed through ``pyutilz.system.kernel_tuning_cache`` (per-host cache,
    code-version checked, async background sweep, measurement-backed fallback) -- NOT a hardcoded threshold.
    ``MLFRAME_BATCH_MI_KERNEL`` env (``v1``/``v2``) force-overrides for A/B + safety. Defaults to v2 (the
    fused, measured-faster, bit-identical kernel) on any miss/failure so an un-tuned host gets the win."""
    _env = os.environ.get("MLFRAME_BATCH_MI_KERNEL", "").strip().lower()
    if _env == "v1":
        return cast(Callable, batch_mi_with_noise_gate)
    if _env == "v2":
        return cast(Callable, batch_mi_with_noise_gate_v2)
    choice = "v2"
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache
        res = KernelTuningCache.load_or_create().get_or_tune(
            "batch_mi_fe_kernel",
            dims={"n_rows": int(n_rows), "n_cols": int(n_cols)},
            tuner=_run_batch_mi_kernel_sweep,
            axes=["n_rows", "n_cols"],
            fallback={"kernel_choice": _batch_mi_kernel_fallback_choice(int(n_rows), int(n_cols))},
            code_version=_BATCH_MI_KERNEL_CODE_VERSION,
            async_sweep=True,
        )
        if isinstance(res, str):
            choice = res
        elif res:
            choice = str(res.get("kernel_choice", "v2"))
    except Exception:
        choice = "v2"
    return cast(Callable, batch_mi_with_noise_gate if choice == "v1" else batch_mi_with_noise_gate_v2)


def _run_batch_mi_kernel_sweep():
    """Per-host v1-vs-v2 crossover sweep for the batched FE-candidate MI kernel -> kernel_choice regions
    keyed on (n_rows, n_cols). Both kernels are bit-identical so equivalence holds at a tight tol; the
    sweep ranks by wall only. Returns [] when the benchmarking helper is unavailable (-> fallback v2)."""
    try:
        from pyutilz.dev.benchmarking import sweep_backend_grid
        from ..discretization import discretize_2d_quantile_batch
    except Exception:
        return []

    def _make_inputs(dims):
        """Build a synthetic (discretized candidates, y-classes, y-frequencies) fixture at the requested (n_rows, n_cols) grid point, using a shared a**2/b-derived base signal so v1/v2 see realistic non-degenerate MI."""
        n = int(dims["n_rows"]); K = int(dims["n_cols"]); nbins = 10
        rng = np.random.default_rng(0)
        a = rng.uniform(0.1, 1.1, n); b = rng.uniform(0.1, 1.1, n); base = a ** 2 / b
        cand = np.empty((n, K), dtype=np.float32)
        for k in range(K):
            cand[:, k] = (base * (1.0 + 0.01 * rng.standard_normal(n)) + 0.001 * k).astype(np.float32)
        np.nan_to_num(cand, copy=False)
        disc = discretize_2d_quantile_batch(cand, n_bins=nbins, dtype=np.int8, assume_finite=True)
        y = a**2 / b
        edges = np.quantile(y, np.linspace(0, 1, nbins + 1)[1:-1])
        yc = np.searchsorted(edges, y).astype(np.int64)
        fy = np.bincount(yc, minlength=int(yc.max()) + 1).astype(np.float64) / n
        return (disc, np.full(K, nbins, dtype=np.int64), yc, fy)

    def _call(kernel):
        """Bind a fixed permutation-MI call signature to ``kernel`` (v1 or v2) so ``sweep_backend_grid`` can invoke both with identical positional data args."""
        def _f(disc, fn, yc, fy):
            """Invoke the bound kernel on one sweep-generated fixture with a fixed permutation/seed/threshold configuration."""
            return kernel(
                disc_2d=disc, factors_nbins=fn, classes_y=yc, classes_y_safe=yc, freqs_y=fy,
                npermutations=25, base_seed=np.uint64(0), min_nonzero_confidence=0.0, use_su=False,
                dtype=np.int32, classes_dtype=np.int8,
            )
        return _f

    try:
        return sweep_backend_grid(
            {"v1": _call(batch_mi_with_noise_gate), "v2": _call(batch_mi_with_noise_gate_v2)},
            {"n_rows": [30_000, 100_000], "n_cols": [600, 3888]},
            _make_inputs,
            reference="v1", repeats=3, equiv_rtol=1e-12, equiv_atol=1e-15,
            result_key="kernel_choice",
        )
    except TypeError:
        return sweep_backend_grid(
            {"v1": _call(batch_mi_with_noise_gate), "v2": _call(batch_mi_with_noise_gate_v2)},
            {"n_rows": [30_000, 100_000], "n_cols": [600, 3888]},
            _make_inputs,
            reference="v1", repeats=3, equiv_rtol=1e-12, equiv_atol=1e-15,
        )
