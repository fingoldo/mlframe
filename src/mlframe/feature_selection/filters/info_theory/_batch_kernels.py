"""Parallel batched MI kernels: per-pair MI over an array of variable-index pairs, and the FE-candidate MI + permutation noise-gate."""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


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
        nb_b = int(nbins[b])
        nb_c = int(nbins[c])
        raw_card = int(nbins[a]) * nb_b * nb_c

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
    classes_dense = np.zeros((n, K), dtype=classes_dtype)
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

    for i in range(npermutations):
        # EXACT same per-permutation LCG seed + Fisher-Yates as parallel_mi_prange.
        state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(i + 1)
        local = classes_y_safe.copy()
        for j in range(ny - 1, 0, -1):
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            kk = int(state >> np.uint64(33)) % (j + 1)
            tmp = local[j]
            local[j] = local[kk]
            local[kk] = tmp

        # Score ALL columns against this single shuffled y, in parallel.
        # bench-attempt-rejected (2026-06-07): a ``if nfailed[k] >= max_failed: continue``
        # early-exit here (skip doomed columns' remaining perm MI) is BYTE-IDENTICAL but
        # showed NO wall win on the scene 2407x299 hard-gate (595.75s without -> 641/650s
        # with, across 2 runs on an idle box) -- with the small default npermutations the
        # early-exit rarely fires while the extra per-(perm,col) branch in this hot prange
        # costs more than it saves. Do not re-add without a perm-heavy (high npermutations,
        # low min_nonzero_confidence) workload that actually triggers the cutoff.
        for k in prange(K):
            if original_mi[k] <= 0.0:
                continue
            mi_perm = _relevance_from_dense(
                use_su, classes_dense, k, freqs_dense, int(kx[k]), local, freqs_y, dtype,
            )
            if mi_perm >= original_mi[k]:
                nfailed[k] += 1

    for k in range(K):
        om = original_mi[k]
        if om > 0.0 and nfailed[k] >= max_failed:
            fe_mi[k] = 0.0
        else:
            fe_mi[k] = om

    return fe_mi


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
    joint_counts = np.zeros((K_x, K_y), dtype=dtype)
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
    return 2.0 * mi_xy / denom
