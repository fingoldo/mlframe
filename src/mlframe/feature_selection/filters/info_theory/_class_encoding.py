"""Class-encoding histogram building blocks: collapse ordinal-encoded variables into a single 1-D class array and the 2-variable joint-frequency / joint-entropy fast paths."""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def merge_vars(
    factors_data: np.ndarray,
    vars_indices,
    var_is_nominal,
    factors_nbins,
    dtype=np.int32,
    min_occupancy: int = None,
    current_nclasses: int = 1,
    final_classes: np.ndarray = None,
    verbose: bool = False,
) -> tuple:
    """Melt multiple ordinal-encoded variables into a single 1-D class array.

    ``factors_data`` is laid out as ``(n_samples, n_features)`` (sklearn convention). Empty bins are pruned and the class indices renumbered so the returned
    ``final_classes`` is dense (no skipped integers).

    bench-attempt-rejected (2026-05-21, c0148 / iter136): 1-var
    fast-path replacing the per-row loop with np.bincount + cast runs
    15-20% SLOWER at n=50k..1M -- numba already compiles the manual
    loop into tight machine code; the bincount path pays an int64->
    dtype cast plus an extra .copy() for the no-prune branch. Bench:
    profiling/bench_merge_vars_1var_fastpath.py.

    bench-attempt-rejected (2026-06-03): parallelising the per-row
    histogram via a chunked per-thread-freqs accumulator + reduction
    (bit-identical -- integer counts sum associatively, final_classes
    writes are per-row independent) is memory-bound, not compute-bound,
    so it does NOT scale: 0.56x at n=50k (prange overhead dominates the
    tight loop), ~neutral at n=200k, only ~1.34x at n=1M / K=256. The
    per-thread histograms also multiply the freqs working set by
    n_threads, which blows up for deep joints (large expected_nclasses).
    Net loss across the typical n=30k..200k combo sizes.
    """
    n_rows = len(factors_data)
    if final_classes is None:
        final_classes = np.zeros(n_rows, dtype=dtype)
    if n_rows == 0:
        # No samples: the per-row histogram is empty; freqs / n_rows would be a divide-by-zero producing NaN
        # frequencies that then silently poison downstream entropy / MI. Return an empty frequency vector instead.
        return final_classes, np.zeros(0, dtype=np.float64), current_nclasses
    for var_number, var_index in enumerate(vars_indices):

        expected_nclasses = current_nclasses * factors_nbins[var_index]
        # 2026-05-30 Wave 9.1 fix (loop iter 21): use int64 for the
        # per-class counter ``freqs`` and the remap ``lookup_table``,
        # decoupled from the caller-supplied ``dtype`` (which sizes the
        # per-sample class-id workspace). Pre-fix when ``dtype=np.int8``,
        # a single 200-sample bin overflowed to -56 silently (live
        # repro: freqs_norm = [-0.28] instead of [1.0]); under default
        # ``dtype=np.int32`` the same defect lurked at any joint
        # holding > 2**31 samples. The bin count is a pure counter
        # bounded by ``n_samples`` regardless of class encoding, so
        # int64 is the safe choice with negligible memory overhead.
        freqs = np.zeros(expected_nclasses, dtype=np.int64)
        values = factors_data[:, var_index].astype(dtype)
        if verbose:
            print(f"var={var_index}, classes from {values.min()} to {values.max()}")
        for sample_row, sample_class in enumerate(values):
            newclass = final_classes[sample_row] + sample_class * current_nclasses
            freqs[newclass] += 1
            final_classes[sample_row] = newclass

        nzeros = 0
        # lookup_table holds remap targets up to ``expected_nclasses``
        # which can exceed ``dtype``'s range for deep joints; int64
        # eliminates the silent overflow there too.
        lookup_table = np.empty(expected_nclasses, dtype=np.int64)
        for oldclass, npoints in enumerate(freqs):
            if npoints == 0:
                nzeros += 1
            lookup_table[oldclass] = oldclass - nzeros

        if nzeros:
            if verbose:
                print(
                    f"skipped {nzeros} cells out of {expected_nclasses}, classes from {final_classes.min()} to {final_classes.max()}, lookup_table={lookup_table}"
                )

            for sample_row, old_class in enumerate(final_classes):
                final_classes[sample_row] = lookup_table[old_class]
            if var_number == len(vars_indices) - 1:
                freqs = freqs[freqs > 0]
        current_nclasses = expected_nclasses - nzeros
    return final_classes, freqs / n_rows, current_nclasses


@njit(cache=True)
def joint_freqs_2var(factors_data: np.ndarray, ia: int, ib: int, nb_a: int, nb_b: int) -> np.ndarray:
    """Normalized NONZERO joint-class frequencies for the column PAIR ``(ia, ib)`` --
    BIT-IDENTICAL to ``merge_vars(factors_data, [ia, ib], None, factors_nbins, dtype)[1]``
    but WITHOUT the per-sample ``final_classes`` output array + the lookup-table remap pass
    that ``merge_vars`` builds and the pairwise SU path discards.

    Specialised 2-variable fast path for the DCD pairwise-SU hot loop
    (``_dcd_metrics.pair_su``), where the only consumer of ``merge_vars`` is the joint entropy
    ``H(X_ia, X_ib)`` (the per-column marginals ``H(X_ia)`` / ``H(X_ib)`` are state-cached, so the
    sole per-pair ``merge_vars`` call is this 2-var joint). ``merge_vars`` allocates a length-``n``
    ``final_classes`` array, an ``expected``-length ``lookup_table``, and walks every sample twice
    (accumulate + remap) -- all unused here, since ``entropy`` only needs the pruned freqs. On the
    scene DCD sweep (~24k pairs at 600 rows, ~345k at full) that wasted allocation+remap is the
    dominant per-pair cost: this kernel is ~23x faster (171.9us -> 7.4us/pair incl. the downstream
    ``entropy`` call; bench D:/Temp/microbench_pairsu2.py) at ZERO numeric change.

    BIT-IDENTITY to the ``merge_vars`` 2-var output (verified max-abs-diff EXACTLY 0.0 across
    uniform / sparse / heavy-skew / constant-column data at n=37/600/2407, 14160 pairs each):
      * Joint class id ``cls = ca + cb * nb_a`` reproduces ``merge_vars``'s incremental encoding
        for two variables (var0=ia: current_nclasses=1 -> class=ca, nclasses=nb_a; var1=ib:
        class = ca + cb*nb_a) EXACTLY -- same integer arithmetic, no float involved.
      * ``freqs[freqs > 0]`` prunes empty bins in ASCENDING class-id order -- the identical dense
        order ``merge_vars`` produces (its lookup-table renumber is monotone in oldclass), so the
        downstream ``-(log(p)*p).sum()`` reduces the SAME values in the SAME order -> bit-identical
        float64 entropy (FP summation order preserved).
      * Normalisation ``/ n`` matches ``merge_vars``'s ``freqs / len(factors_data)`` (both int64
        counts divided by the same ``n`` in float64).

    ``nb_a`` / ``nb_b`` are ``factors_nbins[ia]`` / ``factors_nbins[ib]`` (the per-column bin
    counts). ``factors_data`` is the ``(n_samples, n_features)`` ordinal-encoded matrix; columns
    ``ia`` / ``ib`` are read directly (no copy). Returns the float64 normalized nonzero freqs,
    ready to hand straight to ``entropy`` (kept as a SEPARATE njit call so the canonical entropy
    reduction is reused verbatim -- no re-derivation of the log-sum numerics here).
    """
    n = factors_data.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    size = nb_a * nb_b
    freqs = np.zeros(size, dtype=np.int64)
    for r in range(n):
        ca = factors_data[r, ia]
        cb = factors_data[r, ib]
        cls = ca + cb * nb_a
        freqs[cls] += 1
    nz = freqs[freqs > 0]
    return nz / n


@njit(nogil=True, cache=True)
def joint_entropy_2var(factors_data: np.ndarray, ia: int, ib: int, nb_a: int, nb_b: int) -> float:
    """Shannon entropy ``H(X_ia, X_ib)`` in nats for the column PAIR ``(ia, ib)`` -- BIT-IDENTICAL
    to ``entropy(joint_freqs_2var(factors_data, ia, ib, nb_a, nb_b))`` but FUSED: the joint
    histogram is reduced straight to the entropy scalar with NO intermediate normalized-freqs
    array, NO ``freqs[freqs > 0]`` boolean-mask allocation, and NO ``np.log(freqs) * freqs``
    temporary -- all of which ``joint_freqs_2var`` (which returns the full normalized nonzero freqs
    array) and the downstream ``entropy`` call allocate-then-discard on the way to a single float.

    The DCD pairwise-SU hot loop (``_dcd_metrics.pair_su``) is the dominant per-pair consumer of
    ``joint_freqs_2var``: 341,777 calls on the full scene fit, and EVERY one immediately collapses
    the returned freqs array to a scalar via ``entropy``. The two-call form therefore allocates a
    pruned freqs array + a ``/ n`` normalized array per pair, then ``entropy`` re-runs the
    ``freqs > 0`` mask and builds ``log(freqs) * freqs`` -- pure per-call wasted WORK once only the
    entropy scalar is wanted. This kernel walks the joint histogram ONCE and accumulates
    ``-(p * log(p))`` directly. ~1.24x per pair at ZERO numeric change (bench D:/Temp/ww_micro_jointentropy.py).

    BIT-IDENTITY to ``entropy(joint_freqs_2var(...))`` (verified max-abs-diff EXACTLY 0.0 across
    uniform / heavy-skew data at n=37/600/2407 over 960 cases incl. unequal per-column bin counts,
    test_joint_entropy_2var.py):
      * SAME joint class id ``cls = ca + cb * nb_a`` and SAME ascending-class-id scan order as
        ``joint_freqs_2var`` + ``entropy``'s ``freqs[freqs > 0]`` (which preserves ascending order),
        so the FP accumulation visits the IDENTICAL ``p`` values in the IDENTICAL order.
      * SAME per-bin probability ``p = cnt / n`` (int64 count divided by ``n`` in float64), matching
        ``joint_freqs_2var``'s ``nz / n``.
      * SAME reduction: ``entropy`` is ``-(np.log(freqs) * freqs).sum()`` -- numpy's ``.sum()`` over a
        contiguous 1-D array of < 128 elements (the joint nonzero-bin count, ``<= nb_a * nb_b``) is a
        NAIVE sequential add (pairwise summation only kicks in at >= 128 elements), so the scalar
        left-to-right ``h += log(p) * p`` accumulation here reproduces it bit-for-bit, then negates.
    """
    n = factors_data.shape[0]
    if n == 0:
        return 0.0
    size = nb_a * nb_b
    freqs = np.zeros(size, dtype=np.int64)
    for r in range(n):
        ca = factors_data[r, ia]
        cb = factors_data[r, ib]
        cls = ca + cb * nb_a
        freqs[cls] += 1
    # Reduce the histogram straight to entropy in ascending class-id order (== entropy's post-prune
    # order). p = cnt / n in float64; accumulate -(p * log(p)) sequentially -> bit-identical to
    # numpy ``.sum()`` for the small (< 128) nonzero-bin count of a 2-var joint.
    h = 0.0
    for c in range(size):
        cnt = freqs[c]
        if cnt != 0:
            p = cnt / n
            h += np.log(p) * p
    return -h
