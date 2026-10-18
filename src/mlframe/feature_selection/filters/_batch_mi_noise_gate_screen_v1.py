"""Noise-gate GPU leaf helpers: host-code range screen + the v1 cupy noise gate (carve sibling of ``batch_mi_noise_gate_gpu``).

Moved VERBATIM; ``batch_mi_noise_gate_gpu`` re-exports every name here. This module must not import ``batch_mi_noise_gate_gpu``.
"""

from __future__ import annotations

import numpy as np
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype

from ._batch_mi_noise_gate_kernels import (
    cupy,
    cupy_available,
    _cupy_bincount_known_size,
    _mi_columns_from_counts_cpu,
    _fisher_yates_shuffle,
    _gate_from_mi,
)


def _screen_host_codes(disc_2d, factors_nbins, classes_y, freqs_y, who):
    """Raise ``ValueError`` unless every column ``k`` of ``disc_2d`` holds codes in ``[0, factors_nbins[k])`` and ``classes_y`` lies in ``[0, K_y)``.

    The histogram kernels (and the flat ``bincount`` index of the cupy twins) place a code at ``offset[k] + code * K_y + y``, so a code at or above its own
    column's cardinality lands in the NEXT column's slice (silent wrong MI) or past the buffer (illegal address). A global ``max(nbins)`` bound misses that
    for mixed-cardinality frames, hence the per-column comparison.
    """
    if disc_2d.size:
        nb = np.asarray(factors_nbins, dtype=np.int64)
        lo = disc_2d.min(axis=0).astype(np.int64)
        hi = disc_2d.max(axis=0).astype(np.int64)
        bad = np.flatnonzero((lo < 0) | (hi >= nb))
        if bad.size:
            c = int(bad[0])
            raise ValueError(
                "%s disc_2d codes out of range in column %d (min=%d, max=%d) for nbins=%d; a -1 sentinel or over-range code would index "
                "outside its histogram slice." % (who, c, int(lo[c]), int(hi[c]), int(nb[c]))
            )
    if classes_y.size:
        cy_min, cy_max, k_y = int(classes_y.min()), int(classes_y.max()), int(freqs_y.shape[0])
        if cy_min < 0 or cy_max >= k_y:
            raise ValueError("%s classes_y out of range (min=%d, max=%d) for K_y=%d (illegal address)." % (who, cy_min, cy_max, k_y))


def _bin_range_for_host_discrete(d_disc_resident, disc_2d, factors_nbins, classes_y, freqs_y):
    """Screen the host codes unless a resident device copy (binner-produced, dense by contract) is the one consumed."""
    if d_disc_resident is None:
        _screen_host_codes(disc_2d, factors_nbins, classes_y, freqs_y, "batch_mi_with_noise_gate_cuda_resident")


def batch_mi_with_noise_gate_cupy_v1(
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
) -> np.ndarray:
    """CuPy GPU twin of ``batch_mi_with_noise_gate``. BIT-IDENTICAL.

    PRIOR (v1) kernel - kept per the keep-all-kernel-versions rule so it can be
    re-benched on bigger GPUs. SUPERSEDED on small/consumer cards by
    ``batch_mi_with_noise_gate_cupy`` (GPU-resident, O(1) transfers per batch).

    bench-note (GTX 1050 Ti, 4GB Pascal cc6.1, n=2407 K=2000 nperm=3): this v1
    does 1 H2D + 1 D2H PER permutation (cp.asarray(shuffled) + cp.asnumpy(flat)),
    so a self-profile of MRMR.fit spent 82.7% wall-time here at ~18% GPU-util -
    transfer/sync-bound. The resident rewrite collapses that to O(1) transfers.

    GPU work: for the original y and each CPU-shuffled y, compute the joint
    histogram ``(nbins_k, K_y)`` for every candidate column via a single batched
    ``cupy.bincount`` over the flattened ``(col_code * K_y + y_code)`` index. The
    MI is reduced from those INTEGER counts on the CPU via ``_mi_from_counts_cpu``
    (bit-exact). Raises ``RuntimeError`` if cupy is unavailable.
    """
    if not cupy_available():
        raise RuntimeError("cupy is not available on this host")
    cp = cupy()

    n = int(disc_2d.shape[0])
    K = int(disc_2d.shape[1])
    fe_mi = np.zeros(K, dtype=_crit_np_dtype())
    if K == 0 or n == 0:
        return fe_mi
    _screen_host_codes(disc_2d, factors_nbins, classes_y, freqs_y, "batch_mi_with_noise_gate_cupy_v1")

    K_y = int(freqs_y.shape[0])
    nbins_arr = np.asarray(factors_nbins, dtype=np.int64)

    # Move the discretized frame to device ONCE (int32). The per-column joint
    # code is ``col_code * K_y + y_code``; flattening the whole frame lets a
    # single bincount per shuffled-y collect all columns at once.
    d_disc = cp.asarray(np.ascontiguousarray(disc_2d, dtype=np.int32))  # (n, K)
    # Column-block offsets so column k occupies counts[k] = bins [off[k], off[k]+nb_k*K_y).
    per_col_size = nbins_arr * K_y  # (K,) joint size per column
    offsets = np.zeros(K + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(per_col_size)
    total_size = int(offsets[K])
    # PER-CELL INDEX dtype (parallel to the resident kernel): v1 scores ONE y at a time, so
    # d_base / d_idx are bounded by ``total_size - 1`` - int32 HALVES these (n, K) buffers vs int64 when
    # that fits (gated -> bit-identical, the bincount indices are the same values). Kept for re-bench
    # parity with the resident kernel.
    idx_dtype = cp.int32 if total_size <= 2_147_483_647 else cp.int64
    d_offsets = cp.asarray(offsets[:K].reshape(1, K)).astype(idx_dtype)  # (1, K) broadcastable

    # Column index of each cell, broadcast (n, K).
    # global_index[r, k] = offsets[k] + disc[r,k]*K_y + y_code[r]
    # We build the per-(r,k) base ``offsets[k] + disc[r,k]*K_y`` once, then add
    # the y-code (which changes per shuffle) before each bincount.
    d_base = d_offsets + d_disc.astype(idx_dtype) * cp.asarray(K_y, dtype=idx_dtype)  # (n, K) idx_dtype

    def _joint_counts_for(y_codes_host: np.ndarray) -> np.ndarray:
        """Return a host list of (nbins_k, K_y) int64 count matrices for all K cols
        against ``y_codes_host`` (length n)."""
        d_y = cp.asarray(np.ascontiguousarray(y_codes_host, dtype=np.int64)).reshape(n, 1).astype(idx_dtype)
        d_idx = (d_base + d_y).reshape(-1)  # (n*K,) flat global indices (idx_dtype; <= total_size-1)
        # OPT-D: known-size bincount (skip cupy.bincount's host-sync validations);
        # byte-identical counts. ``total_size`` is exact, ``d_idx`` non-negative by construction.
        flat = _cupy_bincount_known_size(d_idx, total_size)
        return np.asarray(cp.asnumpy(flat))  # (total_size,) int64 on host

    # Original MI. Per-column MI reduction batched into one njit call (bit-identical).
    _col_off = offsets[:K]
    _all_pos = np.ones(K, dtype=_crit_np_dtype())
    counts_orig = _joint_counts_for(np.asarray(classes_y, dtype=np.int64))
    original_mi = _mi_columns_from_counts_cpu(counts_orig, _col_off, nbins_arr, K_y, freqs_y, n, use_su, _all_pos)

    if npermutations <= 0:
        return _gate_from_mi(original_mi, [], 0, min_nonzero_confidence)

    cy_safe = np.asarray(classes_y_safe)
    # bench-attempt-rejected (2026-06-07): perm-reduction early-exit (see cupy-v2 path) -
    # BYTE-IDENTICAL but no scene wall win; keep full reduction + _gate_from_mi.
    perm_mis = []
    for i in range(npermutations):
        shuffled = _fisher_yates_shuffle(cy_safe, np.uint64(base_seed), i)
        counts_p = _joint_counts_for(np.asarray(shuffled, dtype=np.int64))
        mp = _mi_columns_from_counts_cpu(counts_p, _col_off, nbins_arr, K_y, freqs_y, n, use_su, original_mi)
        perm_mis.append(mp)

    return _gate_from_mi(original_mi, perm_mis, npermutations, min_nonzero_confidence)
