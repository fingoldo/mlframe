"""On-device segmented Shannon-entropy reduction for ``friend_graph_gpu.py`` (mrmr_audit_2026-07-20
gpu_residency.md proposal #8: "Port friend_graph_gpu's entropy-from-counts reduction on-device,
relaxed to the codebase's standard ~1e-9 tolerance").

``friend_graph_gpu.py``'s three hot sites (node marginals, feature-target relevance joints, pairwise
edge joints) all follow the SAME pattern: a batched ``cupy.bincount`` produces a device-resident
integer histogram, which is immediately ``cp.asnumpy``'d to host so the bit-exact CPU
``_entropy_from_counts`` (-> the project's njit ``entropy()``) can reduce each per-entity SEGMENT of
that histogram to a scalar. The D2H is paid specifically to run that CPU reduction -- exactly the
self-imposed stricter-than-everywhere-else bar the module's own docstring names ("entropy is computed
by the CPU njit specifically so results are bit-identical to the CPU build").

This module computes the SAME per-segment entropy reduction entirely ON the device, from the
still-resident bincount result, via ONE ``cupy.bincount``-weighted segment-sum (no Python loop, no
per-segment host round-trip) -- then pulls back only the small ``(n_segments,)`` result vector, not
the (potentially much larger) raw counts array. Validated against the CPU path at rtol=1e-9 (the
codebase's own standard GPU/CPU parity bar; the module's own docstring already documents the domain
this collapses -- node-marginal / relevance / pairwise-edge entropy -- as PURE-COUNT arithmetic, so a
float64 device reduction agrees with the CPU njit reduction to FP-reorder noise, not a formula
difference).
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["entropy_segments_gpu"]


def entropy_segments_gpu(cp: Any, counts_dev: Any, n: int, offsets: np.ndarray) -> np.ndarray:
    """Per-segment Shannon entropy (nats) of a device-resident concatenated integer-count array.

    Parameters
    ----------
    cp : module
        The caller's already-imported ``cupy`` module (passed in, not re-imported, matching this
        package's convention for GPU helper functions).
    counts_dev : cupy.ndarray, shape (total,)
        A device-resident concatenation of ``n_segments`` per-entity integer histograms (e.g. the
        raw ``cupy.bincount`` output for node marginals / relevance joints / pairwise joints in
        ``friend_graph_gpu.py`` -- BEFORE any ``cp.asnumpy``).
    n : int
        The row count each segment's counts are normalized by (``freqs = counts / n``), matching
        ``_entropy_from_counts``'s own int/int division exactly.
    offsets : (n_segments + 1,) host int array
        Segment boundaries into ``counts_dev`` (``offsets[i]:offsets[i+1]`` is segment ``i``'s
        counts), the SAME offsets arrays (``off_node`` / ``off_rel`` / the pairwise tile offsets)
        ``friend_graph_gpu.py`` already builds on the host.

    Returns
    -------
    (n_segments,) host float64 array of per-segment entropies, index-aligned with ``offsets``.
    Degenerate segments (all-zero counts, e.g. a genuinely empty joint) return ``0.0``, matching
    ``_entropy_from_counts``'s own explicit ``nz.size == 0 -> 0.0`` branch.
    """
    n_segments = int(offsets.shape[0]) - 1
    if n_segments <= 0:
        return np.zeros(0, dtype=np.float64)
    total = int(offsets[-1])
    if counts_dev.shape[0] != total:
        raise ValueError(f"entropy_segments_gpu: counts_dev length {counts_dev.shape[0]} != offsets[-1] {total}")

    # Segment-id per element: built ONCE on host (offsets is tiny -- n_segments+1 ints) via repeat,
    # uploaded once. Mirrors the same "small host array, one upload" discipline the rest of this
    # package's resident builders use for per-column/per-pair metadata.
    seg_lens = np.diff(offsets).astype(np.int64)
    seg_id_host = np.repeat(np.arange(n_segments, dtype=np.int64), seg_lens)
    seg_id = cp.asarray(seg_id_host)

    counts_f = counts_dev.astype(cp.float64, copy=False)
    freqs = counts_f * (1.0 / float(n))
    # -(freqs * log(freqs)) per nonzero element; zero elements contribute 0 (never a -inf*0 -> nan),
    # matching _entropy_from_counts's explicit nz = counts[counts > 0] prune before the log.
    mask = counts_f > 0
    safe_freqs = cp.where(mask, freqs, 1.0)  # 1.0 is a placeholder that logs to 0, never NaN/-inf
    contrib = cp.where(mask, -(cp.log(safe_freqs) * freqs), 0.0)

    # ONE segment-sum via a weighted bincount -- the same "batch every segment into a single device
    # workload" discipline _wavelet_basis_fe_batched.py's batched_binned_mi_gpu already established
    # for this exact class of "many small per-entity reductions" problem.
    h_dev = cp.bincount(seg_id, weights=contrib, minlength=n_segments)[:n_segments]
    return np.asarray(cp.asnumpy(h_dev), dtype=np.float64)
