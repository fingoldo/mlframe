"""Cross-family candidate collector for the GPU FE batcher (2026-06-26).

Many FE families each emit a SMALL candidate matrix (tens-hundreds of columns on a low-cardinality target).
Scoring them one family at a time issues many small-grid GPU launches that under-occupy a weak card and pay
per-call overhead (H2D + radix-edge + MI launch) each time. This collector concatenates several families'
candidate columns into ONE (n, sum_K) matrix and scores them in ONE batched MI call, amortising the
per-call overhead and raising GPU occupancy (the bigger grid fills more SMs).

Measured on a GTX 1050 Ti (6 SM): scoring F=20 families x 30 cols batched vs separate is ~1.32x faster in
f32 (the launch/occupancy-bound regime); the win grows with the number of small families and is largest in
f32 (f64 is compute-bound, ~1.0x). Per-column MI is independent, so the batched result is BIT-IDENTICAL to
the per-family results (just reassembled by column offset) -- the collector never changes selection.
"""
from __future__ import annotations

import numpy as np


def collect_and_score(matrices, y_codes, nbins: int = 10):
    """Score a list of per-family candidate matrices (each (n, k_f), same n) in ONE batched MI call via the
    dispatcher (CPU/GPU by flags+hardware). Returns a list of per-family (k_f,) float64 MI arrays in input
    order -- bit-identical to scoring each separately (per-column MI is independent), just reassembled by
    column offset. None / empty matrices map to empty results."""
    sizes = []
    valid = []
    for m in matrices:
        if m is None or getattr(m, "size", 0) == 0:
            sizes.append(0)
            continue
        a = np.asarray(m, dtype=np.float64)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        sizes.append(int(a.shape[1]))
        valid.append(a)
    if not valid:
        return [np.zeros(0, dtype=np.float64) for _ in matrices]

    big = np.ascontiguousarray(np.column_stack(valid))
    from .._fe_batch_dispatch import fe_batch_mi
    mi = np.asarray(fe_batch_mi(big, y_codes, nbins))

    out, off = [], 0
    for s in sizes:
        if s == 0:
            out.append(np.zeros(0, dtype=np.float64))
        else:
            out.append(mi[off:off + s])
            off += s
    return out
