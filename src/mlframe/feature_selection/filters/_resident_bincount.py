"""Sync-free ``bincount`` for GPU-resident code.

``cupy.bincount`` unconditionally computes ``size = int(cupy.max(x)) + 1`` -- a blocking device-to-host scalar
drain on EVERY call (cupy 13.x even flags the line ``# synchronize!``). Even when ``minlength`` is passed the
max read still runs. In the strict-resident FE path the bin count is ALREADY known (equi-frequency codes live
in ``[0, nc-1]``; joint cell ids in ``[0, n_cells-1]``), so the max read is pure waste -- it was the single
largest contributor to the residency gap (the kernel-timeline gap analysis attributed ~1,000 of the remaining
scalar D2H stalls to the five bincounts in the per-cell moments pass alone).

``resident_bincount`` scatters directly into a pre-sized ``(nc,)`` buffer via ``cupyx.scatter_add`` -- no max,
no sync. The unweighted count is bit-identical to ``cupy.bincount``; the weighted sum differs only by
floating-point reduction order (~1e-13, atomic accumulation), which is inside the 1e-10 selection-equivalence
band the resident gates assert against.

CONTRACT: every index must satisfy ``0 <= x < nc``. This holds for equi-frequency bin codes / joint cell ids
by construction; out-of-range indices are undefined behaviour (same as feeding cupy.bincount a bad minlength).
"""

from __future__ import annotations


def resident_bincount(cp, x, nc, weights=None, dtype=None):
    """``cupy.bincount(x, weights, minlength=nc)[:nc]`` without the internal ``int(cupy.max(x))`` sync.

    Parameters
    ----------
    cp : the cupy module (passed in so this stays import-light and testable).
    x : 1-D cupy int array of bin indices, all in ``[0, nc)``.
    nc : number of bins (the exact output length).
    weights : optional 1-D cupy array aligned with ``x``; unweighted count when None.
    dtype : output dtype; defaults to float64 (weighted) / int64 (unweighted) to match the
        ``.astype(cp.float64)`` the callers applied to ``cupy.bincount``'s result.
    """
    import cupyx

    n = int(nc)
    if weights is None:
        # int32 (not int64): cupy.add.at / scatter_add rejects int64 (supports int32/uint32/uint64/floats).
        # Counts are bounded by n < 2**31, so int32 is exact and uses half the bandwidth. Callers that need
        # another width pass ``dtype`` (e.g. float64 to feed a ratio directly).
        out = cp.zeros(n, dtype=(cp.int32 if dtype is None else dtype))
        cupyx.scatter_add(out, x, 1)
    else:
        out = cp.zeros(n, dtype=(cp.float64 if dtype is None else dtype))
        cupyx.scatter_add(out, x, weights)
    return out
