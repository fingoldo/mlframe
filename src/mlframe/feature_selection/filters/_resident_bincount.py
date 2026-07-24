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
by construction. GPU_INFRA_C-4 fix: unlike ``cupy.bincount`` (which raises a loud
``ValueError`` immediately on any negative index), ``cupyx.scatter_add`` does NO bounds checking at all -- a
negative index silently WRAPS (numpy/cupy fancy-indexing semantics: ``-1`` scatters into ``out[nc-1]``),
producing a plausible-looking but silently WRONG histogram/MI count instead of crashing, and an index
``>= nc`` is a genuine unchecked out-of-bounds device write (cupy disables device-side bounds checks by
default) -- i.e. potential memory corruption, not merely "a bad answer". A caller-side off-by-one in bin-code
generation that would have crashed loudly via ``cupy.bincount`` instead degrades silently through this
sync-free twin. Pass ``debug_check_bounds=True`` (adds one cheap ``x.min()``/``x.max()`` sync -- opt-in only,
since the whole point of this function is avoiding that sync on the hot path) to fail loudly instead.
"""

from __future__ import annotations


def resident_bincount(cp, x, nc, weights=None, dtype=None, debug_check_bounds: bool = False):
    """``cupy.bincount(x, weights, minlength=nc)[:nc]`` without the internal ``int(cupy.max(x))`` sync.

    Parameters
    ----------
    cp : the cupy module (passed in so this stays import-light and testable).
    x : 1-D cupy int array of bin indices, all in ``[0, nc)``.
    nc : number of bins (the exact output length).
    weights : optional 1-D cupy array aligned with ``x``; unweighted count when None.
    dtype : output dtype; defaults to float64 (weighted) / int64 (unweighted) to match the
        ``.astype(cp.float64)`` the callers applied to ``cupy.bincount``'s result.
    debug_check_bounds : opt-in bounds assert (costs one D2H sync -- never enable on the hot path); raises
        ``ValueError`` if any index in ``x`` is outside ``[0, nc)``, matching ``cupy.bincount``'s own
        loud-failure behaviour instead of this function's default silent-wrap/OOB-write UB.
    """
    import cupyx

    n = int(nc)
    if debug_check_bounds:
        lo, hi = int(x.min()), int(x.max())
        if lo < 0 or hi >= n:
            raise ValueError(f"resident_bincount: index out of [0, {n}) range: min={lo}, max={hi}")
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
