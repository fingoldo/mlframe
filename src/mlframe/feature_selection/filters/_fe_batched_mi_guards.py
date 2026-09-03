"""Input guards for the batched device MI kernels.

Carved out of ``_fe_batched_mi`` for file size: that module crossed the 1000-LOC budget. This guard is a leaf --
it takes an array and raises or returns -- so it moves without dragging anything with it. Re-exported from the
parent, which is its only caller.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _assert_codes_in_range(arr, K: int, name: str, codes_trusted: bool = False) -> None:
    """Guard integer code inputs to the device histogram kernels against out-of-range codes.

    The fused kernels use a code value DIRECTLY as a shared-memory / flat-histogram offset
    (e.g. ``sh[lo*Ky + y[i]]``, ``(col*Kx+xv)*Kb+b[row]``, ``(x*Ky+y)*Kz+zi``). The histogram
    width is sized from ``max()+1`` only, so any code < 0 (a -1 missing/sentinel) or >= K writes
    OUTSIDE the allocated histogram -> cudaErrorIllegalAddress (a hard GPU crash, not a Python
    error). The njit reference (_hermite_fe_mi) guards this exact class explicitly; mirror it here.

    ``codes_trusted`` (FIX1, 2026-06-28): when the caller KNOWS the codes are binner-produced
    (``_gpu_quantile_bin_codes`` / radix / rank always emit dense 0..K-1) the guard is a pure cost -
    it cannot fire, but on a device array it forces TWO blocking ``.item()`` syncs (cp.min + cp.max,
    ~5ms each on a GTX 1050 Ti) at every batched-MI entry. Trusted callers pass True to skip it,
    dropping the guard to ~0 on the resident hot path; untrusted/external code arrays keep the check
    (and the raise contract). For untrusted DEVICE arrays the min/max are computed in ONE stacked
    reduction + ONE ``.get()`` (a single blocking sync instead of two).

    Raises ValueError so an upstream -1 sentinel surfaces as a clear error instead of a GPU
    illegal-address crash. The resident binner never emits negative codes, so on the happy path this
    only ever fires on a genuine upstream bug.
    """
    if codes_trusted:
        return
    try:
        import cupy as cp
        is_dev = isinstance(arr, cp.ndarray)
        xp = cp if is_dev else np
    except ImportError:
        cp = None
        is_dev = False
        xp = np
    if getattr(arr, "size", 1) == 0:
        return
    if is_dev:
        # ONE blocking sync: stack min+max into a 2-vector and a single .get(), not two .item() syncs.
        _lh = cp.stack((xp.min(arr), xp.max(arr))).get()
        lo = int(_lh[0])
        hi = int(_lh[1])
    else:
        lo = int(xp.min(arr))
        hi = int(xp.max(arr))
    if lo < 0:
        raise ValueError(
            "%s contains a negative integer code (min=%d); codes must be 0-based in [0, %d). A -1 "
            "missing/sentinel would index outside the device histogram (illegal address)." % (name, lo, int(K))
        )
    if hi >= int(K):
        raise ValueError(
            "%s code out of range (max=%d >= K=%d); a code >= histogram width would index outside "
            "the device histogram (illegal address)." % (name, hi, int(K))
        )
