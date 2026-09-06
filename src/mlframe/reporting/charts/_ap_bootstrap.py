"""Parallel resample kernel for the PR panel's average-precision bootstrap.

Average precision has no closed-form variance, so the PR panel brackets it by resampling. The loop that did
that was the shape this codebase has a standing rule about: a Python ``for b in range(n_boot)`` calling
already-vectorised numpy once per resample. Each iteration is a handful of full-length passes (a bincount, two
cumulative sums, a dot), so the work is real but every one of them re-enters the interpreter, and the loop
holds the GIL between them. Resamples are independent, which is exactly the case ``prange`` exists for.

The chunked draw is kept, and kept OUTSIDE the kernel: one ``(n_boot, m)`` index matrix is 200 MB at B=500 /
m=50k, and drawing it whole to hand numba a single array would trade the interpreter cost for a memory-rule
violation. Chunking also pins the RNG: the same generator is consumed in the same order as before, so a given
seed produces the same interval it did.

Arithmetic order inside one resample is unchanged (ascending rank), so the per-resample AP matches the numpy
form to floating-point reassociation only -- see ``tests/reporting/test_ap_bootstrap_kernel.py``, which pins
the agreement and the interval.
"""

from __future__ import annotations

import os

import numpy as np

try:  # numba is a hard dependency in practice; the guard keeps an import-time failure from taking the panel down
    from numba import njit, prange

    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only on an install without numba
    _HAVE_NUMBA = False


def jit_is_active() -> bool:
    """True when the compiled kernel is worth calling.

    Under ``NUMBA_DISABLE_JIT=1`` (the nightly coverage job) ``@njit`` bodies run as plain Python, and the
    kernel's explicit per-element loops would be orders of magnitude SLOWER than the numpy form they replace --
    500 resamples x m rows interpreted. The caller keeps the numpy path for that case rather than hanging the
    nightly.
    """
    return _HAVE_NUMBA and os.environ.get("NUMBA_DISABLE_JIT") != "1"


if _HAVE_NUMBA:

    @njit(parallel=True, nogil=True, cache=True, fastmath=False)
    def _ap_chunk_kernel(idx_chunk: np.ndarray, pos_desc: np.ndarray, out: np.ndarray) -> None:
        """Average precision of every resample in ``idx_chunk``, written into ``out``.

        ``idx_chunk`` holds row indices into the score-descending rank order; ``pos_desc`` is the label of each
        rank in that same order. A resample's AP is the precision-weighted positive mass over the resampled
        positives, accumulated in ascending rank -- the identity the numpy form uses, evaluated in the same
        order so the two agree to reassociation.

        NaN marks a resample that drew no positive at all; the caller drops those before taking percentiles.
        """
        n_b, m = idx_chunk.shape
        for b in prange(n_b):
            mult = np.zeros(m, dtype=np.float64)
            for j in range(m):
                mult[idx_chunk[b, j]] += 1.0
            tp = 0.0
            total = 0.0
            acc = 0.0
            for i in range(m):
                w = mult[i] * pos_desc[i]
                tp += w
                total += mult[i]
                if total > 0.0:
                    acc += w * (tp / total)
            out[b] = acc / tp if tp > 0.0 else np.nan
