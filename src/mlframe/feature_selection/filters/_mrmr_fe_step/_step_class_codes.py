"""Dense 0-based class codes for the FE step's MI / CMI scorers."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def dense_class_codes(classes_y: ArrayLike) -> np.ndarray:
    """Map ``classes_y`` to a 1-D array of dense int64 codes 0..K-1, one per row.

    Distinct values stay distinct (no integer cast before ``np.unique``, which would merge fractional labels), and the input is raveled first:
    numpy >= 2 shapes ``return_inverse`` like its input, so an ``(n, 1)`` target would otherwise give ``(n, 1)`` codes.
    """
    flat = np.asarray(classes_y).ravel()
    if flat.size and (np.issubdtype(flat.dtype, np.integer) or flat.dtype == bool):
        # Integer labels over a small range: an O(n) occupancy count + prefix sum gives the same sorted-unique rank as ``np.unique(..., return_inverse)``
        # without sorting the column (a 1M-row sort was ~0.9 s of a strict fit, twice per FE step).
        a = flat.astype(np.int64, copy=False)
        lo = int(a.min())
        span = int(a.max()) - lo + 1
        if span <= (1 << 22):
            shifted = a - lo
            present = np.bincount(shifted, minlength=span) > 0
            rank = np.cumsum(present, dtype=np.int64) - 1
            return rank[shifted]
    _, dense = np.unique(flat, return_inverse=True)
    return dense.astype(np.int64).ravel()
