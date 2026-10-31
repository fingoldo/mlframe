"""Size of the code table of one discretised candidate column.

``uniform`` quantization reserves one more code for missing values, so the codes of a column reach ``quantization_nbins`` and a table sized ``quantization_nbins`` is one entry short: the
contingency kernels then read past it (a crash in the compiled kernels, a silently closed gate where the error is caught). The table is sized by the codes present.
"""

from __future__ import annotations

import numpy as np

from .._numba_utils import scrub_nonfinite_1d, scrub_nonfinite_2d


def code_table_bins(codes: np.ndarray, quantization_nbins: int) -> int:
    """``max(quantization_nbins, largest code + 1)``: the number of table entries that holds every code of ``codes``."""
    arr = np.asarray(codes)
    if arr.size == 0:
        return int(quantization_nbins)
    return max(int(quantization_nbins), int(arr.max()) + 1)


def scrub_nonfinite(a: np.ndarray) -> None:
    """``np.nan_to_num(a, copy=False, nan=0.0, posinf=0.0, neginf=0.0)`` in one pass for a float array of one or two dimensions (numpy's version makes three passes and a temporary)."""
    if a.dtype.kind != "f" or a.ndim not in (1, 2):
        np.nan_to_num(a, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    elif a.ndim == 1:
        scrub_nonfinite_1d(a)
    else:
        scrub_nonfinite_2d(a)
