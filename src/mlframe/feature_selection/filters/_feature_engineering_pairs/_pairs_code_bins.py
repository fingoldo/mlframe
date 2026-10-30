"""Size of the code table of one discretised candidate column.

``uniform`` quantization reserves one more code for missing values, so the codes of a column reach ``quantization_nbins`` and a table sized ``quantization_nbins`` is one entry short: the
contingency kernels then read past it (a crash in the compiled kernels, a silently closed gate where the error is caught). The table is sized by the codes present.
"""

from __future__ import annotations

import numpy as np


def code_table_bins(codes: np.ndarray, quantization_nbins: int) -> int:
    """``max(quantization_nbins, largest code + 1)``: the number of table entries that holds every code of ``codes``."""
    arr = np.asarray(codes)
    if arr.size == 0:
        return int(quantization_nbins)
    return max(int(quantization_nbins), int(arr.max()) + 1)
