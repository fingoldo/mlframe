"""Shared best-effort float coercion for chart data prep (``shap_panels.py`` / ``pdp_ice.py``):
independently duplicated across those modules, consolidated here so a fix can't silently drift
out of sync across copies.
"""
from __future__ import annotations

import numpy as np


def coerce_float_2d(vals: np.ndarray) -> np.ndarray:
    """Best-effort 2-D float64 view of a (possibly mixed / string / categorical) value matrix.

    Numeric columns pass through; a non-numeric column is label-encoded (``pd.factorize``) to
    category codes so downstream plotting still has usable spread instead of a hard crash on
    ``could not convert string to float``.
    """
    vals = np.asarray(vals)
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)
    if vals.dtype.kind in "fiub":
        return vals.astype(np.float64)
    import pandas as pd

    out = np.empty(vals.shape, dtype=np.float64)
    for j in range(vals.shape[1]):
        col = vals[:, j]
        try:
            out[:, j] = col.astype(np.float64)
        except (ValueError, TypeError):
            codes, _ = pd.factorize(pd.Series(col).astype("string"), use_na_sentinel=True)
            # factorize returns -1 for missing; map that to NaN so downstream plotting drops it rather than plotting a
            # spurious -1 category.
            out[:, j] = np.where(codes < 0, np.nan, codes).astype(np.float64)
    return out
