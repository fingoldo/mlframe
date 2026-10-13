"""Multi-target detection for MRMR's ``y``, carved out of ``_mrmr_class.py``.

The class module re-exports ``_mrmr_y_is_multioutput``, so existing imports keep working.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _mrmr_y_is_multioutput(y) -> bool:
    """True when y carries >=2 target columns (2D DataFrame / 2D ndarray); a Series / 1D array / single-column frame is single-target."""
    if isinstance(y, pd.DataFrame):
        return bool(y.shape[1] >= 2)
    if str(type(y).__module__).startswith("polars") and type(y).__name__ == "DataFrame":
        return bool(y.shape[1] >= 2)
    try:
        arr = np.asarray(y)
    except Exception as exc:
        # Guessing single-target here would route a possibly multi-target y down the single-target fit, which mishandles the extra columns.
        raise TypeError(f"MRMR.fit: cannot read y of type {type(y).__name__} as an array to tell single- from multi-target ({type(exc).__name__}: {exc})") from exc
    return arr.ndim >= 2 and arr.shape[-1] >= 2
