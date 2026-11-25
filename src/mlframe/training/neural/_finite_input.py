"""Input guard shared by the neural regressors: a NaN or inf cell is refused, not silently passed to the network."""

from __future__ import annotations

import numpy as np


def _refuse_nonfinite(X, owner: str) -> np.ndarray:
    """``X`` as float32, refusing NaN or inf: the network does not mask missing values, so one cell poisons the loss and every prediction (audit 40-F21)."""
    arr = np.asarray(X, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"{owner}: X contains NaN or inf; impute upstream (the network does not mask missing values).")
    return arr
