"""Column selection for fitted selectors that matches columns by NAME when the frame has them, and checks the width otherwise.

A positional slice (``X.iloc[:, support]``) returns the wrong columns without any error when the caller reorders the frame between ``fit`` and ``transform``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


def _positions(support: Any, n_in: int) -> np.ndarray:
    """Selected column positions from a boolean mask of length ``n_in`` or an integer index array."""
    arr = np.asarray(support)
    if arr.dtype == bool:
        return np.flatnonzero(arr)
    return arr.astype(np.int64, copy=False).ravel()


def select_fitted_columns(X: Any, support: Any, feature_names_in: Optional[Sequence[Any]], n_features_in: int, who: str) -> Any:
    """``X`` narrowed to the fitted ``support`` (boolean mask or integer positions into the fit-time columns).

    A pandas or polars frame is matched by column name (compared as strings, since some selectors store ``str(name)``), so a reordered frame gives the
    same columns as the fit-time order; a frame lacking a selected column raises ``ValueError``. A frame or array without usable names is checked
    against ``n_features_in`` and sliced by position. Pandas frames come back through ``iloc`` and polars frames through ``select``, keeping dtypes.
    """
    selected = _positions(support, n_features_in)
    has_columns = hasattr(X, "columns")
    if has_columns and feature_names_in is not None:
        names = [str(n) for n in feature_names_in]
        cols = list(X.columns)
        scols = [str(c) for c in cols]
        if scols == names:
            pos = selected
        else:
            index_of = {s: i for i, s in enumerate(scols)}
            wanted = [names[int(i)] for i in selected]
            missing = [n for n in wanted if n not in index_of]
            if missing:
                raise ValueError(f"{who}.transform: the frame lacks {len(missing)} selected column(s) seen at fit, e.g. {missing[:5]}")
            pos = np.asarray([index_of[n] for n in wanted], dtype=np.int64)
    else:
        width = int(X.shape[1])
        if width != int(n_features_in):
            raise ValueError(f"{who}.transform: got {width} columns, fit saw {int(n_features_in)}")
        pos = selected
    if has_columns and hasattr(X, "iloc"):
        return X.iloc[:, pos]
    if has_columns and hasattr(X, "select"):
        return X.select([list(X.columns)[int(i)] for i in pos])
    return np.asarray(X)[:, pos]
