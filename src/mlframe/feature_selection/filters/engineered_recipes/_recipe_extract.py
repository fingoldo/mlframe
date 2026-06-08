"""Column extraction + int-coercion helpers shared by the recipe replay paths.

``_extract_column`` pulls one column from a pandas / polars / structured-ndarray
``X`` as a 1-D ndarray without a full-frame copy; ``_coerce_to_int_with_nan_handling``
maps test-time values to int64 for factorize / target-encoding lookups, honouring
the recipe's ``unknown_strategy``. Both are pure (numpy + optional pandas/polars),
so this submodule imports nothing from a sibling.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]


def _extract_column(X: Any, name: str) -> np.ndarray:
    """Pull a single column from X by name as a 1-D ndarray, no full-frame copy. Supports pandas / polars DataFrame and numpy structured arrays."""
    if pd is not None and isinstance(X, pd.DataFrame):
        # ``.values`` is zero-copy for numeric dtypes; categorical/object materialises only the single column.
        return X[name].to_numpy() if hasattr(X[name], "to_numpy") else X[name].values
    if pl is not None and isinstance(X, pl.DataFrame):
        return X[name].to_numpy()
    if isinstance(X, np.ndarray):
        if X.dtype.names is not None:
            return X[name]
        raise KeyError(
            f"Cannot resolve column '{name}' on a plain 2-D ndarray. "
            "Pass a pandas / polars frame or a structured array."
        )
    raise TypeError(f"Unsupported X type for engineered-recipe replay: {type(X)!r}")


def _coerce_to_int_with_nan_handling(
    vals: np.ndarray, n_bins: int, recipe_name: str, col_name: str,
    unknown_strategy: str,
) -> np.ndarray:
    """Coerce test-time values to int64 for factorize lookup, handling NaN/non-integer per ``unknown_strategy`` (clip -> max bin, sentinel -> new bin,
    raise -> error). Float non-NaN casts to int (rounds toward zero); object/categorical via ``astype(int64)``."""
    if np.issubdtype(vals.dtype, np.floating):
        nan_mask = np.isnan(vals)
        if nan_mask.any():
            if unknown_strategy == "raise":
                n_nan = int(nan_mask.sum())
                raise ValueError(
                    f"Recipe '{recipe_name}': column '{col_name}' has "
                    f"{n_nan} NaN value(s) at transform time. Set "
                    f"unknown_strategy='clip' or 'sentinel' to handle "
                    f"silently."
                )
            # 'clip'/'sentinel' both leave NaN unhandled at the float level; replace with n_bins-1 sentinel so the lookup (which encoded the strategy at
            # fit time) resolves it. The clip path in ``_apply_factorize`` handles the rest.
            vals = vals.copy()
            vals[nan_mask] = n_bins - 1
        return vals.astype(np.int64, copy=False)
    if np.issubdtype(vals.dtype, np.integer):
        return vals.astype(np.int64, copy=False)
    # Object / categorical / string -- try int conversion
    try:
        return vals.astype(np.int64, copy=False)
    except (ValueError, TypeError) as e:
        if unknown_strategy == "raise":
            raise ValueError(
                f"Recipe '{recipe_name}': column '{col_name}' has "
                f"non-integer dtype {vals.dtype!r} that cannot be "
                f"coerced. Pass ordinal-encoded ints or set "
                f"unknown_strategy='clip'."
            ) from e
        # Clip-equivalent fallback: all to 0
        return np.zeros(len(vals), dtype=np.int64)
