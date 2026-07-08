"""Frame-flavour-preserving helpers for ``CompositeTargetEstimator``.

Carved out of ``_estimator.py`` to keep that module under the 1k-LOC limit.
All three are rebound onto the class at the parent's module bottom and operate
on the caller's polars / pandas / ndarray frame WITHOUT down-converting flavour
or materialising row data on the hot path (per the project's 100GB-frame rule):

- ``_subset_rows`` -- row mask, preserving flavour.
- ``_drop_columns`` -- drop plumbing columns (group_column), preserving flavour.
- ``_count_feature_columns`` -- width the inner GBDT trains on (for the
  monotone-constraint length check).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import _is_polars_df

try:
    import polars as pl
except ImportError:  # pragma: no cover - polars optional
    pl = None  # type: ignore


def _subset_rows(X: Any, mask: np.ndarray) -> Any:
    """Row-subset X, preserving the dataframe flavour. Polars / pandas
    / ndarray supported. Raises TypeError otherwise."""
    if _is_polars_df(X):
        # ``_is_polars_df`` only returns True when the module-level ``pl``
        # reference is the real polars module, so we can use it directly.
        return X.filter(pl.Series(mask))
    if isinstance(X, pd.DataFrame):
        return X.loc[mask].reset_index(drop=True)
    if isinstance(X, np.ndarray):
        return X[mask]
    raise TypeError(f"CompositeTargetEstimator: unsupported X type {type(X).__name__} for row subsetting.")


def _drop_columns(X: Any, columns: Sequence[str]) -> Any:
    """Return ``X`` without ``columns``, preserving frame flavour.

    Used to strip the wrapper's plumbing columns (group_column for grouped
    transforms) before passing ``X`` to the inner estimator - tree models like
    LightGBM reject object/string dtypes that the wrapper needs for per-row
    group lookups.

    Silently no-op for columns not present (the caller may pass columns that
    were already dropped upstream by feature selection).
    """
    import logging

    logger = logging.getLogger(__name__)
    # Polars
    if _is_polars_df(X):
        present = [c for c in columns if c in X.columns]
        # ``pl.DataFrame.drop`` is zero-copy (Arrow column projection); the
        # remaining columns are shared, no row data is materialised.
        return X.drop(present) if present else X
    if isinstance(X, pd.DataFrame):
        present = [c for c in columns if c in X.columns]
        if not present:
            # No-op fast path: never touch the frame when nothing is dropped
            # (the common case once feature-selection already removed the
            # plumbing column upstream). Returning ``X`` unchanged avoids the
            # block-consolidating copy ``drop`` would otherwise pay.
            return X
        # ``pd.DataFrame.drop(columns=...)`` copies the remaining blocks on
        # pandas<2 / CoW-off but is a lazy view under pandas>=2 CoW, so a blind
        # rewrite to ``X.loc[:, keep]`` would copy too (and reorder columns).
        # Only the single group_column (grouped-transform path) is dropped here,
        # so the blast radius is bounded. Warn on a large frame in the copying
        # regime so a 100+GB grouped predict does not silently double RAM -- the
        # caller can enable CoW or pass a polars frame.
        _cow = False
        try:
            _cow = bool(pd.get_option("mode.copy_on_write"))
        except Exception:
            _cow = False
        if not _cow:
            try:
                _sz = int(X.memory_usage(index=False, deep=False).sum())
            except Exception:
                _sz = 0
            if _sz > 2 * 1024**3:
                logger.warning(
                    "CompositeTargetEstimator: dropping group_column '%s' "
                    "copies a %.1f GB pandas frame (Copy-on-Write is off). "
                    "Enable pandas CoW (pd.set_option('mode.copy_on_write', "
                    "True)) or pass a polars frame for the zero-copy path.",
                    present[0], _sz / 1024 ** 3,
                )
        return X.drop(columns=present)
    # ndarray has no columns -> nothing to drop.
    return X


def _count_feature_columns(X: Any) -> int:
    """Number of feature columns in ``X`` (the frame the inner trains on).

    Used to validate ``monotone_constraints`` length against the POST-drop
    feature count -- ``X`` here is already ``X_valid`` with any plumbing column
    (group_column) removed, so its width is exactly what the inner GBDT sees.
    Polars / pandas use ``len(.columns)`` (Arrow / block metadata, no row
    materialisation); ndarray uses ``shape[1]`` (1-D -> single column). Raises
    ``TypeError`` on an unsupported carrier so a misconfigured constraint fails
    loudly rather than silently skipping validation.
    """
    if _is_polars_df(X) or isinstance(X, pd.DataFrame):
        return len(X.columns)
    if isinstance(X, np.ndarray):
        return 1 if X.ndim == 1 else int(X.shape[1])
    raise TypeError(f"CompositeTargetEstimator: cannot count feature columns of X type " f"{type(X).__name__} to validate monotone_constraints length.")
