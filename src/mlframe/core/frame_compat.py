"""Frame-format compatibility helpers (E3.1, 2026-05-22).

Single-source dispatch for the recurring "is this pandas / polars-DataFrame /
polars-LazyFrame / polars-Series / numpy / object that exposes ``to_pandas`` /
something else" pattern that appears in 14+ ad-hoc sites across the analyzer /
calibration / reporting / evaluation / FE blocks. Each site duck-typed
independently and missed a different edge case (observed in a prod log: the
mini-HPT analyzer treated 25 Float32 polars columns as ``high_cardinality_
categorical`` because its ``isinstance(X, pd.DataFrame)`` check missed polars
and ``np.asarray(X)`` produced an object-dtype array).

Public API
----------
``to_pandas_or_array(X) -> pd.DataFrame | pd.Series | np.ndarray``
    Normalise any common frame/array input to a pandas DataFrame (preferred)
    or numpy ndarray (fallback) WITHOUT silently breaking dtypes. Dispatch
    order:
      1. pandas DataFrame / Series -- returned as-is.
      2. polars DataFrame -- ``.to_pandas()`` (zero-copy where possible).
      3. polars LazyFrame -- ``.collect().to_pandas()``.
      4. polars Series -- ``.to_pandas()`` returns a ``pd.Series`` (NOT wrapped into a
         DataFrame; some callers -- quantile / metric helpers -- prefer the Series form and
         can wrap it themselves when they need column semantics).
      5. numpy array -- returned as-is (the caller decides whether to wrap
         into a DataFrame -- ndim/shape semantics are ambiguous without
         feature names).
      6. Anything else -- ``np.asarray(X)`` fallback.

Migration strategy
------------------
This module SHIPS the helper as a single source of truth. Migration of the
existing 14 sites is left as a separate follow-up because each site has
its own column-classification / dtype-handling needs downstream of the
conversion (the analyzer-side migration already landed in P0 #3 +
follow-up at ``_target_distribution_analyzer.py:_normalise_X``).
"""
from __future__ import annotations

import logging
from typing import Any, Union

import numpy as np
import pandas as pd

__all__ = ("to_pandas_or_array",)

logger = logging.getLogger(__name__)


def _is_polars_module(obj: Any) -> bool:
    """Duck-typed polars detection that avoids a hard polars import dependency."""
    return isinstance(getattr(type(obj), "__module__", None), str) and type(obj).__module__.startswith("polars")


def to_pandas_or_array(
    X: Any,
) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """Normalise heterogeneous frame/array input to a pandas DataFrame
    (preferred) or numpy ndarray.

    See module docstring for the full dispatch table. Returns the original
    object unchanged when it's already a pandas DataFrame / Series / numpy
    ndarray. Polars DataFrame / LazyFrame / Series are converted to pandas
    so downstream code can use ``df.columns`` / ``df[c].dtype`` uniformly.

    Parameters
    ----------
    X
        Input frame or array. Supported: ``pd.DataFrame``, ``pd.Series``,
        ``np.ndarray``, ``polars.DataFrame``, ``polars.LazyFrame``,
        ``polars.Series``, anything that responds to ``np.asarray``.

    Returns
    -------
    pd.DataFrame | pd.Series | np.ndarray
        Pandas DataFrame for any 2-D frame-like input, pandas Series for
        polars Series (caller can wrap if they want a 1-col DataFrame),
        numpy ndarray for everything else.
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X
    if isinstance(X, np.ndarray):
        return X
    if _is_polars_module(X):
        typename = type(X).__name__
        if typename == "LazyFrame":
            # Materialise then to_pandas. LazyFrame has no direct to_pandas method.
            try:
                return X.collect().to_pandas()
            except Exception as exc:
                logger.debug("to_pandas_or_array: polars LazyFrame collect()/to_pandas() failed, falling back to np.asarray: %s", exc)
                return np.asarray(X)
        if typename == "Series":
            # Polars Series -> pandas Series. Caller can wrap to DataFrame if
            # they need columns semantics. We don't auto-wrap because some
            # callers (e.g. quantile / metric helpers) prefer the Series form.
            try:
                return X.to_pandas()
            except Exception as exc:
                logger.debug("to_pandas_or_array: polars Series.to_pandas() failed, falling back to np.asarray: %s", exc)
                return np.asarray(X)
        # DataFrame (or any future polars frame-like).
        to_pandas = getattr(X, "to_pandas", None)
        if callable(to_pandas):
            try:
                return to_pandas()
            except Exception as exc:
                logger.debug("to_pandas_or_array: polars DataFrame.to_pandas() failed, falling back to np.asarray: %s", exc)
                return np.asarray(X)
    # Fallback for everything else: ndarray-ify.
    return np.asarray(X)
