"""Categorical feature engineering for ML. Optimised & rich set of aggregates for 1d vectors."""

__all__ = [
    "compute_countaggs",
    "get_countaggs_names",
]

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from .numerical import compute_numaggs, get_numaggs_names

logger = logging.getLogger(__name__)


_numaggs_names_cache = get_numaggs_names()
_directional_numaggs_names_cache = get_numaggs_names(directional_only=True)


def compute_countaggs(
    arr: pd.Series,
    counts_normalize: bool = True,
    counts_compute_numaggs: bool = True,
    counts_top_n: int = 1,
    counts_return_top_counts: bool = True,
    counts_return_top_values: bool = True,
    counts_compute_values_numaggs: bool = False,
    numerical_kwargs: Optional[dict] = None,
) -> List[float]:
    """Aggregate a series by value-count distribution.

    For variables with many repeated values, or true categoricals, we compute
    ``value_counts(normalize=counts_normalize)`` then optionally derive:

    1. Top-N highest/lowest counts and values (missing slots padded with NaN).
    2. ``compute_numaggs`` over the counts vector.
    3. For numeric variables, ``compute_numaggs(directional_only=True)`` over the values
       vector sorted by count.

    Length of the returned list is guaranteed to match ``len(get_countaggs_names(...))``
    with identical kwargs.
    """
    if numerical_kwargs is None:
        numerical_kwargs = dict(return_unsorted_stats=False)
    value_counts = arr.value_counts(normalize=counts_normalize)
    values = value_counts.index.values
    counts = value_counts.values

    res: list = []

    if counts_compute_numaggs:
        res.extend(compute_numaggs(arr=counts, **numerical_kwargs))

    if counts_top_n:

        if len(counts) >= counts_top_n:
            extra: List[float] = []
        else:
            extra = [np.nan] * (counts_top_n - len(counts))

        if counts_return_top_counts:
            res.extend(counts[:counts_top_n].tolist() + extra)
            res.extend(extra + counts[-counts_top_n:].tolist())
        if counts_return_top_values:
            res.extend(values[:counts_top_n].tolist() + extra)
            res.extend(extra + values[-counts_top_n:].tolist())

    if counts_compute_values_numaggs:
        if pd.api.types.is_numeric_dtype(values):
            processed_numerical_kwargs = numerical_kwargs.copy()
            processed_numerical_kwargs["directional_only"] = True
            extra_features = compute_numaggs(arr=values, **processed_numerical_kwargs)
            if len(extra_features) != len(_directional_numaggs_names_cache):
                raise AssertionError(
                    f"compute_numaggs(directional_only=True) returned {len(extra_features)} values "
                    f"but {len(_directional_numaggs_names_cache)} names are registered; the two must agree."
                )
            res.extend(extra_features)
        else:
            logger.debug(
                "compute_countaggs: counts_compute_values_numaggs=True but values dtype is %s "
                "(non-numeric); padding with NaN.",
                values.dtype,
            )
            res.extend([np.nan] * len(_directional_numaggs_names_cache))

    return res


def get_countaggs_names(
    counts_normalize: bool = True,
    counts_compute_numaggs: bool = True,
    counts_top_n: int = 1,
    counts_return_top_counts: bool = True,
    counts_return_top_values: bool = True,
    counts_compute_values_numaggs: bool = False,
    numerical_kwargs: Optional[dict] = None,
) -> List[str]:
    """Feature names produced by ``compute_countaggs`` under the same kwargs."""
    if numerical_kwargs is None:
        numerical_kwargs = dict(return_unsorted_stats=False)

    res: List[str] = []

    if counts_compute_numaggs:
        suffix = "cntnrm" if counts_normalize else "cnt"
        res.extend([f"{feat}_{suffix}" for feat in get_numaggs_names(**numerical_kwargs)])

    if counts_top_n:
        if counts_return_top_counts:
            res.extend([f"top_{i + 1}_vcnt" for i in range(counts_top_n)])
            res.extend([f"btm_{counts_top_n - i}_vcnt" for i in range(counts_top_n)])
        if counts_return_top_values:
            res.extend([f"top_{i + 1}_vval" for i in range(counts_top_n)])
            res.extend([f"btm_{counts_top_n - i}_vval" for i in range(counts_top_n)])

    if counts_compute_values_numaggs:
        # Names are always emitted; runtime values may be NaN-padded when the input series is non-numeric.
        res.extend([f"{feat}_vvls" for feat in _directional_numaggs_names_cache])

    return res
