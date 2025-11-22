"""Categorical feature engineering for ML. Optimized & rich set of aggregates for 1d vectors."""

# pylint: disable=wrong-import-order,wrong-import-position,unidiomatic-typecheck,pointless-string-statement

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.pythonlib import (
    ensure_installed,
)  # lint: disable=ungrouped-imports,disable=wrong-import-order

# ensure_installed("numpy pandas scipy")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from antropy import *
import pandas as pd, numpy as np
from scipy.stats import entropy
from .numerical import compute_numaggs, get_numaggs_names

warnings.filterwarnings("ignore", message="nperseg =")

numaggs_names = get_numaggs_names()
directional_numaggs_names = get_numaggs_names(directional_only=True)


def compute_countaggs(
    arr: pd.Series,
    counts_normalize: bool = True,  # use relative or absolute counts
    counts_compute_numaggs: bool = True,  # compute numerical aggregates over counts data or not
    counts_top_n: int = 1,  # return that many highest/lowest value counts
    counts_return_top_counts: bool = True,  # return top counts
    counts_return_top_values: bool = True,  # return top values
    counts_compute_values_numaggs: bool = False,  # if all values are in fact numerical, compute numaggs for them rather than their counts (ordered only, in order of their counts)
    numerical_kwargs: dict = None,
):
    """For some variables, especially with many repeated values, or categorical, we can do value_counts(normalize=True or False). Further we can return
    1) Top N highest/lowest values along with their counts (missing are padded with NaNs)
    2) numaggs over counts data
    3) if variable is numeric, numaggs(timeseries_features=True) for values series sorted by counts (timeseries_features=True leaves only aggregates depending on the order of values,
        'cause otherwise it's simply a duplication of num_aggs over regular series)
    """
    if numerical_kwargs is None:
        numerical_kwargs = dict(return_unsorted_stats=False)
    value_counts = arr.value_counts(normalize=counts_normalize)
    value_counts = value_counts[value_counts > 0]
    values = value_counts.index.values
    counts = value_counts.values

    res = []

    if counts_compute_numaggs:
        res.extend(compute_numaggs(arr=counts, **numerical_kwargs))

    if counts_top_n:

        if len(counts) >= counts_top_n:
            extra = []
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
            res.extend(compute_numaggs(arr=values, **processed_numerical_kwargs))
        else:
            res.extend([np.nan] * len(directional_numaggs_names))

    return res


def get_countaggs_names(
    counts_normalize: bool = True,  # use relative or absolute counts
    counts_compute_numaggs: bool = True,  # compute numerical aggregates over counts data or not
    counts_top_n: int = 1,  # return that many highest/lowest value counts
    counts_return_top_counts: bool = True,  # return top counts
    counts_return_top_values: bool = True,  # return top values
    counts_compute_values_numaggs: bool = False,  # if all values are in fact numerical, compute numaggs for them rather than their counts (ordered only, in order of their counts)
    numerical_kwargs: dict = None,
) -> list:

    if numerical_kwargs is None:
        numerical_kwargs = dict(return_unsorted_stats=False)

    res = []

    if counts_compute_numaggs:
        res.extend([feat + "_" + ("cntnrm" if counts_normalize else "cnt") for feat in get_numaggs_names(**numerical_kwargs)])

    if counts_top_n:
        if counts_return_top_counts:
            res.extend(["top_" + str(i + 1) + "_vcnt" for i in range(counts_top_n)])
            res.extend(["btm_" + str(counts_top_n - i) + "_vcnt" for i in range(counts_top_n)])
        if counts_return_top_values:
            res.extend(["top_" + str(i + 1) + "_vval" for i in range(counts_top_n)])
            res.extend(["btm_" + str(counts_top_n - i) + "_vval" for i in range(counts_top_n)])

    if counts_compute_values_numaggs:
        # if pd.api.types.is_numeric_dtype(values):
        processed_numerical_kwargs = numerical_kwargs.copy()
        processed_numerical_kwargs["directional_only"] = True
        res.extend([feat + "_vvls" for feat in directional_numaggs_names])

    return res
