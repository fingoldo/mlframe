# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from typing import *  # noqa: F401 pylint: disable=wildcard-import,unused-wildcard-import
from .config import XGBOOST_MODEL_TYPES, LGBM_MODEL_TYPES, CATBOOST_MODEL_TYPES

import psutil

import polars.selectors as cs
import pandas as pd, polars as pl, numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit

from pyutilz.system import tqdmu
import pyutilz.polarslib as pllib

########################################################################################################################################################################################################################################
# Helper functions
########################################################################################################################################################################################################################################


def MakeSureBlasAndLaPackAreInstalled():
    from numpy.distutils.system_info import get_info

    print(get_info("blas_opt"))
    print(get_info("lapack_opt"))


def ListAllSkLearnClassifiers():
    from sklearn.utils.testing import all_estimators

    for name, Class in all_estimators():
        if name.find("Class") > 0:
            print(Class.__module__, name)


def has_early_stopping_support(model_type: str) -> bool:
    if model_type in XGBOOST_MODEL_TYPES + LGBM_MODEL_TYPES + CATBOOST_MODEL_TYPES:
        return True
    else:
        return False


def get_model_best_iter(model: object) -> int:
    """Extracts ES best iteration number from a model"""
    if isinstance(model, Pipeline):
        real_model = model.steps[-1][1]
    else:
        real_model = model

    for field in "best_iteration best_iteration_ best_epoch".split():
        if hasattr(real_model, field):
            return getattr(real_model, field)


def ensure_no_infinity(df: pd.DataFrame, num_cols_only: bool = True) -> bool:
    if isinstance(df, pd.DataFrame):
        return ensure_no_infinity_pd(df=df, num_cols_only=num_cols_only)
    elif isinstance(df, pl.DataFrame):
        return ensure_no_infinity_pl(df=df, num_cols_only=num_cols_only)


def ensure_no_infinity_pl(df: pl.DataFrame, num_cols_only: bool = True, nans_filler: float = 0, verbose: int = 1) -> pl.DataFrame:
    cols = cs.all() if not num_cols_only else cs.numeric()
    inf_mask = df.select(cols.is_infinite().any())
    inf_cols = inf_mask.transpose(include_header=True, header_name="column", column_names=["is_inf"]).filter(pl.col("is_inf"))["column"].to_list()

    if len(inf_cols) > 0:
        df = df.with_columns(pllib.clean_numeric(pl.col(inf_cols), nans_filler=nans_filler))

        if verbose:
            logger.warning(f"Some factors ({len(inf_cols):_}) contained infinity: {', '.join(inf_cols)}")

    return df


def ensure_no_infinity_pd(df: pd.DataFrame, num_cols_only: bool = True, nans_filler: float = 0, verbose: int = 1) -> pd.DataFrame:
    num_cols = df.head().select_dtypes("number").columns
    inf_cols = []
    if not num_cols_only or len(num_cols) == df.shape[1]:
        tmp = np.isinf(df).any()
        tmp = tmp[tmp == True]
        inf_cols = tmp.index.values.tolist()
    else:
        # protects against TypeError: Object with dtype category cannot perform the numpy op isinf
        if len(num_cols) > 0:
            inf_mask = np.isinf(df[num_cols].to_numpy()).any(axis=0)
            inf_cols = [c for c, is_inf in zip(num_cols, inf_mask) if is_inf]
    if len(inf_cols) > 0:
        for col in inf_cols:
            df[col] = np.nan_to_num(df[col], posinf=nans_filler, neginf=nans_filler)
        if verbose:
            logger.warning(f"Some factors ({len(inf_cols):_}) contained infinity: {', '.join(inf_cols)}")
    return df


_GET_OWN_RAM_LAST_GB: float = 0.0


def get_own_ram_usage():
    """Return the current process's resident set size in gigabytes.

    On Windows (and occasionally on Linux under heavy Arrow/Polars frees)
    psutil has been observed to briefly report an implausibly low rss —
    sometimes effectively 0 — right after a large buffer release. When
    that happens and the previous rss was substantial, we emit a
    warning and return the previously-seen value instead of the bogus
    near-zero reading. This prevents misleading ``RAM usage: 0.0GB``
    lines in training logs that otherwise hide real usage.
    """
    global _GET_OWN_RAM_LAST_GB
    import psutil
    import os
    import logging as _logging

    process = psutil.Process(os.getpid())
    rss_gb = process.memory_info().rss / (1024**3)

    if _GET_OWN_RAM_LAST_GB > 1.0 and rss_gb < 0.1:
        _logging.getLogger(__name__).warning(
            "psutil reported rss=%.3fGB after previous %.1fGB; likely transient "
            "reporting glitch, returning previous value to keep the RAM log honest.",
            rss_gb, _GET_OWN_RAM_LAST_GB,
        )
        return _GET_OWN_RAM_LAST_GB

    _GET_OWN_RAM_LAST_GB = rss_gb
    return rss_gb


def show_sys_ram_usage():

    mem = psutil.virtual_memory()

    print(f"Total: {mem.total / 1e9:.2f} GB")
    print(f"Available: {mem.available / 1e9:.2f} GB")
    print(f"Used: {mem.used / 1e9:.2f} GB")
    print(f"Free: {mem.free / 1e9:.2f} GB")
    print(f"Memory Usage: {mem.percent}%")
