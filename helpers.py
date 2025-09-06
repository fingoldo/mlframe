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
from .config import *

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


def PrintTimeSeriesSplitExample():
    tscv = TimeSeriesSplit(n_splits=3, max_train_size=50)
    TimeSeriesSplit(n_splits=3)
    for train, test in tscv.split(range(100)):
        print("%s %s" % (train, test))

        # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24] [25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49]
        # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49] [50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74]
        # [25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74] [75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99]


########################################################################################################################################################################################################################################
# Assessing output distribution
########################################################################################################################################################################################################################################


def PlotTargetClassesDistribution(y):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.hist(y)
    print(np.mean(y))


def test_stationarity(timeseries, w):
    import pandas as pd
    from statsmodels.tsa.stattools import adfuller

    # Perform Dickey-Fuller test:
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(dftest[0:4], index=["Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"])
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)
    if dfoutput[0] < dftest[4]["1%"]:
        print("This time series is stationary")
    else:
        print("This time series is NON-stationary")


def has_early_stopping_support(model_type: str) -> bool:
    if model_type in XGBOOST_MODEL_TYPES + LGBM_MODEL_TYPES + CATBOOST_MODEL_TYPES:
        return True
    else:
        return False


def get_model_best_iter(model: object) -> int:
    """Extracts ES best iteration number from a model"""
    if isinstance(model, Pipeline):
        real_model = model.steps[-1]
    else:
        real_model = model

    for field in "best_iteration best_iteration_".split():
        if hasattr(real_model, field):
            return getattr(real_model, field)


def ensure_no_infinity(df: pd.DataFrame, num_cols_only: bool = False) -> bool:
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
    if num_cols_only or len(num_cols) == df.shape[1]:
        tmp = np.isinf(df).any()
        tmp = tmp[tmp == True]
        inf_cols = tmp.index.values.tolist()
    else:
        # protects against TypeError: Object with dtype category cannot perform the numpy op isinf
        for col in tqdmu(num_cols, desc="inf checking", leave=False):
            if np.isinf(df[col]).any():
                inf_cols.append(col)
    if len(inf_cols) > 0:
        for col in inf_cols:
            df[col] = np.nan_to_num(df[col], posinf=nans_filler, neginf=nans_filler)
        if verbose:
            logger.warning(f"Some factors ({len(inf_cols):_}) contained infinity: {', '.join(inf_cols)}")
    return df


def get_own_ram_usage():
    import psutil
    import os

    # Get the current process
    process = psutil.Process(os.getpid())

    # Get memory info
    mem_info = process.memory_info()

    # Print Resident Set Size (RSS) in bytes (actual RAM used)

    return mem_info.rss / (1024**3)


def show_sys_ram_usage():

    mem = psutil.virtual_memory()

    print(f"Total: {mem.total / 1e9:.2f} GB")
    print(f"Available: {mem.available / 1e9:.2f} GB")
    print(f"Used: {mem.used / 1e9:.2f} GB")
    print(f"Free: {mem.free / 1e9:.2f} GB")
    print(f"Memory Usage: {mem.percent}%")
