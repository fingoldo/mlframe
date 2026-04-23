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
    """Replace ±inf with ``nans_filler`` in float columns.

    Only **float** columns can hold infinity, so integer / boolean / category /
    datetime columns are skipped — including pandas nullable extension types
    like ``Int8`` (which the polars→pandas bridge produces for nullable
    Boolean inputs since 2026-04-23). Earlier versions called
    ``df[num_cols].to_numpy()`` over all numeric dtypes, which on an Int8
    column with ``pd.NA`` materializes a Python-object array and then crashed
    with ``TypeError: ufunc 'isinf' not supported for the input types``
    (2026-04-23 LGB prod regression: ``hide_budget`` was Int8 + pd.NA).

    Float extension dtypes (``Float32Dtype`` / ``Float64Dtype`` with
    ``pd.NA``) are handled per-column with ``to_numpy(dtype=float, na_value=
    np.nan)`` so the isinf check works on a real float array.
    """
    # Restrict to float-only columns. Integer + bool can't hold inf, so
    # there's no work to do for them; skipping avoids the extension-dtype
    # to_numpy() pitfall above.
    inf_cols = []
    candidate_cols = (
        list(df.columns) if not num_cols_only
        else [c for c in df.columns if pd.api.types.is_float_dtype(df[c].dtype)]
    )
    if not candidate_cols:
        return df

    for col in candidate_cols:
        s = df[col]
        dt = s.dtype
        try:
            if pd.api.types.is_extension_array_dtype(dt):
                # Float64Dtype / Float32Dtype with pd.NA → cast to numpy
                # float, replacing pd.NA with NaN. NaN is not inf, so this
                # doesn't change the inf-detection answer.
                arr = s.to_numpy(dtype=np.float64, na_value=np.nan)
            else:
                arr = s.to_numpy()
            if not np.issubdtype(arr.dtype, np.floating):
                # Should not happen given the float-only filter, but guards
                # against unexpected object-dtype slip-throughs from upstream.
                continue
            if np.isinf(arr).any():
                inf_cols.append(col)
        except (TypeError, ValueError) as exc:
            # Don't let a single weird column abort the whole pre-fit check —
            # log and move on. The column will simply not be sanitised.
            if verbose:
                logger.warning(
                    "ensure_no_infinity_pd: skipped %r (dtype=%s) — "
                    "isinf check failed: %s",
                    col, dt, exc,
                )

    if inf_cols:
        for col in inf_cols:
            df[col] = np.nan_to_num(df[col], posinf=nans_filler, neginf=nans_filler)
        if verbose:
            logger.warning(f"Some factors ({len(inf_cols):_}) contained infinity: {', '.join(inf_cols)}")
    return df


def show_sys_ram_usage():

    mem = psutil.virtual_memory()

    print(f"Total: {mem.total / 1e9:.2f} GB")
    print(f"Available: {mem.available / 1e9:.2f} GB")
    print(f"Used: {mem.used / 1e9:.2f} GB")
    print(f"Free: {mem.free / 1e9:.2f} GB")
    print(f"Memory Usage: {mem.percent}%")
