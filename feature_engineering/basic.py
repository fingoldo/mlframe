"""Basic feature engineering for ML."""

# pylint: disable=wrong-import-order,wrong-import-position,unidiomatic-typecheck,pointless-string-statement

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *


import numpy as np, pandas as pd
import polars as pl, polars.selectors as cs

from pyutilz.system import clean_ram
from mlframe.helpers import get_own_ram_usage

# ----------------------------------------------------------------------------------------------------------------------------
# Inits
# ----------------------------------------------------------------------------------------------------------------------------


import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def create_date_features(
    df: Union[pd.DataFrame, pl.DataFrame],
    cols: List[str],
    delete_original_cols: bool = True,
    methods: Dict[str, np.dtype] = {"day": np.int8, "weekday": np.int8, "month": np.int8},
) -> Union[pd.DataFrame, pl.DataFrame]:
    if len(cols) == 0:
        return df
    
    logger.info(f"In create_date_features. RAM usage: {get_own_ram_usage():.1f}GB.")

    is_pandas = isinstance(df, pd.DataFrame)
    is_polars = isinstance(df, pl.DataFrame)
    if not (is_pandas or is_polars):
        raise ValueError("df must be pandas or polars DataFrame")

    # Map NumPy dtypes to Polars dtypes
    dtype_map = {
        np.int8: pl.Int8,
        np.int16: pl.Int16,
        np.int32: pl.Int32,
        np.int64: pl.Int64,
    }

    if is_pandas:
        for col in cols:
            obj = df[col].dt
            for method, dtype in methods.items():
                df[col + "_" + method] = getattr(obj, method).astype(dtype)
        if delete_original_cols:
            df = df.drop(columns=cols)
    elif is_polars:
        all_exprs = []
        for col in cols:
            for method, np_dtype in methods.items():
                pl_dtype = dtype_map.get(np_dtype)
                if pl_dtype is None:
                    raise ValueError(f"Unsupported dtype {np_dtype} for Polars")

                if method == "weekday":
                    # Adjust to match pandas weekday (0=Monday to 6=Sunday)
                    e = (pl.col(col).dt.weekday() - 1).cast(pl_dtype).alias(col + "_" + method)
                else:
                    e = getattr(pl.col(col).dt, method)().cast(pl_dtype).alias(col + "_" + method)

                all_exprs.append(e)
        print("all_exprs",all_exprs)
        # Add all new columns at once
        df = df.with_columns(all_exprs)        

        logger.info(f"After df = df.with_columns(all_exprs). RAM usage: {get_own_ram_usage():.1f}GB.")

        if delete_original_cols:
            df = df.drop(cols)
            logger.info(f"After df.drop(cols). RAM usage: {get_own_ram_usage():.1f}GB.")

    return df


# ----------------------------------------------------------------------------------------------------------------------------
# PySR
# ----------------------------------------------------------------------------------------------------------------------------


def run_pysr_fe(df: pl.DataFrame, nsamples: int = 100_000, target_columns_prefix: str = "target_", timeout_mins: int = 5, fill_nans: bool = True):

    from pysr import PySRRegressor

    clean_ram()

    model = PySRRegressor(
        turbo=True,
        timeout_in_seconds=timeout_mins * 60,
        maxsize=10,
        niterations=10,  # < Increase me for better results
        binary_operators=["+", "*"],
        unary_operators=[
            "cos",
            "exp",
            "log",
            "sin",
            "inv(x) = 1/x",
            # ^ Custom operator (julia syntax)
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = abs(prediction - target)",
        # ^ Custom loss function (julia syntax)
    )

    # Build a mapping from old → new names
    rename_map = {col: col.replace("=", "_").replace(".", "_") for col in df.columns}

    tmp_df = df.head(nsamples) if nsamples else df
    expr = cs.numeric() - cs.starts_with(target_columns_prefix)
    if fill_nans:
        expr = expr.fill_null(0).fill_nan(0)

    model.fit(tmp_df.select(expr).rename(rename_map).collect(), tmp_df.select(cs.starts_with(target_columns_prefix)).collect())

    del tmp_df
    clean_ram()

    return model
