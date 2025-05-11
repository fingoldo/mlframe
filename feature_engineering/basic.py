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


# ----------------------------------------------------------------------------------------------------------------------------
# Inits
# ----------------------------------------------------------------------------------------------------------------------------


import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def create_date_features(
    df: pd.DataFrame,
    cols: list,
    delete_original_cols: bool = True,
    methods: dict = {"day": np.int8, "weekday": np.int8, "month": np.int8},  # "week": np.int8, #, "quarter": np.int8 #  , "year": np.int16
) -> pd.DataFrame:
    if len(cols) == 0:
        return

    for col in cols:
        obj = df[col].dt
        for method, dtype in methods.items():
            df[col + "_" + method] = getattr(obj, method).astype(dtype)

    if delete_original_cols:
        for col in cols:
            del df[col]
        # df.drop(columns=cols, inplace=True)

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
