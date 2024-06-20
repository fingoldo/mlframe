"""Basic feature engineering for ML."""

# pylint: disable=wrong-import-order,wrong-import-position,unidiomatic-typecheck,pointless-string-statement

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.pythonlib import ensure_installed  # lint: disable=ungrouped-imports,disable=wrong-import-order

# ensure_installed("numpy pandas")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd, numpy as np


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
