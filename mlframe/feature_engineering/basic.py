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
    bulk: bool = False,
    methods: dict = {"day": np.int8, "weekday": np.int8, "month": np.int8},  # "week": np.int8, #, "quarter": np.int8 #  , "year": np.int16
) -> pd.DataFrame:
    if len(cols) == 0:
        return

    if bulk:
        tmp = {}
    else:
        tmp = df
    for col in cols:
        for method, dtype in methods.items():
            tmp.loc[:, col + "_" + method] = getattr(df[col].dt, method).astype(dtype)

    if delete_original_cols:
        df.drop(columns=cols, inplace=True)

    if bulk:
        df = pd.concat([df, pd.DataFrame(tmp)], axis=1)

    return df
