"""Perform exploratory data analysis."""

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

# ensure_installed("pandas")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *  # noqa: F401 pylint: disable=wildcard-import,unused-wildcard-import

import pandas as pd

try:
    from IPython.core.display import display
except:
    display = print


def showcase_dataframe_columns(df: pd.DataFrame, dtype: str = "object", value_counts: pd.Series = None, max_cats_to_show: int = 20, verbose: bool = True):
    """Print short info or value_counts for each dataframe column."""
    sample = df.head(1).select_dtypes(dtype)
    if verbose:
        logger.info("Showcasing %s cols of type %s", sample.shape[1], dtype)
    for col in sample.columns:
        if value_counts:
            tmp = value_counts.get(col)
        else:
            tmp = df[col].value_counts(dropna=False)
        if tmp is not None:
            if len(tmp) <= max_cats_to_show:
                display(tmp.sort_index())
            else:
                print(f"{col}: {len(tmp)} unique vals")
