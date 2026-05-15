"""Perform exploratory data analysis."""

# pylint: disable=wrong-import-order,wrong-import-position,unidiomatic-typecheck,pointless-string-statement

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *  # noqa: F401 pylint: disable=wildcard-import,unused-wildcard-import

from pyutilz.data.pandaslib import showcase_df_columns  # noqa: F401  re-export: canonical EDA in pyutilz
