"""Showcase listing of the built-in pycaret example datasets."""

from __future__ import annotations

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def showcase_pycaret_datasets() -> pd.DataFrame:
    """Return the 20 largest built-in pycaret example datasets, sorted ascending by instance count, for quick manual browsing."""

    from pycaret.datasets import get_data

    df = get_data(verbose=False)
    df["# Instances"] = df["# Instances"].astype(np.int32)
    return df.sort_values("# Instances").tail(20).reset_index(drop=True)
