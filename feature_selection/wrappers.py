"""Feature selection within ML pipelines. Wrappers method. Currently includes recursive feature elimination."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)


while True:
    try:

        # ----------------------------------------------------------------------------------------------------------------------------
        # Normal Imports
        # ----------------------------------------------------------------------------------------------------------------------------

        from typing import *

        import pandas as pd, numpy as np
        import cupy as cp

        from pyutilz.system import tqdmu
        from sklearn.base import is_classifier,is_regressor,BaseEstimator,TransformerMixin
        from sklearn.model_selection import StratifiedGroupKFold,StratifiedKFold,StratifiedShuffleSplit,GroupKFold,GroupShuffleSplit

        from enum import Enum, auto
        from timeit import default_timer as timer
        from pyutilz.numbalib import set_random_seed

except Exception as e:

    logger.warning(e)

    # ----------------------------------------------------------------------------------------------------------------------------
    # Packages auto-install
    # ----------------------------------------------------------------------------------------------------------------------------

    from pyutilz.pythonlib import ensure_installed

    ensure_installed("numpy pandas cupy")
else:
    break

# ----------------------------------------------------------------------------------------------------------------------------
# Inits
# ----------------------------------------------------------------------------------------------------------------------------

LARGE_CONST: float = 1e30

