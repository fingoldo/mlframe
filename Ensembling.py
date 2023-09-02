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
import pandas as pd, numpy as np
from mlframe.feature_engineering.numerical import compute_numaggs, get_numaggs_names, basic_features_names,compute_numerical_aggregates_numba

# *****************************************************************************************************************************************************
# Core ensembling functionality
# *****************************************************************************************************************************************************

def ensemble_probs(predicted_probs,true_labels:np.ndarray)->pd.DataFrame:
    """Probs are non-negative that allows more averages to be applied"""
    row_features=[]
    for i in range(len(predicted_probs)):
        simple_numerical_features = compute_numerical_aggregates_numba(arr=predicted_probs[i,:], geomean_log_mode=False, directional_only=False)
        row_features.append(simple_numerical_features)    
    
    return pd.DataFrame(data=row_features,columns=basic_features_names)["arimean,quadmean,qubmean,geomean,harmmean".split(",")]