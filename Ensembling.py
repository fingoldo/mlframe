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

import joblib
from joblib import delayed
import pandas as pd, numpy as np
from pyutilz.parallel import parallel_run
from mlframe.feature_engineering.numerical import compute_numaggs, get_numaggs_names, compute_numerical_aggregates_numba,get_basic_feature_names

basic_features_names=get_basic_feature_names(return_drawdown_stats=False,return_profit_factor=False,)

# *****************************************************************************************************************************************************
# Core ensembling functionality
# *****************************************************************************************************************************************************

def batch_numaggs(predictions: np.ndarray, get_numaggs_names_len:int, numaggs_kwds:dict,means_only:bool=True)-> np.ndarray:
    row_features = np.empty(shape=(len(predictions), get_numaggs_names_len), dtype=np.float32)
    for i in range(len(predictions)):
        arr = predictions[i, :]
        numerical_features = compute_numaggs(arr=arr,**numaggs_kwds)
        if means_only:
            numerical_features = compute_numerical_aggregates_numba(arr, geomean_log_mode=False, directional_only=False)
        else:
            numerical_features = compute_numaggs(arr=arr, **numaggs_kwds)        
        row_features[i, :] = numerical_features
    return row_features

def enrich_ensemble_preds_with_numaggs(predictions:np.ndarray,models_names:Sequence=[],means_only:bool=False,keep_probs:bool=True,numaggs_kwds: dict = {'whiten_means':False},n_jobs:int=1,only_physical_cores:bool=True)->pd.DataFrame:
    """Probs are non-negative that allows more averages to be applied"""
    
    if predictions.shape[1]>=10:
        numaggs_kwds.update(dict(directional_only=False, return_hurst=True, return_entropy=True))
    else:
        numaggs_kwds.update(dict(directional_only=False, return_hurst=False, return_entropy=False))

    if means_only:
        numaggs_names=basic_features_names        
    else:
        numaggs_names=list(get_numaggs_names(**numaggs_kwds))

    if keep_probs:
        probs_fields_names=models_names if models_names else [f'p{i}' for i in range(predictions.shape[1])]
    else:
        probs_fields_names=[]
        
    if n_jobs==-1:
        n_jobs=joblib.cpu_count(only_physical_cores=only_physical_cores)

    if n_jobs and n_jobs!=1:
        batch_numaggs_results = parallel_run(
            [delayed(batch_numaggs)(predictions=arr,means_only=means_only,get_numaggs_names_len=len(numaggs_names), numaggs_kwds=numaggs_kwds) for arr in np.array_split(predictions, n_jobs)], backend=None,
        )
        if keep_probs:
            idx = predictions.shape[1]
        else:
            idx = 0
        row_features = np.empty(shape=(len(predictions), len(numaggs_names) + idx), dtype=np.float32)
        row_features[:, idx:] = np.concatenate(batch_numaggs_results)
        if keep_probs:
            row_features[:, :idx] = predictions        
    else:
        row_features=[]
        for i in range(len(predictions)):
            arr=predictions[i,:]
            if means_only:
                numerical_features = compute_numerical_aggregates_numba(arr, geomean_log_mode=False, directional_only=False)
            else:
                numerical_features = compute_numaggs(arr=arr, **numaggs_kwds)
            if keep_probs:
                line=arr.tolist()
            else:
                line=[]
            line.extend(numerical_features)

            row_features.append(line)    
    
    columns=probs_fields_names+numaggs_names
    
    res=pd.DataFrame(data=row_features,columns=columns)
    if means_only:
        return res[probs_fields_names+"arimean,quadmean,qubmean,geomean,harmmean".split(",")]
    else:
        return res