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
from mlframe.feature_engineering.numerical import compute_numaggs, get_numaggs_names, compute_numerical_aggregates_numba, get_basic_feature_names
from mlframe.training import train_and_evaluate_model

basic_features_names = get_basic_feature_names(
    return_drawdown_stats=False,
    return_profit_factor=False,
    whiten_means=False,
)

# *****************************************************************************************************************************************************
# Core ensembling functionality
# *****************************************************************************************************************************************************


def batch_numaggs(predictions: np.ndarray, get_numaggs_names_len: int, numaggs_kwds: dict, means_only: bool = True) -> np.ndarray:
    row_features = np.empty(shape=(len(predictions), get_numaggs_names_len), dtype=np.float32)
    for i in range(len(predictions)):
        arr = predictions[i, :]
        numerical_features = compute_numaggs(arr=arr, **numaggs_kwds)
        if means_only:
            numerical_features = compute_numerical_aggregates_numba(arr, geomean_log_mode=False, directional_only=False)
        else:
            numerical_features = compute_numaggs(arr=arr, **numaggs_kwds)
        row_features[i, :] = numerical_features
    return row_features


def enrich_ensemble_preds_with_numaggs(
    predictions: np.ndarray,
    models_names: Sequence = [],
    means_only: bool = False,
    keep_probs: bool = True,
    numaggs_kwds: dict = {"whiten_means": False},
    n_jobs: int = 1,
    only_physical_cores: bool = True,
) -> pd.DataFrame:
    """Probs are non-negative that allows more averages to be applied"""

    if predictions.shape[1] >= 10:
        numaggs_kwds.update(dict(directional_only=False, return_hurst=True, return_entropy=True))
    else:
        numaggs_kwds.update(dict(directional_only=False, return_hurst=False, return_entropy=False))

    if means_only:
        numaggs_names = basic_features_names
    else:
        numaggs_names = list(get_numaggs_names(**numaggs_kwds))

    if keep_probs:
        probs_fields_names = models_names if models_names else [f"p{i}" for i in range(predictions.shape[1])]
    else:
        probs_fields_names = []

    if n_jobs == -1:
        n_jobs = joblib.cpu_count(only_physical_cores=only_physical_cores)

    if n_jobs and n_jobs != 1:
        batch_numaggs_results = parallel_run(
            [
                delayed(batch_numaggs)(predictions=arr, means_only=means_only, get_numaggs_names_len=len(numaggs_names), numaggs_kwds=numaggs_kwds)
                for arr in np.array_split(predictions, n_jobs)
            ],
            backend=None,
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
        row_features = []
        for i in range(len(predictions)):
            arr = predictions[i, :]
            if means_only:
                numerical_features = compute_numerical_aggregates_numba(arr, geomean_log_mode=False, directional_only=False, whiten_means=False)
            else:
                numerical_features = compute_numaggs(arr=arr, **numaggs_kwds)
            if keep_probs:
                line = arr.tolist()
            else:
                line = []
            line.extend(numerical_features)

            row_features.append(line)

    columns = probs_fields_names + numaggs_names

    res = pd.DataFrame(data=row_features, columns=columns)
    if means_only:
        return res[probs_fields_names + "arimean,quadmean,qubmean,geomean,harmmean".split(",")]
    else:
        return res


def ensemble_probabilistic_predictions(
    *preds,
    ensemble_method="harm",
    ensure_prob_limits: bool = True,
    max_mae: float = 0.04,
    max_std: float = 0.06,
    uncertainty_quantile: float = 0.2,
    normalize_stds_by_mean_preds: bool = True,
):
    """Ensembles probabilistic predictions. All elements of the preds tuple must have the same shape.
    uncertainty_quantile>0 produces separate charts for points where the models are confident (agree).
    """

    assert ensemble_method in ("harm", "arithm", "median", "quad", "qube", "geo")
    confident_indices = None

    if len(preds) > 2:

        skipped_preds_indices = set()

        # -----------------------------------------------------------------------------------------------------------------------------------------------------
        # Disregard whole predictions deviating from the median too much
        # -----------------------------------------------------------------------------------------------------------------------------------------------------

        # compute median preds first
        median_preds = np.quantile(np.array(preds), 0.5, axis=0)

        for i, pred in enumerate(preds):
            tot_mae = 0.0
            tot_std = 0.0
            l = pred.shape[1]
            for j in range(l):
                diffs = np.abs((pred[:, j] - median_preds[:, j]))
                mae = np.mean(diffs)
                std = np.sqrt(np.mean(((diffs - mae) ** 2)))
                tot_mae += mae
                tot_std += std
            tot_mae /= l
            tot_std /= l
            if (max_mae > 0 and tot_mae > max_mae) or (tot_std > 0 and tot_std > max_std):
                print(f"ens member {i} excluded due to high distance from the median, mae={tot_mae:4f}, std={tot_std:4f}")
                skipped_preds_indices.add(i)
        if skipped_preds_indices:
            if len(skipped_preds_indices) < len(preds):
                preds = [el for i, el in enumerate(preds) if i not in skipped_preds_indices]
                print(f"Using {len(preds)} members of ensemble")
            else:
                print(f"ensemble_probabilistic_predictions filters too restrictive ({len(skipped_preds_indices)} vs {l}), skipping them")

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Actual ensembling
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    if ensemble_method == "harm":
        ensembled_predictions = 1 / np.mean(np.array([1 / pred for pred in preds]), axis=0)
    elif ensemble_method == "arithm":
        ensembled_predictions = np.mean(np.array(preds), axis=0)
    elif ensemble_method == "median":
        ensembled_predictions = np.quantile(np.array(preds), 0.5, axis=0)
    elif ensemble_method == "quad":
        ensembled_predictions = np.sqrt(np.mean(np.array([pred**2 for pred in preds]), axis=0))
    elif ensemble_method == "qube":
        ensembled_predictions = np.power(np.mean(np.array([pred**3 for pred in preds]), axis=0), 1 / 3)
    elif ensemble_method == "geo":
        ensembled_predictions = np.power(np.prod(preds, axis=0), 1 / len(preds))

    if ensure_prob_limits:
        # if ensembled_predictions.min() < 0 or ensembled_predictions.max() > 1.0:
        # print("normalizing OOB probs")
        tot = np.sum(ensembled_predictions, axis=1)
        ensembled_predictions = ensembled_predictions / tot.reshape(-1, 1)

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Confidence estimates
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    if uncertainty_quantile:

        std_preds = np.std(np.array(preds), axis=0)
        if normalize_stds_by_mean_preds:
            mean_preds = np.mean(np.array(preds), axis=0)
            uncertainty = (std_preds / mean_preds).mean(axis=1)
        else:
            uncertainty = std_preds.mean(axis=1)

        confident_indices = uncertainty >= np.quantile(uncertainty, uncertainty_quantile)

    return ensembled_predictions, confident_indices


def score_ensemble(
    target: pd.Series,
    models_and_predictions: Sequence,
    test_idx: np.ndarray,
    val_idx: np.ndarray,
    ensemble_name: str,
    target_label_encoder: object = None,
    max_mae: float = 0.08,
    max_std: float = 0.08,
    ensure_prob_limits: bool = True,
    nbins: int = 100,
    ensembling_methods="arithm harm median quad qube geo".split(),
    uncertainty_quantile: float = 0.0,
    normalize_stds_by_mean_preds: bool = True,
    custom_ice_metric: Callable = None,
):
    """Compares different ensembling methods for a list of models."""

    res = []
    for ensemble_method in ensembling_methods:
        val_ensembled_predictions, val_confident_indices = ensemble_probabilistic_predictions(
            *(el[4] for el in models_and_predictions),
            ensemble_method=ensemble_method,
            max_mae=max_mae,
            max_std=max_std,
            ensure_prob_limits=ensure_prob_limits,
            uncertainty_quantile=uncertainty_quantile,
            normalize_stds_by_mean_preds=normalize_stds_by_mean_preds,
        )
        test_ensembled_predictions, test_confident_indices = ensemble_probabilistic_predictions(
            *(el[2] for el in models_and_predictions),
            ensemble_method=ensemble_method,
            max_mae=max_mae,
            max_std=max_std,
            ensure_prob_limits=ensure_prob_limits,
            uncertainty_quantile=uncertainty_quantile,
            normalize_stds_by_mean_preds=normalize_stds_by_mean_preds,
        )
        res.append(
            train_and_evaluate_model(
                model=None,
                test_probs=test_ensembled_predictions,
                val_probs=val_ensembled_predictions,
                df=None,
                target=target,
                default_drop_columns=[],
                model_name=f"Ensemble {ensemble_method} {ensemble_name}",
                train_idx=None,
                test_idx=test_idx,
                val_idx=val_idx,
                target_label_encoder=target_label_encoder,
                show_val_chart=True,
                nbins=nbins,
                custom_ice_metric=custom_ice_metric,
            )
        )
        if uncertainty_quantile:
            res.append(
                train_and_evaluate_model(
                    model=None,
                    test_probs=test_ensembled_predictions[test_confident_indices],
                    val_probs=val_ensembled_predictions[val_confident_indices],
                    df=None,
                    target=target,
                    default_drop_columns=[],
                    model_name=f"Conf Ensemble {ensemble_method} {ensemble_name}",
                    train_idx=None,
                    test_idx=test_idx[test_confident_indices],
                    val_idx=val_idx[val_confident_indices],
                    target_label_encoder=target_label_encoder,
                    show_val_chart=True,
                    nbins=nbins,
                    custom_ice_metric=custom_ice_metric,
                )
            )
    return res
