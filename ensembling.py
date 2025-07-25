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

# from pyutilz.pythonlib import ensure_installed;ensure_installed("numpy pandas joblib")

import copy
import joblib
from joblib import delayed
import pandas as pd, numpy as np
from pyutilz.parallel import parallel_run
from mlframe.feature_engineering.numerical import compute_numaggs, get_numaggs_names, compute_numerical_aggregates_numba, get_basic_feature_names

SIMPLE_ENSEMBLING_METHODS: list = "arithm harm median quad qube geo".split()

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
    verbose: bool = True,
) -> tuple:
    """Ensembles probabilistic predictions. All elements of the preds tuple must have the same shape.
    uncertainty_quantile>0 produces separate charts for points where the models are confident (agree).
    """

    assert ensemble_method in ("harm", "arithm", "median", "quad", "qube", "geo")
    confident_indices = None

    preds = [p for p in preds if p is not None]
    if len(preds) == 0:
        return None, None

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
            if (max_mae > 0 and tot_mae > max_mae) or (max_std > 0 and tot_std > max_std):
                if verbose:
                    print(f"ens member {i} excluded due to high distance from the median, mae={tot_mae:4f}, std={tot_std:4f}")
                skipped_preds_indices.add(i)
        if skipped_preds_indices:
            if len(skipped_preds_indices) < len(preds):
                preds = [el for i, el in enumerate(preds) if i not in skipped_preds_indices]
                if verbose:
                    print(f"Using {len(preds)} members of ensemble")
            else:
                if verbose:
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

    if np.isnan(ensembled_predictions).any():  # replace possible nans with values from a safe fallback array
        arith_mean = np.mean(np.array(preds), axis=0)
        ensembled_predictions = np.where(np.isnan(ensembled_predictions), arith_mean, ensembled_predictions)

    if ensure_prob_limits:
        ensembled_predictions = np.clip(ensembled_predictions, 0.0, 1.0)

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
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    val_idx: np.ndarray,
    ensemble_name: str,
    df: pd.DataFrame = None,
    target_label_encoder: object = None,
    max_mae: float = 0.05,
    max_std: float = 0.06,
    ensure_prob_limits: bool = True,
    nbins: int = 100,
    ensembling_methods=SIMPLE_ENSEMBLING_METHODS,
    uncertainty_quantile: float = 0.0,
    normalize_stds_by_mean_preds: bool = True,
    custom_ice_metric: Callable = None,
    custom_rice_metric: Callable = None,
    subgroups: dict = None,
    max_ensembling_level: int = 1,
    verbose: bool = True,
    **kwargs,
):
    """Compares different ensembling methods for a list of models."""

    from mlframe.training import train_and_evaluate_model

    res = {}
    level_models_and_predictions = models_and_predictions

    if (
        level_models_and_predictions[0].val_probs is not None
        or level_models_and_predictions[0].test_probs is not None
        or level_models_and_predictions[0].train_probs is not None
    ):
        is_regression = False
    else:
        is_regression = True
        ensure_prob_limits = False

    for ensembling_level in range(max_ensembling_level):

        next_level_models_and_predictions = []

        for ensemble_method in ensembling_methods:

            if not is_regression:
                predictions = (el.val_probs for el in level_models_and_predictions)
            else:
                predictions = (el.val_preds.reshape(-1, 1) for el in level_models_and_predictions)

            val_ensembled_predictions, val_confident_indices = ensemble_probabilistic_predictions(
                *predictions,
                ensemble_method=ensemble_method,
                max_mae=max_mae,
                max_std=max_std,
                ensure_prob_limits=ensure_prob_limits,
                uncertainty_quantile=uncertainty_quantile,
                normalize_stds_by_mean_preds=normalize_stds_by_mean_preds,
                verbose=verbose,
            )

            if not is_regression:
                predictions = (el.test_probs for el in level_models_and_predictions)
            else:
                predictions = (el.test_preds.reshape(-1, 1) for el in level_models_and_predictions)

            test_ensembled_predictions, test_confident_indices = ensemble_probabilistic_predictions(
                *predictions,
                ensemble_method=ensemble_method,
                max_mae=max_mae,
                max_std=max_std,
                ensure_prob_limits=ensure_prob_limits,
                uncertainty_quantile=uncertainty_quantile,
                normalize_stds_by_mean_preds=normalize_stds_by_mean_preds,
                verbose=verbose,
            )

            if not is_regression:
                predictions = (el.train_probs for el in level_models_and_predictions)
            elif level_models_and_predictions[0].train_preds is not None:
                predictions = (el.train_preds for el in level_models_and_predictions)
                predictions = (el.reshape(-1, 1) if (el is not None) else el for el in predictions)

            train_ensembled_predictions, train_confident_indices = ensemble_probabilistic_predictions(
                *predictions,
                ensemble_method=ensemble_method,
                max_mae=max_mae,
                max_std=max_std,
                ensure_prob_limits=ensure_prob_limits,
                uncertainty_quantile=uncertainty_quantile,
                normalize_stds_by_mean_preds=normalize_stds_by_mean_preds,
                verbose=verbose,
            )

            internal_ensemble_method = f"{ensemble_method} L{ensembling_level}" if ensembling_level > 0 else ensemble_method

            if not is_regression:
                predictive_kwargs = dict(
                    train_probs=train_ensembled_predictions,
                    test_probs=test_ensembled_predictions,
                    val_probs=val_ensembled_predictions,
                )
            else:
                predictive_kwargs = dict(
                    train_preds=train_ensembled_predictions.flatten() if (train_ensembled_predictions is not None) else None,
                    test_preds=test_ensembled_predictions.flatten() if (test_ensembled_predictions is not None) else None,
                    val_preds=val_ensembled_predictions.flatten() if (val_ensembled_predictions is not None) else None,
                )

            next_ens_results = train_and_evaluate_model(
                model=None,
                df=None,
                target=target,
                default_drop_columns=[],
                model_name_prefix=f"Ens{internal_ensemble_method.upper()} {ensemble_name}",
                train_idx=train_idx,
                test_idx=test_idx,
                val_idx=val_idx,
                target_label_encoder=target_label_encoder,
                compute_valset_metrics=True,
                nbins=nbins,
                custom_ice_metric=custom_ice_metric,
                custom_rice_metric=custom_rice_metric,
                subgroups=subgroups,
                **predictive_kwargs,
                **kwargs,
            )
            next_level_models_and_predictions.append(next_ens_results)
            res[internal_ensemble_method] = next_ens_results

            if uncertainty_quantile:
                res[internal_ensemble_method + " conf"] = train_and_evaluate_model(
                    model=None,
                    train_probs=train_ensembled_predictions[train_confident_indices],
                    test_probs=test_ensembled_predictions[test_confident_indices],
                    val_probs=val_ensembled_predictions[val_confident_indices],
                    df=None,
                    target=target,
                    default_drop_columns=[],
                    model_name_prefix=f"Conf Ensemble {internal_ensemble_method} {ensemble_name}",
                    train_idx=train_idx[train_confident_indices],
                    test_idx=test_idx[test_confident_indices],
                    val_idx=val_idx[val_confident_indices],
                    target_label_encoder=target_label_encoder,
                    compute_valset_metrics=True,
                    nbins=nbins,
                    custom_ice_metric=custom_ice_metric,
                    custom_rice_metric=custom_rice_metric,
                    subgroups=subgroups,
                    **kwargs,
                )
        level_models_and_predictions = next_level_models_and_predictions
    return res


def compare_ensembles(
    ensembles: dict,
    sort_metric: str = "test.1.integral_error",
    show_plot: bool = True,
    figsize: tuple = (15, 3),
) -> pd.DataFrame:
    items = []
    for ens_name, ens_perf in ensembles.items():
        perf = copy.deepcopy(ens_perf.metrics)
        for set_name, set_perf in perf.items():
            if set_perf:
                for col in "feature_importances robustness_report".split():
                    if col in set_perf:
                        del set_perf[col]
        ser = pd.json_normalize(perf).iloc[0, :]
        ser.name = ens_name
        items.append(ser)

    res = pd.DataFrame(items)
    if sort_metric in res:
        res = res.sort_values(sort_metric)

        if show_plot:
            if "test." in sort_metric:
                val_metric = sort_metric.replace("test.", "val.")
                if val_metric in res:
                    blank_metric = sort_metric.replace("test.", "")
                    ax = res.set_index(val_metric).sort_index()[sort_metric].plot(title=f"Ensembles {blank_metric}, val vs test", figsize=figsize)
                    ax.set_ylabel(sort_metric)
    return res
