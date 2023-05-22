"""ML pipelines related procedures.
"""

# pylint: disable=wrong-import-order,wrong-import-position,unidiomatic-typecheck,pointless-string-statement

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

import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import pyplot as plt

import pandas as pd, numpy as np
from pyutilz.pythonlib import ensure_dict_elem
from pyutilz.pythonlib import sort_dict_by_value
from pyutilz.strings import slugify


def get_fqn(params_pipe: dict, sep: str = ":", hide_keys: tuple = ("models", "wrappers")) -> str:
    """Returns str repr of all the pipeline steps"""
    return sep.join([str(key) + "=" + str(value) for key, value in sorted(params_pipe.items()) if key not in hide_keys])


def agg_pipeline_metric(cv_results, metric: str = "root_mean_squared_error", func=np.nanmean) -> float:
    """Average some pipleine & dataset's over all estimators, to compare pipelines.
    Allows to quickly find out what faetureset or pipeline is superior.
    """
    res = []

    for key, metrics in cv_results["results"].items():
        if key.endswith("metrics"):
            res.append(metrics.get(metric, np.nan))

    return func(np.array(res))


def replay_cv_results(fname: str):
    """Visualize CV results from stored dump file."""
    cv_results = joblib.load(fname)
    for title, runs in cv_results.items():
        logger.info("Dataset: %s", title)
        for paramset_hash, cv_results in runs.items():
            logger.info("Pipeline: %s", paramset_hash)
            compare_cv_metrics(cv_results=cv_results, extended=False)
    return cv_results


def optimize_pipeline_by_gridsearch(X, Y, title: str, cv_func: object, cv_results: dict = {}, possible_pipeline_blocks: dict = {}, constants: dict = {}):
    """For a given dataset, checks all possible combinations of the pipeline blocks,
    starting from the least comp. expensive: no FS, no HPT, no OD.
    Saves results on each cycle, summarizes by desired params like estimator type (even over different CVs) etc.

    params vary by detail level:
        it can be either remove_outliers=True/False,
            or actual contamination level of Isolation forest could be specified

    Reminds of grid/random search, but on a pipeline level...
    Make it dask-compatible (Coiled?) to run on cheap spot compute in future?
    But let's be simple for now!

    Has ability to specify constants. In that case, variable with name of constant will be omit from the search.

    Must support exclusion rules, eg, feature_selection_optimistic only has sense if feature_selection_trials is >0

    """
    if not possible_pipeline_blocks:
        # what's this paramset unique (but meaningful) hash?
        paramset_hash = get_fqn(constants)

        # print(paramset_hash)
        # return
        ensure_dict_elem(cv_results, title, {})
        logger.info("Submitting CV pipeline %s", paramset_hash)
        cv_results[title][paramset_hash] = cv_func(X=X, Y=Y, title=title, **constants)

        joblib.dump(cv_results, f"cv_results-{slugify(title)}.dump", compress=9)

        compare_cv_metrics(cv_results=cv_results[title][paramset_hash], extended=False)
        # compare_cv_metrics(cv_results=cv_results, extended=True)
    else:
        # Need to travel all the keys from top to the bottom until actual computing can be started.
        for var, options in possible_pipeline_blocks.items():
            if var not in constants:
                unexplored_options = possible_pipeline_blocks.copy()
                del unexplored_options[var]
                for opt in options:
                    optimize_pipeline_by_gridsearch(
                        X, Y, title=title, cv_func=cv_func, possible_pipeline_blocks=unexplored_options, constants={**constants, **{var: opt}}
                    )


def compare_cv_metrics(cv_results: dict, metric: str = "root_mean_squared_error", extended: bool = False, cmap=cm.viridis, figsize=(20, 8), agg_fcn=np.median):
    plt.figure(figsize=figsize)

    mean_scores = {}
    metrics_source = "extended_metrics" if extended else "metrics"
    min_score = 1e38
    max_score = -1e38
    for estimator_name in cv_results["results"]["cv_results"].keys():
        vals = cv_results["results"]["cv_results"][estimator_name][metrics_source].get(metric)
        if vals is not None:
            mean_score = agg_fcn(np.array(vals))
            if mean_score > max_score:
                max_score = mean_score
            elif mean_score < min_score:
                min_score = mean_score
            mean_scores[estimator_name] = mean_score

    norm = mpl.colors.Normalize(vmin=min_score, vmax=max_score)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    mean_scores = sort_dict_by_value(mean_scores)

    for estimator_name, mean_score in mean_scores.items():
        vals = cv_results["results"]["cv_results"][estimator_name][metrics_source].get(metric)
        plt.plot(vals, label=f"{estimator_name}: {mean_score:.4f}", linestyle="dotted" if "Dummy" in estimator_name else "solid")  # , c=m.to_rgba(mean_score)
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.title(f"CV {metrics_source} comparison: {metric}")
    plt.show()


def compute_ml_metrics(y_true, y_preds, scorers: Sequence, storage: dict = None) -> dict:
    metrics = {}
    for scorer in scorers:
        scorer_name = scorer.__name__
        score = scorer(y_true, y_preds)
        metrics[scorer_name] = score
        if storage is not None:
            if scorer_name not in storage:
                storage[scorer_name] = []
            storage[scorer_name].append(score)
    return metrics


def visualize_prediction_vs_truth(
    y_true,
    y_preds,
    samples=(1, 50, 75),
    title="",
    metrics: dict = {},
) -> dict:
    fig, axs = plt.subplots(1, len(samples), sharey=False, figsize=(20, 5))

    title_line = title
    if "root_mean_squared_error" in metrics:
        title_line += f" RMSE={metrics['root_mean_squared_error']:0.3f}"
    if "competition_custom_mean_squared_error" in metrics:
        title_line += f" COMP_RMSE={metrics['competition_custom_mean_squared_error']:0.3f}"
    if "multioutput_mean_mutual_info_regression" in metrics:
        title_line += f" MI={metrics['multioutput_mean_mutual_info_regression']:0.3f}"

    fig.suptitle(title_line)

    for i, sample in enumerate(samples):
        sample_title = f"#{sample}"
        if sample < len(y_preds):
            axs[i].plot(y_preds[sample], linestyle="--", marker="o", label="predicted")
            if isinstance(y_true, pd.DataFrame):
                sample_title += " " + str(y_true.index.values[sample])
                axs[i].plot(y_true.iloc[sample].values, marker="v", label="true")
            else:
                axs[i].plot(y_true[sample], marker="v", label="true")
            axs[i].legend()
            axs[i].set_title(sample_title)
    plt.show()
