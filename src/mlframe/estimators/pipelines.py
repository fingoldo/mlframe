"""ML pipelines related procedures."""

from __future__ import annotations

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

from typing import Any, Optional, Sequence

import os
import tempfile

import joblib

from mlframe.utils.safe_pickle import (
    _sha256_of_file as _safe_pickle_sha256_of_file,
    verify_sidecar as _safe_pickle_verify_sidecar,
)


def _sha256_of_file(path: str, chunk: int = 1 << 20) -> str:
    """Back-compat shim delegating to :func:`mlframe.utils.safe_pickle._sha256_of_file`."""
    return str(_safe_pickle_sha256_of_file(path, chunk=chunk))


def _verify_sidecar(path: str) -> bool:
    """Back-compat shim delegating to :func:`mlframe.utils.safe_pickle.verify_sidecar`."""
    return _safe_pickle_verify_sidecar(path)

class _LazyModule:
    """Transparent lazy proxy: imports the wrapped module on first attribute
    access. Keeps matplotlib (~0.15s import) off the eager import path -- this
    module is pulled in via ``estimators.__init__`` on any feature-selection
    import, yet matplotlib is only needed by its plotting helpers.
    """

    def __init__(self, name: str):
        self._lm_name = name
        self._lm_mod: Optional[Any] = None

    def __getattr__(self, attr):
        if self._lm_mod is None:
            import importlib

            self._lm_mod = importlib.import_module(self._lm_name)
        return getattr(self._lm_mod, attr)


mpl = _LazyModule("matplotlib")
cm = _LazyModule("matplotlib.cm")
plt = _LazyModule("matplotlib.pyplot")

from mlframe.metrics.core import _close_unless_interactive
from mlframe.metrics.calibration import _show_plots_unless_agg

import pandas as pd, numpy as np
from pyutilz.pythonlib import ensure_dict_elem
from pyutilz.pythonlib import sort_dict_by_value
from pyutilz.strings import slugify


def get_fqn(params_pipe: dict, sep: str = ":", hide_keys: tuple = ("models", "wrappers")) -> str:
    """Returns str repr of all the pipeline steps"""
    return sep.join([str(key) + "=" + str(value) for key, value in sorted(params_pipe.items()) if key not in hide_keys])


def agg_pipeline_metric(cv_results, metric: str = "root_mean_squared_error", func=np.nanmean) -> float:
    """Average some pipeline & dataset's over all estimators, to compare pipelines.
    Allows to quickly find out what faetureset or pipeline is superior.
    """
    res = []

    for key, metrics in cv_results["results"].items():
        if key.endswith("metrics"):
            res.append(metrics.get(metric, np.nan))

    return float(func(np.array(res)))


def replay_cv_results(fname: str, trusted_root: Optional[str] = None):
    """Visualize CV results from stored dump file.

    If ``trusted_root`` is provided, ``fname`` must resolve inside it.
    """
    if trusted_root is not None:
        abs_root = os.path.abspath(trusted_root)
        abs_fname = os.path.abspath(fname)
        try:
            common = os.path.commonpath([abs_root, abs_fname])
        except ValueError:
            raise ValueError(f"Path {abs_fname} is not inside trusted_root {abs_root}")
        if common != abs_root:
            raise ValueError(f"Path {abs_fname} is not inside trusted_root {abs_root}")
    if not _verify_sidecar(fname):
        raise ValueError(f"sha256 sidecar mismatch for {fname}; refusing to load")
    # Trusts the sha256 sidecar verified just above: integrity/corruption gate, NOT authenticity (an attacker with dir write access rewrites both).
    cv_results = joblib.load(fname)
    for title, runs in cv_results.items():
        logger.info("Dataset: %s", title)
        for paramset_hash, cv_results in runs.items():
            logger.info("Pipeline: %s", paramset_hash)
            compare_cv_metrics(cv_results=cv_results, extended=False)
    return cv_results


def optimize_pipeline_by_gridsearch(X, Y, title: str, cv_func: Any, cv_results: Optional[dict] = None, possible_pipeline_blocks: Optional[dict] = None, constants: Optional[dict] = None, output_dir: Optional[str] = None):
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
    if constants is None:
        constants = {}
    if cv_results is None:
        cv_results = {}
    if possible_pipeline_blocks is None:
        possible_pipeline_blocks = {}
    if not possible_pipeline_blocks:
        # what's this paramset unique (but meaningful) hash?
        paramset_hash = get_fqn(constants)

        # print(paramset_hash)
        # return
        ensure_dict_elem(cv_results, title, {})
        logger.info("Submitting CV pipeline %s", paramset_hash)
        cv_results[title][paramset_hash] = cv_func(X=X, Y=Y, title=title, **constants)

        out_dir = output_dir if output_dir is not None else tempfile.gettempdir()
        joblib.dump(cv_results, os.path.join(out_dir, f"cv_results-{slugify(title)}.dump"), compress=9)

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


def compare_cv_metrics(cv_results: dict, metric: str = "root_mean_squared_error", extended: bool = False, cmap=None, figsize=(20, 8), agg_fcn=np.median):
    """Plot per-fold values of ``metric`` for every estimator in a cross-validation results dict.

    Reads ``cv_results["results"]["cv_results"][estimator][metrics|extended_metrics][metric]``, aggregates
    each estimator's folds with ``agg_fcn`` for the legend, sorts estimators by that aggregate, and draws one
    line per estimator (dummy baselines dotted). Returns the matplotlib figure.
    """
    if cmap is None:  # resolved here rather than as a default arg so the lazy matplotlib proxy is not triggered at import
        cmap = cm.viridis
    fig = plt.figure(figsize=figsize)

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
            if mean_score < min_score:
                min_score = mean_score
            mean_scores[estimator_name] = mean_score

    mean_scores = sort_dict_by_value(mean_scores)

    for estimator_name, mean_score in mean_scores.items():
        vals = cv_results["results"]["cv_results"][estimator_name][metrics_source].get(metric)
        plt.plot(vals, label=f"{estimator_name}: {mean_score:.4f}", linestyle="dotted" if "Dummy" in estimator_name else "solid")  # , c=m.to_rgba(mean_score)
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.title(f"CV {metrics_source} comparison: {metric}")
    # Route display through the headless-aware helpers so the gridsearch loop (one figure per explored config)
    # does not leak pyplot figures: on Agg plt.show() is a no-op that never releases the figure, and unclosed
    # figures accumulate (matplotlib warns past 20) holding MBs each. Returns the figure for programmatic retrieval.
    was_shown = _show_plots_unless_agg()
    _close_unless_interactive(fig, was_shown=was_shown)
    return fig


def compute_ml_metrics(y_true, y_preds, scorers: Sequence, storage: Optional[dict] = None) -> dict:
    """Score predictions against every callable in ``scorers``, keyed by the scorer's ``__name__``.

    Returns a ``{scorer_name: score}`` dict. When ``storage`` is given, each score is also appended to
    ``storage[scorer_name]`` (accumulating across calls, e.g. across CV folds).
    """
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
    metrics: Optional[dict] = None,
):
    """Plot predicted vs. true target series for a few selected samples side by side.

    Draws one subplot per index in ``samples`` overlaying ``y_true`` and ``y_preds`` for that sample, with
    any provided RMSE / competition-RMSE / MI values folded into the figure title.
    """
    if metrics is None:
        metrics = {}
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
    was_shown = _show_plots_unless_agg()
    _close_unless_interactive(fig, was_shown=was_shown)
    return fig
