# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

# from pyutilz.pythonlib import ensure_installed;ensure_installed("numba numpy pandas scipy plotly")

from numba import njit
from math import floor
from scipy.special import expit
from matplotlib import pyplot as plt
import numpy as np, pandas as pd, polars as pl
from sklearn.metrics import log_loss, average_precision_score
from pyutilz.pythonlib import store_params_in_object, get_parent_func_args

import plotly.express as px
import plotly.graph_objects as go
from plotly.io import write_image

from collections import defaultdict
from pyutilz.pythonlib import sort_dict_by_value
from mlframe.stats import get_tukey_fences_multiplier_for_quantile

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def fast_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if isinstance(y_true, (pd.Series, pl.Series)):
        y_true = y_true.to_numpy()
    if isinstance(y_score, (pd.Series, pl.Series)):
        y_score = y_score.to_numpy()
    """np.argsort needs to stay out of njitted func."""
    if y_score.ndim == 2:
        y_score = y_score[:, -1]
    desc_score_indices = np.argsort(y_score)[::-1]
    return fast_numba_auc_nonw(y_true=y_true, y_score=y_score, desc_score_indices=desc_score_indices)


@njit()
def fast_numba_auc_nonw(y_true: np.ndarray, y_score: np.ndarray, desc_score_indices: np.ndarray) -> float:
    """code taken from fastauc lib."""
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    last_counted_fps = 0
    last_counted_tps = 0
    tps, fps = 0, 0
    auc = 0

    l = len(y_true) - 1
    for i in range(l + 1):
        tps += y_true[i]
        fps += 1 - y_true[i]
        if i == l or y_score[i + 1] != y_score[i]:
            auc += (fps - last_counted_fps) * (last_counted_tps + tps)
            last_counted_fps = fps
            last_counted_tps = tps
    tmp = tps * fps * 2
    if tmp > 0:
        return auc / tmp
    else:
        return 0


@njit()
def fast_precision(y_true: np.ndarray, y_pred: np.ndarray, nclasses: int = 2, zero_division: int = 0):
    # storage inits
    allpreds = np.zeros(nclasses, dtype=np.int64)
    hits = np.zeros(nclasses, dtype=np.int64)
    # count stats
    for true_class, predicted_class in zip(y_true, y_pred):
        allpreds[predicted_class] += 1
        if predicted_class == true_class:
            hits[predicted_class] += 1
    precisions = hits / allpreds
    return precisions[-1]


@njit()
def fast_classification_report(y_true: np.ndarray, y_pred: np.ndarray, nclasses: int = 2, zero_division: int = 0):
    """Custom classification report, proof of concept."""

    N_AVG_ARRAYS = 3  # precisions, recalls, f1s

    # storage inits
    weighted_averages = np.empty(N_AVG_ARRAYS, dtype=np.float64)
    macro_averages = np.empty(N_AVG_ARRAYS, dtype=np.float64)
    supports = np.zeros(nclasses, dtype=np.int64)
    allpreds = np.zeros(nclasses, dtype=np.int64)
    misses = np.zeros(nclasses, dtype=np.int64)
    hits = np.zeros(nclasses, dtype=np.int64)

    # count stats
    for true_class, predicted_class in zip(y_true, y_pred):
        supports[true_class] += 1
        allpreds[predicted_class] += 1
        if predicted_class == true_class:
            hits[predicted_class] += 1
        else:
            misses[predicted_class] += 1

    # main calcs
    accuracy = hits.sum() / len(y_true)
    balanced_accuracy = np.nan_to_num(hits / supports, copy=True, nan=zero_division).mean()

    recalls = hits / supports
    precisions = hits / allpreds
    f1s = 2 * (precisions * recalls) / (precisions + recalls)

    # fix nans & compute averages
    i = 0
    for arr in (precisions, recalls, f1s):
        np.nan_to_num(arr, copy=False, nan=zero_division)
        weighted_averages[i] = (arr * supports).sum() / len(y_true)
        macro_averages[i] = arr.mean()
        i += 1

    return hits, misses, accuracy, balanced_accuracy, supports, precisions, recalls, f1s, macro_averages, weighted_averages


@njit()
def fast_calibration_binning(y_true: np.ndarray, y_pred: np.ndarray, nbins: int = 100):
    """Computes bins of predicted vs actual events frequencies. Corresponds to sklearn's UNIFORM strategy."""

    pockets_predicted = np.zeros(nbins, dtype=np.int64)
    pockets_true = np.zeros(nbins, dtype=np.int64)

    # compute span

    min_val, max_val = 1.0, 0.0
    for predicted_prob in y_pred:
        if predicted_prob > max_val:
            max_val = predicted_prob
        if predicted_prob < min_val:
            min_val = predicted_prob
    span = max_val - min_val

    if span > 0:
        multiplier = (nbins - 1) / span
        for true_class, predicted_prob in zip(y_true, y_pred):
            ind = floor((predicted_prob - min_val) * multiplier)
            pockets_predicted[ind] += 1
            pockets_true[ind] += true_class
    else:
        ind = 0
        for true_class, predicted_prob in zip(y_true, y_pred):
            pockets_predicted[ind] += 1
            pockets_true[ind] += true_class

    idx = np.nonzero(pockets_predicted > 0)[0]

    hits = pockets_predicted[idx]
    if len(hits) > 0:
        freqs_predicted, freqs_true = (min_val + (np.arange(nbins)[idx] + 0.5) * span / nbins).astype(np.float64), pockets_true[idx] / pockets_predicted[idx]
    else:
        freqs_predicted, freqs_true = np.array((), dtype=np.float64), np.array((), dtype=np.float64)

    return freqs_predicted, freqs_true, hits


def show_calibration_plot(
    freqs_predicted: np.ndarray,
    freqs_true: np.ndarray,
    hits: np.ndarray,
    show_plots: bool = True,
    plot_file: str = "",
    plot_title: str = "",
    figsize: tuple = (12, 6),
    backend: str = "matplotlib",
    label_freq: str = "Observed Frequency",
    label_perfect: str = "Perfect",
    label_real: str = "Real",
    label_prob: str = "Predicted Probability",
    colorbar_label: str = "Bin population",
    use_size: bool = False,
):
    """Plots reliability digaram from the binned predictions."""

    assert backend in ("plotly", "matplotlib")

    x_min, x_max = np.min(freqs_predicted), np.max(freqs_predicted)

    if backend == "matplotlib":
        cm = plt.cm.get_cmap("RdYlBu")
        fig = plt.figure(figsize=figsize)
        sc = plt.scatter(x=freqs_predicted, y=freqs_true, marker="o", s=5000 * hits / hits.sum(), c=hits, label=label_freq, cmap=cm)
        plt.plot([x_min, x_max], [x_min, x_max], "g--", label=label_perfect)
        plt.xlabel(label_prob)
        plt.ylabel(label_freq)
        cbar = plt.colorbar(sc)
        cbar.set_label(colorbar_label)  # , fontsize=12
        if plot_title:
            plt.title(plot_title)

        if plot_file:
            fig.savefig(plot_file)

        if show_plots:
            plt.ion()
            plt.show()
        else:
            plt.close(fig)
    else:

        df = pd.DataFrame(
            {
                label_prob: freqs_predicted,
                label_freq: freqs_true,
                "NCases": hits,
            }
        )
        hover_data = {label_prob: ":.2%", label_freq: ":.2%", "NCases": True}
        print(hover_data)

        if use_size:
            df["size"] = 5000 * hits / hits.sum()
            hover_data["size"] = False

        fig = go.Figure()
        # fig = px.scatter(data_frame=df ,x=label_prob,y=label_freq,size="size" if use_size else None, color="NCases", labels={'x':label_prob, 'y':label_freq},hover_data=hover_data)
        fig.add_trace(
            go.Scatter(
                data_frame=df,
                x=label_prob,
                y=label_freq,
                size="size" if use_size else None,
                color="NCases",
                labels={"x": label_prob, "y": label_freq},
                hover_data=hover_data,
                name=label_real,
            )
        )
        fig.add_trace(go.Scatter(x=[x_min, x_max], y=[x_min, x_max], line={"color": "green", "dash": "dash"}, name=label_perfect, mode="lines"))
        fig.update(layout_coloraxis_showscale=False)
        if plot_title:
            fig.update_layout(title=plot_title)

        if plot_file:
            ext = plot_file.split(".")[-1]
            if not ext:
                ext = "png"
            write_image(fig, file=plot_file, format=ext)

        if show_plots:
            fig.show()
    return fig


@njit()
def maximum_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    return np.nanmax(mape)


@njit()
def calibration_metrics_from_freqs(
    freqs_predicted: np.ndarray, freqs_true: np.ndarray, hits: np.ndarray, nbins: int, array_size: int, use_weights: bool = True
):
    calibration_coverage = len(set(np.round(freqs_predicted, int(np.log10(nbins))))) / nbins
    if len(hits) > 0:
        diffs = np.abs((freqs_predicted - freqs_true))
        if use_weights:
            weights = hits / array_size
            calibration_mae = np.sum(diffs * weights)
            calibration_std = np.sqrt(np.sum(((diffs - calibration_mae) ** 2) * weights))
        else:
            calibration_mae = np.mean(diffs)
            calibration_std = np.sqrt(np.mean(((diffs - calibration_mae) ** 2)))
    else:
        calibration_mae, calibration_std = 1.0, 1.0

    return calibration_mae, calibration_std, calibration_coverage


@njit()
def fast_calibration_metrics(y_true: np.ndarray, y_pred: np.ndarray, nbins: int = 100, use_weights: bool = False, verbose: int = 0):
    freqs_predicted, freqs_true, hits = fast_calibration_binning(y_true=y_true, y_pred=y_pred, nbins=nbins)
    if verbose:
        print(freqs_predicted, freqs_true)
    return calibration_metrics_from_freqs(
        freqs_predicted=freqs_predicted, freqs_true=freqs_true, hits=hits, nbins=nbins, array_size=len(y_true), use_weights=use_weights
    )


def fast_aucs(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Compute both ROC AUC and PR AUC efficiently."""
    if y_score.ndim == 2:
        y_score = y_score[:, -1]
    desc_score_indices = np.argsort(y_score)[::-1]
    return fast_numba_aucs(y_true=y_true, y_score=y_score, desc_score_indices=desc_score_indices)


@njit()
def fast_numba_aucs(y_true: np.ndarray, y_score: np.ndarray, desc_score_indices: np.ndarray) -> tuple[float, float]:
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]

    total_pos = np.sum(y_true_sorted)
    if total_pos == 0:
        return 0.0, 0.0

    # Variables for ROC AUC
    last_counted_fps = 0
    last_counted_tps = 0
    tps, fps = 0, 0
    roc_auc = 0.0

    # Variables for PR AUC (aligned with sklearn's step-wise Riemann sum)
    prev_recall = 0.0
    pr_auc = 0.0

    n = len(y_true_sorted)
    for i in range(n):
        tps += y_true_sorted[i]
        fps += 1 - y_true_sorted[i]

        if i == n - 1 or y_score_sorted[i + 1] != y_score_sorted[i]:
            # Update ROC AUC
            delta_fps = fps - last_counted_fps
            sum_tps = last_counted_tps + tps
            roc_auc += delta_fps * sum_tps
            last_counted_fps = fps
            last_counted_tps = tps

            # Update PR AUC (key change: use current_precision instead of average)
            current_precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
            current_recall = tps / total_pos
            delta_recall = current_recall - prev_recall
            pr_auc += delta_recall * current_precision  # Riemann sum
            prev_recall = current_recall

    # Normalize ROC AUC
    denom_roc = tps * fps * 2
    if denom_roc > 0:
        roc_auc /= denom_roc
    else:
        roc_auc = 0.0

    return roc_auc, pr_auc


def fast_aucs_per_group(y_true: np.ndarray, y_score: np.ndarray, group_ids: np.ndarray) -> Tuple[float, float, Dict[int, Tuple[float, float]]]:
    """
    Compute overall AUCs and per-group AUCs efficiently.

    Returns:
        - Overall ROC AUC
        - Overall PR AUC
        - Dictionary mapping group_id -> (roc_auc, pr_auc)
    """
    if y_score.ndim == 2:
        y_score = y_score[:, -1]

    # Overall AUCs
    desc_score_indices = np.argsort(y_score)[::-1]
    overall_roc_auc, overall_pr_auc = fast_numba_aucs(y_true, y_score, desc_score_indices)

    # Per-group AUCs
    unique_groups = np.unique(group_ids)
    group_aucs = {}

    for group_id in unique_groups:
        group_mask = group_ids == group_id
        group_y_true = y_true[group_mask]
        group_y_score = y_score[group_mask]

        if len(group_y_true) > 1:  # Need at least 2 samples
            group_desc_indices = np.argsort(group_y_score)[::-1]
            roc_auc, pr_auc = fast_numba_aucs(group_y_true, group_y_score, group_desc_indices)
            group_aucs[int(group_id)] = (roc_auc, pr_auc)
        else:
            group_aucs[int(group_id)] = (0.0, 0.0)

    return overall_roc_auc, overall_pr_auc, group_aucs


def fast_aucs_per_group_optimized(y_true: np.ndarray, y_score: np.ndarray, group_ids: np.ndarray = None) -> Tuple[float, float, Dict[int, Tuple[float, float]]]:
    """
    More memory-efficient version that groups data by group first.
    Better for cases with many groups and reasonable group sizes.
    """
    if y_score.ndim == 2:
        y_score = y_score[:, -1]

    # Overall AUCs
    desc_score_indices = np.argsort(y_score)[::-1]
    overall_roc_auc, overall_pr_auc = fast_numba_aucs(y_true, y_score, desc_score_indices)

    # By group very efficiently
    if group_ids is not None:
        sort_indices = np.argsort(group_ids)
        sorted_group_ids = group_ids[sort_indices]
        sorted_y_true = y_true[sort_indices]
        sorted_y_score = y_score[sort_indices]

        group_aucs = compute_grouped_group_aucs(sorted_group_ids, sorted_y_true, sorted_y_score)
    else:
        group_aucs = {}

    return overall_roc_auc, overall_pr_auc, group_aucs


@njit()
def compute_grouped_group_aucs(sorted_group_ids: np.ndarray, sorted_y_true: np.ndarray, sorted_y_score: np.ndarray) -> Dict[int, Tuple[float, float]]:
    """
    Compute AUCs for each group from pre-sorted data.
    """
    group_aucs = {}
    n = len(sorted_group_ids)

    if n == 0:
        return group_aucs

    start_idx = 0
    current_group = sorted_group_ids[0]

    for i in range(1, n + 1):
        # Check if we've reached end or found a new group
        if i == n or sorted_group_ids[i] != current_group:
            end_idx = i
            group_size = end_idx - start_idx

            if group_size > 1:
                # Extract group data
                group_y_true = sorted_y_true[start_idx:end_idx]
                group_y_score = sorted_y_score[start_idx:end_idx]

                # Sort by score for this group
                group_desc_indices = np.argsort(group_y_score)[::-1]

                # Compute AUCs for this group
                roc_auc, pr_auc = fast_numba_aucs_simple(group_y_true, group_y_score, group_desc_indices)
                group_aucs[int(current_group)] = (roc_auc, pr_auc)
            else:
                group_aucs[int(current_group)] = (0.0, 0.0)

            # Move to next group
            if i < n:
                start_idx = i
                current_group = sorted_group_ids[i]

    return group_aucs


@njit()
def fast_numba_aucs_simple(y_true: np.ndarray, y_score: np.ndarray, desc_score_indices: np.ndarray) -> Tuple[float, float]:
    """
    Simplified version of your original function for per-group computation.
    """
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]
    total_pos = np.sum(y_true_sorted)

    if total_pos == 0:
        return 0.0, 0.0

    # Variables for ROC AUC
    last_counted_fps = 0
    last_counted_tps = 0
    tps, fps = 0, 0
    roc_auc = 0.0

    # Variables for PR AUC
    prev_recall = 0.0
    pr_auc = 0.0
    n = len(y_true_sorted)

    for i in range(n):
        tps += y_true_sorted[i]
        fps += 1 - y_true_sorted[i]

        if i == n - 1 or y_score_sorted[i + 1] != y_score_sorted[i]:
            # Update ROC AUC
            delta_fps = fps - last_counted_fps
            sum_tps = last_counted_tps + tps
            roc_auc += delta_fps * sum_tps
            last_counted_fps = fps
            last_counted_tps = tps

            # Update PR AUC
            current_precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
            current_recall = tps / total_pos
            delta_recall = current_recall - prev_recall
            pr_auc += delta_recall * current_precision
            prev_recall = current_recall

    # Normalize ROC AUC
    denom_roc = tps * fps * 2
    if denom_roc > 0:
        roc_auc /= denom_roc
    else:
        roc_auc = 0.0

    return roc_auc, pr_auc


def compute_mean_aucs_per_group(group_aucs: dict) -> tuple:

    # Compute mean per-group AUCs
    group_roc_aucs = [aucs[0] for aucs in group_aucs.values()]
    group_pr_aucs = [aucs[1] for aucs in group_aucs.values()]

    mean_roc_auc, mean_pr_auc = np.mean(group_roc_aucs), np.mean(group_pr_aucs)

    return mean_roc_auc, mean_pr_auc


def fast_calibration_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nbins: int = 10,
    show_plots: bool = True,
    show_points_density_in_title: bool = False,
    show_brier_loss_in_title: bool = True,
    show_cmaew_in_title: bool = True,
    show_roc_auc_in_title: bool = True,
    show_pr_auc_in_title: bool = True,
    show_logloss_in_title: bool = True,
    show_coverage_in_title: bool = False,
    plot_file: str = "",
    figsize: tuple = (15, 6),
    ndigits: int = 3,
    backend: str = "matplotlib",
    title: str = "",
    use_weights=True,
    verbose: bool = False,
    group_ids: np.ndarray = None,
    **ice_kwargs,
):
    """Bins predictions, then computes regresison-like error metrics between desired and real binned probs."""

    assert backend in ("plotly", "matplotlib")

    brier_loss = brier_score_loss(y_true=y_true, y_prob=y_pred)

    freqs_predicted, freqs_true, hits = fast_calibration_binning(y_true=y_true, y_pred=y_pred, nbins=nbins)
    if verbose:
        print("freqs_predicted", freqs_predicted)
        print("freqs_true", freqs_true)
    min_hits, max_hits = np.min(hits), np.max(hits)
    calibration_mae, calibration_std, calibration_coverage = calibration_metrics_from_freqs(
        freqs_predicted=freqs_predicted, freqs_true=freqs_true, hits=hits, nbins=nbins, array_size=len(y_true), use_weights=use_weights
    )

    fig = None

    # roc_auc, pr_auc = fast_roc_auc(y_true=y_true, y_score=y_pred), average_precision_score(y_true=y_true, y_score=y_pred)
    # roc_auc, pr_auc = fast_aucs(y_true=y_true, y_score=y_pred)
    roc_auc, pr_auc, group_aucs = fast_aucs_per_group_optimized(y_true=y_true, y_score=y_pred, group_ids=group_ids)
    mean_group_roc_auc, mean_group_pr_auc = compute_mean_aucs_per_group(group_aucs) if group_aucs else (None, None)

    ice = integral_calibration_error_from_metrics(
        calibration_mae=calibration_mae,
        calibration_std=calibration_std,
        calibration_coverage=calibration_coverage,
        brier_loss=brier_loss,
        roc_auc=roc_auc,
        **ice_kwargs,
    )
    ll = log_loss(y_true=y_true, y_pred=y_pred)

    if plot_file or show_plots:
        plot_title = f"ICE={ice:.{ndigits}f}"
        if show_brier_loss_in_title:
            plot_title += f", BR={brier_loss*100:.{ndigits}f}%"
        if show_cmaew_in_title:
            plot_title += f", CMAE{'W' if use_weights else ''}={calibration_mae*100:.{ndigits}f}%±{calibration_std*100:.{ndigits}f}%"
        if show_coverage_in_title:
            plot_title += f", COV={calibration_coverage*100:.{int(np.log10(nbins))}f}%"
        if show_roc_auc_in_title:
            plot_title += f", ROC AUC={roc_auc:.{ndigits}f}"
            if mean_group_roc_auc is not None:
                plot_title += f"({mean_group_roc_auc:.{ndigits}f} per-group)"
        if show_pr_auc_in_title:
            plot_title += f", PR AUC={pr_auc:.{ndigits}f}"
            if mean_group_pr_auc is not None:
                plot_title += f"({mean_group_pr_auc:.{ndigits}f} per-group)"
        if show_logloss_in_title:
            plot_title += f", LL={ll:.{ndigits}f}"
        if show_points_density_in_title:
            plot_title += f", DENS=[{max_hits:_};{min_hits:_}]"
        if title:
            plot_title = title.strip() + "\n" + plot_title
        fig = show_calibration_plot(
            freqs_predicted=freqs_predicted,
            freqs_true=freqs_true,
            hits=hits,
            plot_title=plot_title,
            show_plots=show_plots,
            plot_file=plot_file,
            figsize=figsize,
            backend=backend,
        )

    return brier_loss, calibration_mae, calibration_std, calibration_coverage, roc_auc, pr_auc, ice, ll, fig


def predictions_time_instability(preds: pd.Series) -> float:
    """Computes how stable are true values or predictions over time.
    It's hard to use predictions that change upside down from point to point.
    For binary classification instability ranges from 0 to 1, for regression from 0 to any value depending on the target stats.
    """
    return np.abs(np.diff(preds)).mean()


# ----------------------------------------------------------------------------------------------------------------------------
# Errors & scorers
# ----------------------------------------------------------------------------------------------------------------------------


def compute_probabilistic_multiclass_error(
    y_true: Union[pd.Series, pd.DataFrame, np.ndarray],
    y_score: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence],
    labels: np.ndarray = None,
    method: str = "multicrit",
    mae_weight: float = 3,
    std_weight: float = 2,
    roc_auc_weight: float = 1.5,
    brier_loss_weight: float = 0.4,
    min_roc_auc: float = 0.54,
    roc_auc_penalty: float = 0.00,
    use_weighted_calibration: bool = True,
    weight_by_class_npositives: bool = False,
    nbins: int = 10,
    verbose: bool = False,
    ndigits: int = 4,
    **kwargs,  # as scorer can pass kwargs of this kind: {'needs_proba': True, 'needs_threshold': False}
):
    """Given a sequence of per-class probabilities (predicted by some model), and ground truth targets,
    computes weighted sum of per-class errors.
    Supports several error estimation methods: "multicrit", "brier_score", "precision".
    If number of classes is only 2, skips class 0 as it's fully complementary to class 1.
    """

    assert method in ("multicrit", "brier_score", "precision")

    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        y_true = y_true.values
    if isinstance(y_score, (pd.Series, pd.DataFrame)):
        y_score = y_score.values
    if labels is not None and isinstance(labels, (pd.Series, pd.DataFrame)):
        labels = labels.values

    if isinstance(y_score, Sequence):
        probs = y_score
    else:
        if len(y_score.shape) == 1:
            y_score = np.vstack([1 - y_score, y_score]).T
        probs = [y_score[:, i] for i in range(y_score.shape[1])]

    total_error = 0.0
    weights_sum = 0

    for class_id in range(len(probs)):

        if len(probs) == 2 and class_id == 0:
            continue

        y_pred = probs[class_id]
        if labels is not None:
            correct_class = y_true == labels[class_id]
        else:
            correct_class = y_true == class_id

        if isinstance(correct_class, (pd.Series, np.ndarray)):
            correct_class = correct_class.astype(np.int8)
        elif isinstance(correct_class, pl.Series):
            correct_class = correct_class.cast(pl.Int8).to_numpy()

        if method == "multicrit":

            if False:
                calibration_mae, calibration_std, calibration_coverage = fast_calibration_metrics(
                    y_true=correct_class,
                    y_pred=y_pred,
                    use_weights=use_weighted_calibration,
                    nbins=nbins,
                )
                brier_loss = brier_score_loss(y_true=correct_class, y_prob=y_pred)

                desc_score_indices = np.argsort(y_pred)[::-1]
                roc_auc = fast_numba_auc_nonw(y_true=correct_class, y_score=y_pred, desc_score_indices=desc_score_indices)

                class_error = integral_calibration_error_from_metrics(
                    calibration_mae=calibration_mae,
                    calibration_std=calibration_std,
                    calibration_coverage=calibration_coverage,
                    brier_loss=brier_loss,
                    roc_auc=roc_auc,
                    mae_weight=mae_weight,
                    std_weight=std_weight,
                    brier_loss_weight=brier_loss_weight,
                    roc_auc_weight=roc_auc_weight,
                    min_roc_auc=min_roc_auc,
                    roc_auc_penalty=roc_auc_penalty,
                )
            else:
                brier_loss, calibration_mae, calibration_std, calibration_coverage, roc_auc, pr_auc, ice, ll, *_, fig = fast_calibration_report(
                    y_true=correct_class,
                    y_pred=y_pred,
                    show_plots=False,
                    use_weights=use_weighted_calibration,
                    verbose=False,
                    mae_weight=mae_weight,
                    std_weight=std_weight,
                    brier_loss_weight=brier_loss_weight,
                    roc_auc_weight=roc_auc_weight,
                    min_roc_auc=min_roc_auc,
                    roc_auc_penalty=roc_auc_penalty,
                )
                class_error = ice

            if verbose:
                print(
                    f"\t class_id={class_id}, BR={brier_loss:.{ndigits}f}, calibration_mae={calibration_mae:.{ndigits}f} ± {calibration_std:.{ndigits}f}, roc_auc={roc_auc:.{ndigits}f}, class_error={class_error:.{ndigits}f}"
                )

        elif method == "brier_score":
            class_error = brier_score_loss(y_true=correct_class, y_prob=y_pred)
        elif method == "precision":
            class_error = fast_precision(y_true=correct_class, y_pred=(y_pred >= 0.5).astype(np.int8), zero_division=0)

        if weight_by_class_npositives:
            weight = correct_class.sum()
        else:
            weight = 1

        total_error += class_error * weight
        weights_sum += weight

    total_error /= weights_sum

    if verbose:
        print(f"method={method}, size={len(correct_class):_} total_error={total_error:.{ndigits}f}")

    return total_error


class CB_EVAL_METRIC:
    """Custom probabilistic prediction error metric balancing predictive power with calibration.
    Can regularly create a calibration plot.
    """

    def __init__(
        self,
        metric: Callable,
        higher_is_better: bool,
        calibration_plot_period: int = 0,
        max_arr_size: int = 0,
    ) -> None:

        # save params
        store_params_in_object(obj=self, params=get_parent_func_args())

        self.nruns = 0

    def is_max_optimal(self):
        return self.higher_is_better

    def evaluate(self, approxes, target, weight):
        output_weight = 1  # weight is not used

        # to avoid expensive train set metric evaluation, we simply return 0 for any input larger than max_arr_size
        if self.max_arr_size and len(approxes[0]) > self.max_arr_size:
            return 0, output_weight

        # For catboost, approxes are logits and need to be converted to true probs first (softmax-ed).
        if len(approxes) == 1:
            y_pred = expit(approxes[0])
            probs = [1 - y_pred, y_pred]
            class_id = 1
        else:
            # For num_classes>1, it requires per-class exponentiation & normalizing by the sum of exp values.
            probs = []
            tot_sum = np.zeros_like(approxes[0])
            for class_id in range(len(approxes)):
                y_pred = np.exp(approxes[class_id])
                probs.append(y_pred)
                tot_sum += y_pred
            for class_id in range(len(approxes)):
                probs[class_id] /= tot_sum

        total_error = self.metric(y_true=target, y_score=probs)

        self.nruns += 1

        # Additional visualization of training process (for the last class_id) is possible.

        if self.calibration_plot_period and (self.nruns % self.calibration_plot_period == 0):
            y_true = (target == class_id).astype(np.int8)
            brier_loss, calibration_mae, calibration_std, calibration_coverage, roc_auc, pr_auc, ice, ll, *_, fig = fast_calibration_report(
                y_true=y_true,
                y_pred=y_pred,
                title=f"{len(approxes[0]):_} records of class {class_id}, integral error={total_error:.4f}, nruns={self.nruns:_}\r\n",
                show_roc_auc_in_title=True,
                use_weights=True,
                verbose=False,
            )

        return total_error, output_weight

    def get_final_error(self, error, weight):
        return error


@njit()
def integral_calibration_error_from_metrics(
    calibration_mae: float,
    calibration_std: float,
    calibration_coverage: float,
    brier_loss: float,
    roc_auc: float,
    mae_weight: float = 3,
    std_weight: float = 2,
    roc_auc_weight: float = 1.5,
    brier_loss_weight: float = 0.4,
    min_roc_auc: float = 0.54,
    roc_auc_penalty: float = 0.00,
) -> float:
    """Compute Integral Calibration Error (ICE) from base ML metrics.
    ICE is a weighted sum of baseline losses-"roc_auc goodness over 0.5".
    If roc_auc is not good enough, it incurs additional penalty.
    """
    res = brier_loss * brier_loss_weight + calibration_mae * mae_weight + calibration_std * std_weight - np.abs(roc_auc - 0.5) * roc_auc_weight
    if np.abs(roc_auc - 0.5) < (min_roc_auc - 0.5):
        res += roc_auc_penalty
    return res


def integral_calibration_error_xgboost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    calibration_mae, calibration_std, calibration_coverage = fast_calibration_metrics(y_true=y_true, y_pred=y_pred)
    return integral_calibration_error_from_metrics(calibration_mae=calibration_mae, calibration_std=calibration_std, calibration_coverage=calibration_coverage)


def integral_calibration_error_keras(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    calibration_mae, calibration_std, calibration_coverage = fast_calibration_metrics(y_true=y_true.numpy()[:, -1], y_pred=y_pred.numpy()[:, -1])
    return integral_calibration_error_from_metrics(calibration_mae=calibration_mae, calibration_std=calibration_std, calibration_coverage=calibration_coverage)


@njit()
def brier_score_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return np.mean((y_true - y_prob) ** 2)


@njit()
def probability_separation_score(y_true: np.ndarray, y_prob: np.ndarray, class_label: int = 1, std_weight: float = 0.5) -> float:
    idx = y_true == class_label
    if idx.sum() == 0:
        return np.nan
        if class_label == 1:
            res = 0.0
        else:
            res = 1.0
    else:
        res = np.mean(y_prob[idx])
        if std_weight != 0.0:
            addend = np.std(y_prob[idx]) * std_weight
            if class_label == 1:
                res = res - addend
            else:
                res = res + addend
    return res


def create_robustness_subgroups(
    df: pd.DataFrame,
    features: Sequence[Union[str, pd.Series]],
    cont_nbins: int = 3,
    min_pop_cat_thresh: Union[float, int] = 1000,
    merge_lowpop_cats: bool = True,
    exclude_terminal_lowpop_cats: bool = True,
    rare_group_name: str = "*RARE*",
) -> dict:
    """Subrouping defines a way of splitting an ndarray into subgroups, for which ML metrics should be calculated separately.
    When we care about not just overall performance on the entire dataset, but also want it to be consistent & fair across subgroups of our observations.
    Subgroups can mean different geograpgical regions, types of clients, types of situations etc, among which we want our performance to be smooth & equal.
    They can simply mean time (say, we want our model to perform equally well on the entire timespan, not just on previous year).

    For categorical variables, natural bin is each category's name.
    If some bins are low-populated (<min_pop_cat_thresh of entire dataset or abs), it's better to join them into a single 'rarevals' bin.
    Subgroups can have different weights (by default equal).

    How is ML metric altered when subgroups are taken into account? From/to the original ML metric on entire dataset, weighted sum of its stdevs over
    subgroups is deducted/added, depending on greater_is_better flag of the metric. Ideally fair model will have zero stdevs & therefore will be unaffected.

    Final ML report:
    subgroup name, nbins, metric stdev, nbad outliers, ngood outliers, best/worst bins names & perf."""

    if isinstance(min_pop_cat_thresh, float):
        assert min_pop_cat_thresh > 0 and min_pop_cat_thresh < 1.0
        min_pop_cat_thresh = int(len(df) * min_pop_cat_thresh)  # convert to abs value
    elif isinstance(min_pop_cat_thresh, int):
        assert min_pop_cat_thresh > 0 and min_pop_cat_thresh <= len(df) // 2

    subgroups = {}
    for feature_name in features:

        if feature_name in ("**ORDER**", "**RANDOM**"):
            subgroups[feature_name] = feature_name
            continue

        if isinstance(feature_name, pd.Series):
            feature_vals = feature_name
            feature_name = feature_vals.name
        else:
            feature_vals = df[feature_name]

        val_cnts = feature_vals.value_counts()

        if feature_vals.dtype.name not in ("category", "object", "date", "datetime"):
            if len(val_cnts) > cont_nbins:
                feature_vals = pd.qcut(feature_vals, q=cont_nbins, labels=None)  # use qcut for equipopulated binning
                val_cnts = feature_vals.value_counts()  # this needs recalculation now

        # use categories as natural bins. ensure that low-populated cats are merged if possible (merge_lowpop_cats)
        # or excluded (exclude_terminal_lowpop_cats).

        rarecats = val_cnts[val_cnts < min_pop_cat_thresh]
        if len(rarecats) > 0:
            cats = rarecats.index.values.tolist()
            if merge_lowpop_cats and rarecats.sum() >= min_pop_cat_thresh:
                # merging is possible
                feature_vals = feature_vals.copy().replace({cat: rare_group_name for cat in cats})
                val_cnts = feature_vals.value_counts()  # this needs recalculation now
                cats_to_use = val_cnts.index.values.tolist()
                logger.info(f"For feature {feature_name}, had to merge {len(cats):_} bins {','.join(map(str,cats))}, {rarecats.sum():_} records.")
            else:
                if exclude_terminal_lowpop_cats:
                    cats_to_use = val_cnts[val_cnts >= min_pop_cat_thresh].index.values.tolist()
                    logger.info(f"For feature {feature_name}, had to exclude {len(cats):_} bins {','.join(map(str,cats))}, {rarecats.sum():_} records.")
        else:
            cats_to_use = val_cnts.index.values.tolist()

        if len(cats_to_use) > 1:
            subgroups[feature_name] = dict(bins=feature_vals, bins_names=cats_to_use)
        else:
            logger.warning(f"Feature {feature_name} can't particiate in subgrouping: it has only one bin.")

    return subgroups


def create_robustness_subgroups_indices(
    subgroups: dict, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, group_weights: dict = {}, cont_nbins: int = 3
) -> dict:
    res = {}
    if len(val_idx) == len(test_idx):
        logger.warning(f"Validation and test sets have the same size. Robustness subgroups estimation will be incorrect.")
    for arr in (train_idx, test_idx, val_idx):
        npoints = len(arr)
        robustness_subgroups_indices = {}
        for group_name, group_params in subgroups.items():
            group_indices = {}
            if group_name in ("**ORDER**", "**RANDOM**"):
                bins, unique_bins = create_robustness_standard_bins(group_name=group_name, npoints=npoints, cont_nbins=cont_nbins)
            else:
                bins = group_params.get("bins")
                assert bins.index.is_unique
                bins = bins.loc[arr]
                unique_bins = None

            if unique_bins is None:
                if isinstance(bins, pd.Series):
                    unique_bins = bins.unique()
                else:
                    unique_bins = np.unique(bins)

            for bin_name in unique_bins:
                idx = bins == bin_name
                group_indices[bin_name] = np.where(idx)[0]

            robustness_subgroups_indices[group_name] = dict(bins=group_indices, weight=group_weights.get(group_name, 1.0))

        res[npoints] = robustness_subgroups_indices

    return res


def create_robustness_standard_bins(group_name: str, npoints: int, cont_nbins: int) -> tuple:

    step_size = npoints // cont_nbins
    bins = np.empty(shape=npoints, dtype=np.int16)
    start = 0
    unique_bins = range(cont_nbins)
    for i in unique_bins:
        bins[start : start + step_size] = i
        start += step_size
    if group_name == "**RANDOM**":
        np.random.shuffle(bins)

    return bins, unique_bins


def compute_robustness_metrics(
    metrics: dict,
    metrics_higher_is_better: dict,
    subgroups: dict,
    subset_index: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cont_nbins: int = 3,
    top_n: int = 5,
) -> pd.DataFrame:
    """* is added to the bin name if bin's metric is an outlier (compted using Tukey's fence & IQR)."""

    if subgroups:

        res = []
        quantile = 0.25
        quantiles_to_compute = [0.5 - quantile, 0.5, 0.5 + quantile]
        tukey_mult = get_tukey_fences_multiplier_for_quantile(quantile=quantile, sd_sigma=2.7)

        for group_name, group_params in subgroups.items():
            if group_name in ("**ORDER**", "**RANDOM**"):
                bins, unique_bins = create_robustness_standard_bins(group_name=group_name, npoints=len(y_true), cont_nbins=cont_nbins)
            else:
                bins = group_params.get("bins")
                if bins is not None:
                    assert subset_index is not None
                    bins = bins.loc[subset_index]
                bins_names = group_params.get("bins_names")
                unique_bins = None

            npoints = []
            perfs = defaultdict(dict)
            if unique_bins is None:
                if isinstance(bins, pd.Series):
                    unique_bins = bins.unique()
                else:
                    unique_bins = np.unique(bins)
            for bin_name in unique_bins:
                idx = bins == bin_name
                n_points = idx.sum()
                if n_points:
                    npoints.append(n_points)
                    for metric_name, metric_func in metrics.items():
                        if y_pred.ndim == 2:
                            metric_value = metric_func(y_true[idx], y_pred[idx, :])
                        else:
                            metric_value = metric_func(y_true[idx], y_pred[idx])
                        perfs[metric_name][f"{bin_name} [{n_points}]"] = metric_value

            for metric_name, metric_perf in perfs.items():

                metric_perf = sort_dict_by_value(metric_perf)
                npoints = np.array(npoints)
                line = dict(
                    factor=group_name,
                    metric=metric_name,
                    nbins=len(unique_bins),
                    npoints_from=npoints.min(),
                    npoints_median=int(np.median(npoints)),
                    npoints_to=npoints.max(),
                )

                # -----------------------------------------------------------------------------------------------------------------------------------------------------
                # Compute quantiles of the metric value.
                # -----------------------------------------------------------------------------------------------------------------------------------------------------

                performances = np.array(list(metric_perf.values()))
                quantiles = np.quantile(performances, q=quantiles_to_compute)
                iqr = quantiles[-1] - quantiles[0]
                min_boundary = quantiles[0] - tukey_mult * iqr
                max_boundary = quantiles[-1] + tukey_mult * iqr

                """
                for q, value in zip(quantiles_to_compute, quantiles):
                    line[f"q{q:.2f}"] = value
                """

                line[f"metric_mean"] = performances.mean()
                line[f"metric_std"] = performances.std()

                # -----------------------------------------------------------------------------------------------------------------------------------------------------
                # Show top-n best/worst groups. postfix * means metric value for the bin is an outlier.
                # -----------------------------------------------------------------------------------------------------------------------------------------------------

                l = len(metric_perf)
                real_top_n = min(l // 2, top_n)

                for i, (bin_name, metric_value) in enumerate(metric_perf.items()):
                    if metric_value < min_boundary or metric_value > max_boundary:
                        postfix = "*"
                    else:
                        postfix = ""
                    if i < real_top_n:
                        if metrics_higher_is_better[metric_name]:
                            line["bin-worst-" + str(i + 1)] = f"{bin_name}: {metric_value:.3f}{postfix}"
                        else:
                            line["bin-best-" + str(i + 1)] = f"{bin_name}: {metric_value:.3f}{postfix}"
                    elif i >= l - real_top_n:
                        if metrics_higher_is_better[metric_name]:
                            line["bin-best-" + str(l - i)] = f"{bin_name}: {metric_value:.3f}{postfix}"
                        else:
                            line["bin-worst-" + str(l - i)] = f"{bin_name}: {metric_value:.3f}{postfix}"

                res.append(line)
        if res:
            res = pd.DataFrame(res).set_index(["factor", "nbins", "npoints_from", "npoints_median", "npoints_to", "metric"])
            return res.reindex(sorted(res.columns), axis=1)


def robust_mlperf_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: Callable,
    higher_is_better: bool,
    subgroups: dict = None,
    whole_set_weight: float = 0.5,
    min_group_size: int = 100,
) -> float:
    """Bins idices need to be aware of arr sizes: boostings can call the metric on
    multiple sets of differnt lengths - train, val, etc. Arrays will be pure numpy, so no other means to
    distinguish except the arr size."""

    weights_sum = whole_set_weight
    total_metric_value = metric(y_true, y_score) * whole_set_weight

    l = len(y_true)
    if subgroups and l in subgroups:

        for group_name, group_params in subgroups[l].items():

            bins = group_params.get("bins")
            bin_weight = group_params.get("weight", 1.0)

            perfs = []
            for bin_name, bin_indices in bins.items():
                if len(bin_indices) < min_group_size:
                    continue
                if isinstance(y_score, Sequence):
                    if len(y_score) == 2:
                        metric_value = metric(y_true[bin_indices], [el[bin_indices] for el in y_score])
                    else:
                        metric_value = metric(y_true[bin_indices], y_score[1][bin_indices])
                else:
                    if y_score.ndim == 2:
                        metric_value = metric(y_true[bin_indices], y_score[bin_indices, :])
                    else:
                        metric_value = metric(y_true[bin_indices], y_score[bin_indices])
                perfs.append(metric_value)

            if perfs:
                perfs = np.array(perfs)
                bin_metric_value = perfs.mean()
                if higher_is_better:
                    bin_metric_value -= perfs.std()
                else:
                    bin_metric_value += perfs.std()

                weights_sum += bin_weight
                total_metric_value += bin_metric_value * bin_weight

    return total_metric_value / weights_sum
