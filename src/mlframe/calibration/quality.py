"""Assesing quality of a classifier in terms of how often probabilities predicted by it convert into real occurences.
"""

from __future__ import annotations


# ****************************************************************************************************************************
# Imports
# ****************************************************************************************************************************

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlframe.reporting.spec import FigureSpec

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------


from numba import njit, prange
import pandas as pd, numpy as np
from matplotlib import pyplot as plt

from sklearn.feature_selection import mutual_info_regression
from properscoring import brier_score, crps_ensemble
from sklearn.metrics import (
    median_absolute_error,
    explained_variance_score,
    mean_squared_log_error,
    mean_absolute_percentage_error,
)  # ,mean_pinball_loss

# fast_brier_score_loss is the project's numba Brier, proven sklearn-equivalent by metrics tests;
# use it over sklearn.metrics.brier_score_loss (avoids the sklearn call overhead in the calibration path).
from mlframe.metrics.core import fast_brier_score_loss, fast_r2_score  # sklearn-equivalent, faster

from scipy.stats import ks_1samp, cramervonmises, anderson, chisquare, entropy

# ----------------------------------------------------------------------------------------------------------------------------
# Inits
# ----------------------------------------------------------------------------------------------------------------------------

uniform_cdf = lambda x: x  # CDF of uniform distribution [0, 1]

# ----------------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------------


def mutual_information_score(y: np.ndarray, y_preds: np.ndarray) -> float:
    """Estimate the mutual information (in nats) between true outcomes ``y`` and predictions ``y_preds``.

    Thin wrapper over sklearn's ``mutual_info_regression`` (a k-nearest-neighbour MI estimator,
    Kraskov/Stoegbauer/Grassberger) with ``n_neighbors=2``. Both inputs are reshaped to column
    vectors; the single scalar MI estimate is returned. Higher is better (more shared information
    between prediction and outcome). This is NOT the Hyvarinen score (a proper scoring rule based on
    the score function of the density) despite the historical misnomer -- see the deprecated
    ``hyvarinen_score`` alias below.
    """
    return mutual_info_regression(y.reshape(-1, 1), y_preds.reshape(-1, 1), n_neighbors=2)[0]


def hyvarinen_score(y: np.ndarray, y_preds: np.ndarray) -> float:
    """Deprecated alias for :func:`mutual_information_score`.

    Historically misnamed: this never computed the Hyvarinen score -- it returns a kNN mutual-information
    estimate. Kept as a warning-emitting shim for backward compatibility; use ``mutual_information_score``.
    """
    import warnings

    warnings.warn(
        "hyvarinen_score is a misnomer (it returns a kNN mutual-information estimate, not the "
        "Hyvarinen score) and is deprecated; use mutual_information_score instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return mutual_information_score(y, y_preds)


def crps(y: np.ndarray, y_preds: np.ndarray) -> float:
    """Computes mean Continuous Ranked Probability Score of true binary outcomes versus predicted probabilities."""
    return crps_ensemble(observations=y, forecasts=y_preds).mean()


# ----------------------------------------------------------------------------------------------------------------------------
# Core functionality
# ----------------------------------------------------------------------------------------------------------------------------

METRICS_TO_SHOW = {
    #
    "R2": fast_r2_score,
    # "EV": explained_variance_score,
    #
    # "MSE": mean_squared_error,
    # "MSLE":mean_squared_log_error,
    # "MAE": mean_absolute_error,
    # "MAPE": mean_absolute_percentage_error,
    # "MEAE": median_absolute_error,
    "BR": fast_brier_score_loss,
    # "MI": mutual_information_score,
    "CRPS": crps,
    #
    # "MPL": mean_pinball_loss,
    # "SEP": get_separation_percent,
    # "BL": get_betting_loss,
}


def make_custom_calibration_plot(
    y: np.ndarray,
    probs: np.ndarray,
    nclasses: int,
    classes: list | None = None,
    nbins: int = 100,
    competing_probs: list | None = None,
    X: np.ndarray | None = None,
    display_labels: dict | None = None,
    figsize: tuple = (15, 5),
    skip_plotting: bool = False,
) -> tuple:
    """Custom implementation of calibration plot"""
    if classes is None:
        classes = []
    if competing_probs is None:
        competing_probs = []
    if display_labels is None:
        display_labels = {}

    metrics = {}
    if not classes:
        classes = range(nclasses)
    else:
        nclasses = len(classes)

    if skip_plotting:
        fig, ax_probs = None, None
    else:
        fig, ax_probs = plt.subplots(
            ncols=nclasses,
            nrows=1,
            sharex=False,
            sharey=False,
            figsize=figsize,
        )

    # Non-integer class labels (strings, arbitrary objects) can't index into probs[:, pos_label]
    # or ax_probs[pos_label]. Enumerate classes and use the positional index for indexing;
    # keep the original class value for label/title/metric keys.
    for plot_idx, pos_label in enumerate(classes):

        title = f"Calibration plot for {display_labels.get(pos_label,'class '+str(pos_label))}:"
        # fig.suptitle(title)

        if isinstance(probs, np.ndarray):
            prob_pos = probs[:, plot_idx]
        else:
            prob_pos = probs.iloc[:, plot_idx].values

        if isinstance(y, np.ndarray):
            y_true = (y == pos_label).astype(np.int8)
        elif isinstance(y, (pd.DataFrame, pd.Series)):
            y_true = (y.values == pos_label).astype(np.int8)
        else:
            raise TypeError("Unexpected y type: %s", type(y))

        class_performance_metrics = show_classifier_calibration(
            y_true,
            prob_pos,
            legend_label="Model Probs",
            ax=ax_probs if nclasses == 1 else ax_probs[plot_idx],
            title=title,
            append=False,
            nbins=nbins,
            skip_plotting=skip_plotting,
        )
        metrics[pos_label] = class_performance_metrics

        # Same axis, competing probs, if any

        for competing_vars in competing_probs:
            if len(competing_vars[plot_idx]) > 0:
                var_name = competing_vars[plot_idx]
                prob_pos = X[var_name]
            else:
                named_vars = [var for var in competing_vars if len(var) > 0]

                prob_pos = 1.0 - X[named_vars].sum(axis=1)
                var_name = named_vars[0]  # any of them

            if type(prob_pos) is not np.ndarray:
                prob_pos = prob_pos.values
            var_name = "_".join(var_name.split("_")[1:])
            show_classifier_calibration(
                y_true, prob_pos, legend_label=var_name, ax=ax_probs if nclasses == 1 else ax_probs[plot_idx], title=title, append=True, nbins=nbins
            )
    if skip_plotting:
        plt.close(fig)
    return fig, metrics


# cache=True + nogil=True + fastmath=False: caches compiled code across interpreter runs
# (cuts repeat-session warm-up) and releases the GIL for threaded callers. Signature
# intentionally left unspecified because callers pass both int and float dtypes for y_true.
@njit(cache=True, nogil=True, fastmath=False)
def bin_predictions(
    y_true: np.array,
    y_pred: np.array,
    indices: np.array,
    nbins: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    pockets_predicted, pockets_true = np.zeros(nbins, dtype=np.float64), np.zeros(nbins, dtype=np.float64)
    data = np.zeros((nbins, 4), dtype=np.float64)
    s = len(y_pred)
    l = 0
    bin_size = s // nbins
    for i in range(nbins):
        if i == nbins - 1:
            r = s
        else:
            r = l + bin_size
        # Wave 21 P2: nanmean so a NaN in y_pred/y_true within a bin doesn't
        # poison the (avg_x, avg_y) pair -> propagates into ECE/MCE numbers
        # reported on the calibration chart. Operator may spot the NaN bin
        # but the numeric metrics would be silently wrong.
        # bench-attempt-rejected (2026-07): fusing these two np.nanmean passes into a single scalar
        # nan-aware loop over indices[l:r] was bit-identical but SLOWER at typical n: the vectorized
        # y_pred[indices[l:r]] gather feeds np.nanmean a contiguous SIMD-friendly buffer, whereas the
        # fused loop does two random-access gathers per element (poor cache locality). n=1e6/nbins=20
        # 15.3ms->27.7ms (0.55x), n=1e5/nbins=20+10%NaN 0.58ms->1.27ms (0.46x). Only wins when nbins
        # is large vs n (per-call nanmean overhead dominates), which is not the ECE common case.
        avg_x = np.nanmean(y_pred[indices[l:r]])
        avg_y = np.nanmean(y_true[indices[l:r]])
        pockets_predicted[i] = avg_x
        pockets_true[i] = avg_y
        data[i, :] = np.array([avg_x, avg_y * (r - l), r - l, avg_y], dtype=np.float64)
        l = r
    return pockets_predicted, pockets_true, data


def estimate_calibration_quality_binned(
    y_true: np.array,
    y_pred: np.array,
    nbins: int = 20,
    indices: np.array | None = None,
    metrics_to_show: dict = METRICS_TO_SHOW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    # ECE binning scheme here is EQUAL-MASS (equal-count): predictions are argsorted and split into
    # ``nbins`` equal-population pockets (see ``bin_predictions``). This is NOT comparable to the
    # equal-width-[0,1] ECE in ``calibration/policy._ece_score`` nor the data-adaptive
    # [min,max]-span ECE in ``metrics/calibration/_calibration_metrics.compute_ece_and_brier_decomposition``:
    # the three schemes partition the score axis differently, so their ECE numbers differ on the same
    # (y_true, y_pred) and must not be cross-compared -- compare only within one scheme.
    if indices is None:
        indices = np.argsort(y_pred)
    # With n_samples < nbins the equal-mass bin_size = s // nbins is 0, so every non-final pocket is an empty
    # slice and np.nanmean fills the (avg_x, avg_y) pairs with NaN -> silently NaN-laden ECE/CRPS. Cap nbins
    # to the sample count so each pocket holds at least one row.
    n_samples = len(y_pred)
    if n_samples == 0:
        raise ValueError("estimate_calibration_quality_binned: empty y_pred")
    nbins = min(nbins, n_samples)
    pockets_predicted, pockets_true, data = bin_predictions(y_true=y_true, y_pred=y_pred, indices=indices, nbins=nbins)
    # r2 = np.corrcoef(pockets_predicted, pockets_true)[0, 1] ** 2

    return (
        pockets_predicted,
        pockets_true,
        data,
        {
            # Brier is a per-sample proper scoring rule — compute on raw (y_true, y_pred).
            # All other metrics evaluate calibration curve fidelity — compute on binned pockets.
            fname: (f(y_true, y_pred) if f is fast_brier_score_loss else f(pockets_true, pockets_predicted))
            for fname, f in metrics_to_show.items()
        },
    )


def show_classifier_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    indices: np.ndarray | None = None,
    nbins: int = 20,
    alpha: float = 0.40,
    show_table: bool = False,
    nintervals: int = 1,
    ax: object = None,
    marker_size: int = 15,
    metrics_digits: int = 4,
    connected: bool = True,
    legend_label: str | None = None,
    append: bool = False,
    metrics_to_show: dict = METRICS_TO_SHOW,
    skip_plotting: bool = False,
) -> dict | list | pd.DataFrame | None:

    if nintervals < 1:
        raise ValueError(f"show_classifier_calibration: nintervals must be >= 1, got {nintervals}")

    s = len(y_true)
    step = s // nintervals
    l = 0

    if ax is None:
        ax = plt

    # Collect per-interval performances (previous code only returned the last interval).
    # Initialised before the loop so the show_table / empty-all_performances return branches
    # below never read an unbound name when nintervals == 0 (the loop body never runs).
    all_performances: list = []
    data: list = []
    performances: dict = {}
    for i in range(nintervals):
        if i == nintervals - 1:
            r = s
        else:
            r = l + step

        try:
            x, y, data, performances = estimate_calibration_quality_binned(
                y_true[l:r], y_pred[l:r], nbins=nbins, indices=indices, metrics_to_show=metrics_to_show
            )
        except (ValueError, ZeroDivisionError, IndexError) as e:
            # Expected data-shape / empty-interval failures from binning: log and abort this call, returning None.
            # Narrowed from a bare ``except Exception`` so genuinely unexpected errors (bugs, KeyboardInterrupt,
            # programming errors) propagate instead of being silently swallowed into a None return.
            logger.exception("estimate_calibration_quality_binned failed for slice [%d:%d]", l, r)
            return None
        all_performances.append(performances)

        if not skip_plotting:
            metrics_formatted = " ".join([f"{metric_name}: {round(metric_value,metrics_digits)}" for metric_name, metric_value in performances.items()])

            if legend_label:
                metrics_formatted = legend_label + ": " + metrics_formatted

            if connected:
                ax.plot(x, y, alpha=alpha, label=metrics_formatted, markersize=marker_size, marker="o")
            else:
                ax.scatter(x, y, alpha=alpha, label=metrics_formatted, s=marker_size)
            l = r
    # Derived from title only; needed by the show_table branch too, so it must be bound even when
    # skip_plotting short-circuits the plotting block below.
    is_profit = "profit" in title.lower()
    if not skip_plotting:
        x_min, x_max = np.min(x), np.max(x)
        # y_min, y_max = np.min(y), np.max(y)
        ax.legend(loc="lower right")
        if not append:
            # Set general params for the first time
            ax.plot([x_min, x_max], [x_min, x_max], "g--", label="Perfect")
            try:
                ax.set_xlabel("Expected")
                ax.set_ylabel("Real")
                ax.set_title("%s, %d bins, %d points" % (title, nbins, len(y_true)))
            except (RuntimeError, ValueError, AttributeError):
                pass

            if is_profit:
                ax.axhline(0.0, color="g", linestyle="--")
                ax.axvline(0.0, color="g", linestyle="--")
            # if x_max>=1:
            #    ax.ylim([-.10, 1])
            #    ax.xlim([-.10, 1])
        # plt.show(block=False)
        # plt.pause(0.001)
    if show_table:
        if is_profit:
            return pd.DataFrame(data, columns=["Predicted ROI", "TotalWinnings", "NBets", "Real ROI"])
        else:
            return pd.DataFrame(data, columns=["Prob", "Won", "Predicted", "Freq"])
    else:
        # Return all intervals when more than one was requested; preserve previous
        # single-dict contract for nintervals == 1.
        if nintervals == 1:
            return all_performances[0] if all_performances else performances
        return all_performances


# ---------------------------------------------------------------------------------------------------------------
# Probability Integral Transform (PIT)
# ---------------------------------------------------------------------------------------------------------------
#
# BINARY PIT CAVEAT (applies to every GoF statistic below -- KS / Cramer-von Mises / Anderson-Darling /
# chi-square / ECI / MSD / WPD):
# The PIT construction ``pit = where(y==1, p, 1-p)`` yields a genuinely continuous Uniform(0,1) only for a
# continuous forecast of a continuous outcome. For a BINARY outcome the PIT is a TWO-ATOM MIXTURE (mass at p
# and at 1-p per row), NOT a continuous Uniform(0,1) even when the model is perfectly calibrated. All the
# distribution-vs-uniform GoF tests below therefore SYSTEMATICALLY REJECT well-calibrated binary models (the
# empirical CDF is a step function that cannot match the continuous uniform CDF), so their p-values are not
# interpretable for binary calibration. For binary models prefer a reliability-diagram / ECE-style metric
# (see ``estimate_calibration_quality_binned`` here and ``calibration/policy._ece_score``); treat these PIT
# statistics as relative-ranking diagnostics only, never as absolute calibration hypothesis tests.


def build_pit_diagram_spec(
    pit_values: np.ndarray,
    *,
    caption: str = "",
    bins: int = 20,
    figsize: tuple = (15, 5),
) -> "FigureSpec":
    """Build the single-source PIT-diagram FigureSpec (density histogram + KS-vs-uniform title).

    Same PIT histogram the binary ``PIT`` panel renders; kept here so ``plot_pit_diagram``
    routes through the renderer pipeline instead of a standalone pyplot figure.
    """
    from mlframe.reporting.spec import FigureSpec, HistogramPanelSpec

    pit_values = np.clip(np.asarray(pit_values, dtype=np.float64), 0.0, 1.0)
    ks_stat = kolmogorov_smirnov_statistic(pit_values) if len(pit_values) else float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    heights, _ = np.histogram(pit_values, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    title = (caption + " " if caption else "") + f"PIT Diagram (KS-vs-uniform={ks_stat:.4f})"
    panel = HistogramPanelSpec(
        values=heights,
        bin_centers=centers,
        bin_width=float(edges[1] - edges[0]),
        title=title,
        xlabel="PIT value",
        ylabel="Density",
        density=False,
    )
    return FigureSpec(suptitle="", panels=((panel,),), figsize=figsize)


def plot_pit_diagram(
    predicted_probs: np.ndarray | None = None,
    true_labels: np.ndarray | None = None,
    pit_values: np.ndarray | None = None,
    caption: str = "",
    bins: int = 20,
    figsize: tuple = (15, 5),
    plot_file: str = "",
    plot_outputs: str = "",
) -> None:
    """
    Plots a Probability Integral Transform (PIT) diagram for binary predictions.

    Args:
        predicted_probs (array-like): Predicted probabilities for the positive class.
        true_labels (array-like): Binary true labels (0 or 1).
        bins (int): Number of bins for the histogram.
        plot_file (str): when set, save the figure here (``.png`` appended if no extension).
        plot_outputs (str): optional plot-output DSL (e.g. ``"matplotlib[png] + plotly[html]"``);
            overrides ``plot_file``'s single-format inference when supplied.

    Routes through the spec pipeline (``build_pit_diagram_spec`` + the renderer) so the PIT
    histogram shares the single binary-PIT implementation rather than a standalone pyplot
    figure. The renderer closes its figure afterwards and shows only on an interactive backend.
    """
    if pit_values is None:
        predicted_probs = np.asarray(predicted_probs)
        true_labels = np.asarray(true_labels)
        pit_values = np.where(true_labels == 1, predicted_probs, 1 - predicted_probs)

    from mlframe.reporting.output import parse_plot_output_dsl
    from mlframe.reporting.renderers import render_and_save

    spec = build_pit_diagram_spec(pit_values, caption=caption, bins=bins, figsize=figsize)

    if plot_outputs:
        outputs = parse_plot_output_dsl(plot_outputs)
        base = os.path.splitext(plot_file)[0] if plot_file else "pit_diagram"
    elif plot_file:
        root, ext = os.path.splitext(plot_file)
        fmt = ext.lstrip(".").lower() or "png"
        outputs = parse_plot_output_dsl(f"matplotlib[{fmt}]")
        base = root or "pit_diagram"
    else:
        # No on-disk target: render via matplotlib and show only when interactive (no-op on Agg).
        outputs = parse_plot_output_dsl("matplotlib[png]")
        base = ""

    if base:
        render_and_save(spec, outputs, base)
    else:
        from mlframe.reporting.renderers import get_renderer
        from mlframe.metrics.calibration import _close_unless_interactive, _show_plots_unless_agg
        renderer = get_renderer("matplotlib")
        fig = renderer.render(spec)
        was_shown = _show_plots_unless_agg()
        _close_unless_interactive(fig, was_shown=was_shown)


def kolmogorov_smirnov_statistic(pit_values: np.ndarray) -> float:
    """Calculate the KS statistic for PIT values."""

    statistic, _ = ks_1samp(pit_values, uniform_cdf, alternative="two-sided")
    return statistic


def cramer_von_mises_statistic(pit_values: np.ndarray) -> float:
    """Calculate the Cramér-von Mises statistic for PIT values."""
    result = cramervonmises(pit_values, uniform_cdf)
    return result.statistic


@njit(cache=True, nogil=True, fastmath=True)
def _anderson_darling_kernel(sorted_pit: np.ndarray, n: int) -> float:
    """Fused single-pass A-D accumulation over the already-sorted PIT array.

    Replaces the numpy path's ``arange`` + reversed-copy + two full ``log`` arrays + ``clip`` array
    with one in-register loop: each element is clipped to [eps, 1-eps], log'd, paired with its
    order-symmetric partner, and weighted by ``(2k-1)`` in a single accumulator.
    """
    eps = 1e-12
    acc = 0.0
    for k in range(n):
        a = sorted_pit[k]
        if a < eps:
            a = eps
        elif a > 1.0 - eps:
            a = 1.0 - eps
        b = sorted_pit[n - 1 - k]
        if b < eps:
            b = eps
        elif b > 1.0 - eps:
            b = 1.0 - eps
        acc += (2 * (k + 1) - 1) * (np.log(a) + np.log(1.0 - b))
    return -n - (1.0 / n) * acc


@njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _anderson_darling_kernel_parallel(sorted_pit: np.ndarray, n: int) -> float:
    """``prange`` twin of ``_anderson_darling_kernel`` for large n. The body is a pure ``+=`` accumulation over independent indices, which numba parallelises as a
    reduction (race-free, unlike the ``if x>m:m=x`` running-max form). Output diverges from the serial kernel only by FP reduction-order (~1e-7 relative on a
    goodness-of-fit statistic), never selection-altering. Wins ~5.6x on the kernel at n=10M; gated above ``_AD_PARALLEL_THRESHOLD`` so small inputs keep the
    serial kernel below the prange thread-launch floor."""
    eps = 1e-12
    acc = 0.0
    for k in prange(n):
        a = sorted_pit[k]
        if a < eps:
            a = eps
        elif a > 1.0 - eps:
            a = 1.0 - eps
        b = sorted_pit[n - 1 - k]
        if b < eps:
            b = eps
        elif b > 1.0 - eps:
            b = 1.0 - eps
        acc += (2 * (k + 1) - 1) * (np.log(a) + np.log(1.0 - b))
    return -n - (1.0 / n) * acc


_AD_PARALLEL_THRESHOLD = 200_000


def anderson_darling_statistic(pit_values: np.ndarray) -> float:
    """
    Calculate the Anderson-Darling statistic for a uniform distribution.
    Parameters:
        pit_values (array-like): Array of PIT values.
    Returns:
        float: Anderson-Darling statistic.
    """
    n = len(pit_values)
    # Wave 47 (2026-05-20): (1/n) on empty pit_values divides by zero.
    if n == 0:
        return float("nan")
    sorted_pit = np.sort(np.asarray(pit_values, dtype=np.float64))
    if n >= _AD_PARALLEL_THRESHOLD:
        return _anderson_darling_kernel_parallel(sorted_pit, n)
    return _anderson_darling_kernel(sorted_pit, n)


def chi_square_statistic(pit_values: np.ndarray, bins: int = 10) -> float:
    """Calculate the Chi-Square statistic for PIT values (raw statistic only).

    Returns ONLY the raw Pearson chi-square statistic ``sum (O_i - E_i)^2 / E_i`` over the ``bins``
    equal-width PIT bins against the uniform expectation ``E_i = n / bins``. The associated p-value from
    ``scipy.stats.chisquare`` is deliberately discarded: it assumes ``dof = bins - 1``, but the expected
    counts here are FIXED (uniform, no parameters estimated from the data), and -- see the binary-PIT caveat
    above -- a binary PIT is a two-atom mixture, not a continuous uniform, so the chi-square reference
    distribution does not hold. Compare the raw statistic across models/bins as a relative diagnostic; do
    NOT read it as a calibration hypothesis test. (If a p-value is ever needed, recompute it explicitly with
    the correct dof rather than trusting the default here.)
    """
    # Empty pit_values gives all-zero observed AND expected counts, and chisquare then returns silent NaN
    # (0/0); mirror the n==0 guard in anderson_darling_statistic and surface NaN explicitly.
    if len(pit_values) == 0:
        return float("nan")
    observed, bin_edges = np.histogram(pit_values, bins=bins, range=(0, 1))
    expected = np.ones_like(observed) * len(pit_values) / bins
    chi2_stat, _ = chisquare(f_obs=observed, f_exp=expected)
    return chi2_stat


def entropy_calibration_index(pit_values: np.ndarray, bins: int = 10, miller_madow: bool = True) -> float:
    """Calculate the Entropy-Based Calibration Index (ECI).

    ECI = log(bins) - H(binned PIT). A perfectly-calibrated model has a uniform PIT distribution, true entropy log(bins), and true ECI exactly 0.

    The plug-in entropy estimator H_hat = -sum p_i log p_i is negatively biased at finite n (~ -(K-1)/(2N)), which inflates ECI above 0 and spuriously reports miscalibration even on a calibrated model. ``miller_madow=True`` (default) applies the Miller-Madow correction H_mm = H_hat + (K_obs - 1)/(2N) using the count of NON-EMPTY bins K_obs, which REDUCES (but does not exactly cancel) that leading bias term: the exact -(K-1)/(2N) bias uses the full support size K=bins, so the MM term only fully cancels it when every bin is occupied (K_obs == bins); with empty bins (K_obs < bins) it under-corrects. It is nonetheless a strict improvement over the raw plug-in in practice. Bench `_benchmarks/bench_eci_miller_madow.py`: at n=500, MM is closer to the true 0 in 12/14 cells (bins=10/20 x 7 seeds), mean |ECI| 0.0132 -> 0.0039. Pass ``miller_madow=False`` for the legacy plug-in estimate.

    Uses raw counts normalized to a probability distribution (feeding ``density=True`` to ``scipy.stats.entropy`` gives wrong entropy whenever bin width != 1).
    """
    counts, _ = np.histogram(pit_values, bins=bins, range=(0, 1), density=False)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    uniform_entropy = np.log(bins)
    observed_entropy = entropy(probs)
    if miller_madow:
        k_obs = int(np.count_nonzero(counts))
        observed_entropy = observed_entropy + (k_obs - 1) / (2.0 * total)
    eci = uniform_entropy - observed_entropy
    # The Miller-Madow correction can push observed_entropy above log(bins) on a
    # genuinely-calibrated (near-uniform) input, driving eci negative below its
    # perfect-calibration floor of 0, which is meaningless for a "calibration index".
    # Clamp at 0 (mirrors how _drift.py clamps MM-corrected KL/JS divergences).
    return max(eci, 0.0)


def mean_squared_deviation(pit_values: np.ndarray) -> float:
    """Calculate the Mean Squared Deviation (MSD) from the uniform mean (0.5)."""
    msd = np.mean((pit_values - 0.5) ** 2)
    return msd


def weighted_pit_deviation(pit_values: np.ndarray) -> float:
    """Calculate the Weighted PIT Deviation (WPD)."""
    # Use a larger eps (1e-6) to prevent extreme weights on near-boundary PIT values from
    # dominating the mean. 1e-10 produced weights up to ~1e10 which wrecked numerical stability.
    weights = 1.0 / np.clip(pit_values * (1.0 - pit_values), 1e-6, None)
    wpd = np.mean(weights * (pit_values - 0.5) ** 2)
    return wpd
