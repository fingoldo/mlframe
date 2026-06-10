"""Calibration plot rendering for ``mlframe.metrics.core``.

Split out from ``core.py`` to keep that file below the 1k-line monolith
threshold. Behaviour preserved bit-for-bit; every moved symbol is
re-exported from ``core`` so existing
``from mlframe.metrics.core import show_calibration_plot`` (and the
other 4 moved names) imports continue to work.

What lives here:
  - ``DEFAULT_TITLE_METRICS_TOKENS`` (constant)
  - ``render_title_metric_token`` (per-token formatter for chart titles)
  - ``fast_calibration_binning`` (NaN-safe histogram for ECE / Brier plots)
  - ``_close_unless_interactive`` (matplotlib figure cleanup helper)
  - ``show_calibration_plot`` (the main reliability-diagram plotter)
"""
from __future__ import annotations

import logging
import math
import os
import sys
import warnings
from math import floor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numba
import matplotlib
from matplotlib import pyplot as plt

# Single source of truth for numba kwargs across mlframe.metrics modules.
from .._numba_params import NUMBA_NJIT_PARAMS

logger = logging.getLogger(__name__)


# 2026-05-28 audit: added KS / MCC / BSS to the default title set
# per user feedback - the most informative single-number summaries
# beyond the existing calibration / AUC family. KS shows score
# discrimination on imbalanced classes; MCC summarises all 4 cells
# of the confusion matrix; BSS shows whether the probabilities beat
# the marginal baseline.
DEFAULT_TITLE_METRICS_TOKENS: tuple = (
    "ICE", "BR_DECOMP", "ECE", "CMAEW", "LL",
    "ROC_AUC", "PR_AUC", "KS", "MCC", "BSS",
)


def render_title_metric_token(
    token: str,
    *,
    ndigits: int,
    ice: float,
    brier_loss: float,
    ece: float,
    brier_reliability: float,
    brier_resolution: float,
    brier_uncertainty: float,
    calibration_mae: float,
    calibration_std: float,
    use_weights: bool,
    calibration_coverage: float,
    nbins: int,
    ll: Optional[float],
    max_hits: int,
    min_hits: int,
    roc_auc: float,
    mean_group_roc_auc: Optional[float],
    pr_auc: float,
    mean_group_pr_auc: Optional[float],
    precision: float,
    recall: float,
    f1: float,
    # 2026-05-28 audit batch additions. Default to NaN so older
    # callers (no extras computed) still render the historical
    # token set without crashing; the new tokens just emit "N/A".
    ks: float = np.nan,
    mcc: float = np.nan,
    bss: float = np.nan,
) -> str:
    """Render one calibration-report title fragment for a token.

    Returns the empty string when the token has no usable data (e.g. LL with
    single-class y_true). Tokens are validated by ReportingConfig at config
    construction time, so unknown tokens cannot reach this function in practice -
    the final ``return ""`` is a defence-in-depth against bypassed validation.

    Percent-suffixed metrics (BR / BR_DECOMP / ECE / CMAEW / PR / RE / F1)
    render with one fewer decimal than ``ndigits`` -- ``%`` already adds
    two extra characters per metric and the headline still reads cleanly
    at ``9.1%`` instead of ``9.10%``. Bare-scalar metrics (ICE, LL,
    ROC_AUC, PR_AUC) keep ``ndigits`` since precision matters more there
    (single-decimal AUC squashes 0.974 vs 0.976 to "1.0"). User feedback
    2026-05-04. ``COV`` derives its own precision from log10(nbins) and
    is unchanged.
    """
    pct_digits = max(0, ndigits - 1)
    if token == "ICE":
        return f"ICE={ice:.{ndigits}f}"
    if token == "BR":
        return f"BR={brier_loss * 100:.{pct_digits}f}%"
    if token == "BR_DECOMP":
        # 2026-04-27 Session 7 batch 8 (user feedback): compact form
        # of the Brier decomposition. The math is BR = REL - RES + UNC
        # (Murphy 1973), so the most informative compact rendering is
        # the actual signed-sum: ``BR=X%(RL<rel>%+U<unc>%-RS<res>%)`` where
        # RL = ReLiability (calibration error, lower is better),
        # U  = Uncertainty (irreducible noise = base_rate * (1-base_rate)),
        # RS = ReSolution (subtractive: how well bins separate from base
        # rate, higher is better). Reads naturally as the formula with
        # signs preserved, ~30% shorter than the labelled form.
        return (
            f"BR={brier_loss * 100:.{pct_digits}f}%"
            f"(RL{brier_reliability * 100:.{pct_digits}f}%"
            f"+U{brier_uncertainty * 100:.{pct_digits}f}%"
            f"-RS{brier_resolution * 100:.{pct_digits}f}%)"
        )
    if token == "ECE":
        return f"ECE={ece * 100:.{pct_digits}f}%"
    if token == "CMAEW":
        return (
            f"CMAE{'W' if use_weights else ''}="
            f"{calibration_mae * 100:.{pct_digits}f}%"
            f"±{calibration_std * 100:.{pct_digits}f}%"
        )
    if token == "COV":
        # log10(nbins) decides COV's decimal precision; matches pre-template behaviour.
        cov_prec = max(0, int(np.log10(max(nbins, 1))))
        return f"COV={calibration_coverage * 100:.{cov_prec}f}%"
    if token == "LL":
        if ll is None:
            return ""
        return f"LL={ll:.{ndigits}f}"
    if token == "DENS":
        return f"DENS=[{max_hits:_};{min_hits:_}]"
    if token == "ROC_AUC":
        suffix = ""
        if mean_group_roc_auc is not None and not np.isnan(mean_group_roc_auc):
            suffix = f"[{mean_group_roc_auc:.{ndigits}f}]"
        if np.isnan(roc_auc):
            return f"ROC AUC=N/A{suffix}"
        return f"ROC AUC={roc_auc:.{ndigits}f}{suffix}"
    if token == "PR_AUC":
        suffix = ""
        if mean_group_pr_auc is not None and not np.isnan(mean_group_pr_auc):
            suffix = f"[{mean_group_pr_auc:.{ndigits}f}]"
        if np.isnan(pr_auc):
            base = f"PR AUC=N/A{suffix}"
        else:
            base = f"PR AUC={pr_auc:.{ndigits}f}{suffix}"
        return (
            f"{base}, PR={precision * 100:.{pct_digits}f}%,"
            f"RE={recall * 100:.{pct_digits}f}%,F1={f1 * 100:.{pct_digits}f}%"
        )
    if token == "KS":
        if np.isnan(ks):
            return "KS=N/A"
        return f"KS={ks:.{ndigits}f}"
    if token == "MCC":
        if np.isnan(mcc):
            return "MCC=N/A"
        return f"MCC={mcc:.{ndigits}f}"
    if token == "BSS":
        # Brier Skill Score - negative means worse than marginal baseline.
        # Keep the sign in the title (it's information, not noise).
        if np.isnan(bss):
            return "BSS=N/A"
        return f"BSS={bss:.{ndigits}f}"
    return ""


@numba.njit(**NUMBA_NJIT_PARAMS)
def fast_calibration_binning(y_true: np.ndarray, y_pred: np.ndarray, nbins: int = 100):
    """Computes bins of predicted vs actual events frequencies. Corresponds to sklearn's UNIFORM strategy.

    ``freqs_predicted`` is the MEAN predicted probability within each bin (sum(y_pred)/hits),
    not the bin centre. The bin centre is wrong-width and biases the reliability x-positions
    (and the downstream calibration_mae) toward the grid rather than where the predictions
    actually sit. The centre is used only as an empty-bin fallback (no present bin can hit it).
    """

    pockets_predicted = np.zeros(nbins, dtype=np.int64)
    pockets_true = np.zeros(nbins, dtype=np.int64)
    pockets_pred_sum = np.zeros(nbins, dtype=np.float64)

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
            pockets_pred_sum[ind] += predicted_prob
    else:
        ind = 0
        for true_class, predicted_prob in zip(y_true, y_pred):
            pockets_predicted[ind] += 1
            pockets_true[ind] += true_class
            pockets_pred_sum[ind] += predicted_prob

    idx = np.nonzero(pockets_predicted > 0)[0]

    hits = pockets_predicted[idx]
    if len(hits) > 0:
        # Mean predicted prob per present bin; bin-centre kept only as the empty-bin fallback geometry.
        centres = (min_val + (np.arange(nbins)[idx] + 0.5) * span / nbins).astype(np.float64)
        freqs_predicted = pockets_pred_sum[idx] / hits
        for b in range(len(hits)):
            if hits[b] == 0:
                freqs_predicted[b] = centres[b]
        freqs_true = pockets_true[idx] / pockets_predicted[idx]
    else:
        freqs_predicted, freqs_true = np.array((), dtype=np.float64), np.array((), dtype=np.float64)

    return freqs_predicted, freqs_true, hits


@numba.njit(**NUMBA_NJIT_PARAMS)
def _quantile_binning_kernel(y_true: np.ndarray, y_pred: np.ndarray, edges: np.ndarray):
    """Bin (y_true, y_pred) into the pockets defined by ascending ``edges`` (len = nbins+1).

    Equal-population (quantile) edges are computed by the Python wrapper; this kernel just
    assigns each sample to its pocket via a linear scan over the (small) edge array and
    returns mean-pred / observed-freq / population per non-empty pocket. Mirrors
    fast_calibration_binning's outputs so downstream metrics are strategy-agnostic.
    """
    nbins = len(edges) - 1
    pockets_predicted = np.zeros(nbins, dtype=np.int64)
    pockets_true = np.zeros(nbins, dtype=np.int64)
    pockets_pred_sum = np.zeros(nbins, dtype=np.float64)

    for i in range(len(y_pred)):
        p = y_pred[i]
        # Find pocket: largest b with edges[b] <= p. Clamp to [0, nbins-1].
        b = 0
        for e in range(1, nbins):
            if p >= edges[e]:
                b = e
            else:
                break
        pockets_predicted[b] += 1
        pockets_true[b] += y_true[i]
        pockets_pred_sum[b] += p

    idx = np.nonzero(pockets_predicted > 0)[0]
    hits = pockets_predicted[idx]
    if len(hits) > 0:
        freqs_predicted = pockets_pred_sum[idx] / hits
        freqs_true = pockets_true[idx] / pockets_predicted[idx]
    else:
        freqs_predicted, freqs_true = np.array((), dtype=np.float64), np.array((), dtype=np.float64)
    return freqs_predicted, freqs_true, hits


def calibration_binning(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nbins: int = 100,
    strategy: str = "auto",
):
    """Bin predictions for a reliability diagram. Strategy dispatcher (default ``"auto"``).

    - ``"uniform"``: equal-width bins over [min, max] (sklearn UNIFORM). Hot njit path.
    - ``"quantile"``: equal-population bins via ``np.quantile`` edges. Rare-event models
      (1-5% positives) concentrate near 0 and collapse into 1-2 uniform bins; quantile
      binning spreads the mass across all ``nbins`` so the reliability diagram is readable.
    - ``"auto"``: quantile when the positive base rate < 10%, else uniform.

    Returns ``(freqs_predicted, freqs_true, hits)`` — same contract as fast_calibration_binning;
    ``freqs_predicted`` is the per-bin mean predicted probability in both strategies.
    """
    if strategy not in ("uniform", "quantile", "auto"):
        raise ValueError(f"strategy must be 'uniform', 'quantile', or 'auto'; got {strategy!r}.")
    n = len(y_pred)
    if n == 0:
        empty = np.array((), dtype=np.float64)
        return empty, empty, np.array((), dtype=np.int64)

    resolved = strategy
    if strategy == "auto":
        base_rate = float(np.mean(y_true))
        resolved = "quantile" if (0.0 < base_rate < 0.10) else "uniform"

    if resolved == "uniform":
        return fast_calibration_binning(y_true=y_true, y_pred=y_pred, nbins=nbins)

    # Quantile: equal-population edges. Dedup collapsed edges (heavy ties at 0) so an
    # empty/degenerate pocket isn't manufactured; fall back to uniform when <2 edges remain.
    qs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(np.asarray(y_pred, dtype=np.float64), qs)
    edges = np.unique(edges)
    if len(edges) < 2:
        return fast_calibration_binning(y_true=y_true, y_pred=y_pred, nbins=nbins)
    return _quantile_binning_kernel(
        np.asarray(y_true).astype(np.int64, copy=False),
        np.asarray(y_pred, dtype=np.float64),
        edges.astype(np.float64),
    )


def _close_unless_interactive(figs, was_shown: bool) -> None:
    """Close one or more matplotlib figures unless we're running inside a
    real IPython / Jupyter kernel.

    Parameters
    ----------
    figs : Figure or iterable of Figures
        The figure(s) to potentially close. ``None`` and empty iterables
        are silently ignored.
    was_shown : bool
        Whether the caller already ran ``plt.ion(); plt.show()``. When
        ``False`` we always close (caller decided not to display); when
        ``True`` we close only outside an interactive Python kernel
        (the Jupyter / IPython inline backends register the figure with
        their display hooks during ``plt.show()``, so closing right
        after preserves the inline render).

    Detection: a true IPython kernel sets the ``__IPYTHON__`` builtin;
    the bare REPL sets ``sys.ps1``. The naive ``"IPython" in
    sys.modules`` heuristic is unreliable -- matplotlib + many ML
    libraries drag IPython into ``sys.modules`` as a transitive
    dependency even from plain Python scripts, giving false positives
    that cause figures to leak across the training-suite calls (~12
    cal-plots / fit, tripping the matplotlib ``More than 20 figures
    have been opened`` warning + a real per-fit slowdown).

    Background: 2026-05-09 leak fix on top of show_calibration_plot;
    extracted as a helper after the same pattern was needed in
    feature_importance.py and training/evaluation.py.
    """
    try:
        is_interactive = bool(__IPYTHON__)  # type: ignore[name-defined]  # noqa: F821
    except NameError:
        is_interactive = hasattr(sys, "ps1")
    if is_interactive and was_shown:
        return  # let the kernel's inline display keep the rendered figure
    # Either we never showed (no display happened), OR we're in a
    # script / GUI session where leaving figures alive just leaks them.
    if figs is None:
        return
    if hasattr(figs, "savefig"):  # single Figure
        plt.close(figs)
        return
    for f in figs:
        plt.close(f)


def _show_plots_unless_agg() -> bool:
    """Call ``plt.ion(); plt.show()`` only when matplotlib is on an
    interactive backend. Returns True iff show was actually invoked.

    Background: when the global backend is ``Agg`` (matplotlib's headless
    default for pytest / CI / scripted runs and the one pinned by
    ``tests/training/conftest.py``), ``plt.show()`` emits the
    "FigureCanvasAgg is non-interactive, and thus cannot be shown"
    UserWarning and renders nothing. The interactive-kernel branch
    (IPython / Jupyter) is already routed through ``display(fig)``
    upstream; this helper guards the bare-Python fallback.
    """
    import matplotlib as _mpl
    if _mpl.get_backend().lower() in {"agg", "pdf", "ps", "svg", "cairo"}:
        return False  # non-interactive backend; plt.show would just warn
    plt.ion()
    plt.show()
    return True


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
    show_prob_histogram: bool = True,
    prob_histogram_yscale: str = "linear",
    show_inline_population_labels: bool = True,
    label_histogram: str = "Bin population",
    plot_outputs: Optional[str] = None,
    base_path: Optional[str] = None,
    dpi: Optional[int] = None,
):
    """Plots reliability digaram from the binned predictions.

    With ``show_prob_histogram=True`` (default) a probability-distribution
    histogram is drawn under the reliability scatter, sharing the X axis.
    Histogram bar heights are bin populations (``hits``) and bars are
    coloured by population using the same ``RdYlBu`` colormap as the
    calibration scatter, so the bottom plot reads consistently with the
    top one (a single bar that matches the colorbar tells you "this bin
    holds N samples"). Y-scale defaults to ``linear`` -- the legacy
    ``"auto"`` mode (log iff max/min skew > 100) flipped to log on
    skewed distributions and made empty bins look populated; pass
    ``prob_histogram_yscale="log"`` if you genuinely need log.
    Inline per-bin population annotations (the small text labels next to each
    scatter point) are independently controlled by
    ``show_inline_population_labels`` so users can keep both, drop both, or
    keep only one.
    """

    # Wave 31 (2026-05-20): assert -> ValueError so -O preserves input validation.
    if backend not in ("plotly", "matplotlib"):
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'; got {backend!r}.")
    if prob_histogram_yscale not in ("auto", "log", "linear"):
        raise ValueError(
            f"prob_histogram_yscale must be 'auto', 'log', or 'linear'; "
            f"got {prob_histogram_yscale!r}."
        )

    # DSL render path (single source of truth for the reliability diagram via
    # build_calibration_spec). Checked BEFORE the no-consumer short-circuit so a
    # caller that supplies plot_outputs + base_path always gets disk artifacts
    # (PNG + plotly HTML) regardless of show_plots / plot_file / session kind.
    # backend="plotly" is also routed here so the legacy inline-plotly branch
    # below never runs -- one PlotlyRenderer, no duplicated styling.
    if (plot_outputs and base_path) or (backend == "plotly" and (plot_file or show_plots)):
        from mlframe.reporting.charts.calibration import build_calibration_spec
        from mlframe.reporting.output import parse_plot_output_dsl
        from mlframe.reporting.renderers import render_and_save
        spec = build_calibration_spec(
            freqs_predicted, freqs_true, hits,
            plot_title=plot_title,
            show_prob_histogram=show_prob_histogram,
            show_inline_population_labels=show_inline_population_labels,
            label_freq=label_freq, label_prob=label_prob,
            label_histogram=label_histogram,
            colorbar_label=colorbar_label,
            figsize=figsize,
            yscale=prob_histogram_yscale,
        )
        if plot_outputs and base_path:
            _outputs = parse_plot_output_dsl(plot_outputs)
            _base = base_path
        else:
            # backend="plotly" with a legacy plot_file: derive the plotly DSL
            # clause from the file extension (os.path.splitext handles
            # extension-less paths correctly) and strip it to form the base path.
            _root, _ext = os.path.splitext(plot_file) if plot_file else ("", "")
            _fmt = _ext.lstrip(".").lower()
            if _fmt not in ("html", "png", "svg", "pdf", "json"):
                _fmt = "html"
            _outputs = parse_plot_output_dsl(f"plotly[{_fmt}]")
            _base = _root or "calibration"
        render_and_save(spec, _outputs, _base)
        return None

    # 2026-05-09: short-circuit when there is NO plot consumer. The
    # ``show_plots=True`` default expresses "render if a human can see
    # it"; in a script / CI / fuzz process (no IPython kernel, no
    # ``sys.ps1``) that's a contradiction -- ``plt.show()`` is a
    # documented no-op (UserWarning: ``FigureCanvasAgg is non-
    # interactive, and thus cannot be shown``) AND nothing is written
    # to disk because ``plot_file`` is empty. The figure render is
    # 130-180 ms / call and dominates the warm wall on plot-heavy
    # multilabel fits (e.g. c0104: 12 cal-plots / fit -> 1.6-2.2 s of
    # pure waste per fit). Caller can opt back in by passing a real
    # ``plot_file`` (saves to disk regardless of session) OR by
    # running inside IPython / Jupyter where the inline display
    # backends will pick up the figure.
    if show_plots and not plot_file:
        try:
            _is_interactive = bool(__IPYTHON__)  # type: ignore[name-defined]  # noqa: F821
        except NameError:
            _is_interactive = hasattr(sys, "ps1")
        if not _is_interactive:
            return None

    if freqs_predicted.size == 0:
        # calibration_binning returns an empty array when all bins are filtered out
        # (single-class preds, all-NaN preds, or the sparse-hits filter at metrics.py:600).
        # np.min/np.max on empty raises an opaque ValueError; bail out with a warning so
        # the surrounding training loop does not crash on degenerate calibration data.
        logger.warning("show_calibration_plot: no bin data available; skipping plot.")
        return None

    # nbins-derived bar width: use the bin centre spacing as the bar width.
    # When all bins have data this matches fast_calibration_binning's geometry;
    # if some bins are empty (sparse hits filter at metrics.py:600) we fall back
    # to the average centre spacing across present bins so bars don't overlap.
    if len(freqs_predicted) > 1:
        _bar_width = float(np.mean(np.diff(np.sort(freqs_predicted))))
    else:
        _bar_width = 0.05

    def _resolve_yscale(hits_arr) -> str:
        """auto -> log iff max/min skew > 100, else linear. Explicit modes pass through."""
        if prob_histogram_yscale != "auto":
            return prob_histogram_yscale
        if len(hits_arr) == 0:
            return "linear"
        max_h = float(np.max(hits_arr))
        min_h = max(float(np.min(hits_arr)), 1.0)
        return "log" if (max_h / min_h) > 100.0 else "linear"

    if backend == "matplotlib":
        # Function to format hits values with B, M, K suffixes
        def format_population(n):
            if n >= 1e9:
                return f"{n/1e9:.1f}B"
            elif n >= 1e6:
                return f"{n/1e6:.1f}M"
            elif n >= 1e3:
                return f"{n/1e3:.1f}K"
            else:
                return f"{n:.0f}"

        def _draw_calibration_axes(ax, fig, draw_xlabel: bool, *, cbar_ax=None):
            """Render the reliability scatter + perfect-calibration line + colorbar on ``ax``.

            ``cbar_ax`` (optional) is the axes list / single ax the
            colorbar attaches to. When the calibration plot stacks
            with a histogram below, pass ``[ax_main, ax_hist]`` so the
            colorbar spans both — otherwise the colorbar steals
            horizontal space from only the calibration axes, making
            the histogram's plot-area visibly wider and breaking the
            shared-X alignment (2026-04-27 user feedback).
            """
            cm = matplotlib.colormaps["RdYlBu"]
            sc = ax.scatter(
                x=freqs_predicted, y=freqs_true, marker="o",
                s=5000 * hits / hits.sum(), c=hits, label=label_freq, cmap=cm,
            )
            ax.plot(
                [min(freqs_predicted), max(freqs_predicted)],
                [min(freqs_predicted), max(freqs_predicted)],
                "g--", label=label_perfect,
            )
            if draw_xlabel:
                ax.set_xlabel(label_prob)
            ax.set_ylabel(label_freq)
            cbar = fig.colorbar(sc, ax=(cbar_ax if cbar_ax is not None else ax))
            cbar.set_label(colorbar_label)
            if show_inline_population_labels:
                vertical_offset = 0.02
                for x, y, hit in zip(freqs_predicted, freqs_true, hits):
                    ax.text(
                        x, y + vertical_offset, format_population(hit),
                        fontsize=8, ha="right", va="bottom",
                    )

        def _draw_histogram_axes(ax):
            """Render the predicted-probability histogram under the calibration axes.

            Bars are coloured by the same ``RdYlBu`` colormap + same
            normalisation as the top calibration scatter, so the colorbar
            reads consistently across both subplots: a tall blue bar in
            the histogram matches the blue scatter bubble at the same X
            (both encode "this bin is populated").
            """
            cm = matplotlib.colormaps["RdYlBu"]
            # Same normalisation as the scatter (which uses ``c=hits`` and
            # auto-normalises across the value range). Reproduce that here:
            _h_min = float(np.min(hits)) if len(hits) else 0.0
            _h_max = float(np.max(hits)) if len(hits) else 1.0
            if _h_max <= _h_min:
                _h_max = _h_min + 1.0
            _bar_colors = cm((hits - _h_min) / (_h_max - _h_min))
            ax.bar(
                freqs_predicted, hits,
                width=_bar_width, align="center",
                color=_bar_colors, edgecolor="white", linewidth=0.5,
            )
            ax.set_xlabel(label_prob)
            ax.set_ylabel(label_histogram)
            ax.set_yscale(_resolve_yscale(hits))

        # Save-only fast path: bypass pyplot + GUI backend (Qt init costs ~1.7s per call).
        # Using Figure + FigureCanvasAgg directly drops this to ~0.2s. Also thread-safe,
        # which matters for parallel val/test evaluation.
        #
        # 2026-05-09: tried extending the guard to also fire for
        # ``show_plots=True`` outside an IPython kernel, hypothesising
        # that the GUI-bound path's pyplot+Qt overhead was wasted in
        # script / CI runs. Two A/B benches on c0104 (5 fits each)
        # showed 2733 +- 236 ms and 3145 +- 723 ms vs the
        # close-fix-only baseline of 2275 +- 185 ms -- a regression,
        # not a win. Revert; keep the original guard. Closing figures
        # in the interactive path (already done above) is enough.
        if plot_file and not show_plots:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            # layout=None: matplotlib's default geometry already fits the
            # multi-axis colorbar + 2-line title with no clipping; the
            # constrained-layout solver adds ~170 ms per call without visual
            # benefit. Honour the dpi kwarg so ReportingConfig.plot_dpi
            # propagates; None defers to matplotlib's default.
            _fig_kwargs = {"figsize": figsize, "layout": None}
            if dpi is not None:
                _fig_kwargs["dpi"] = dpi
            fig = Figure(**_fig_kwargs)
            FigureCanvasAgg(fig)
            if show_prob_histogram:
                gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
                ax_main = fig.add_subplot(gs[0, 0])
                ax_hist = fig.add_subplot(gs[1, 0], sharex=ax_main)
                # Colorbar spans BOTH axes so each subplot loses the
                # same horizontal slice -> X-axes stay aligned via
                # sharex (was: colorbar attached only to ax_main,
                # making ax_hist visually wider — user feedback 2026-04-27).
                _draw_calibration_axes(ax_main, fig, draw_xlabel=False,
                                       cbar_ax=[ax_main, ax_hist])
                _draw_histogram_axes(ax_hist)
                # hide top axes' x tick labels since hist below carries them via sharex
                plt.setp(ax_main.get_xticklabels(), visible=False)
                if plot_title:
                    ax_main.set_title(plot_title)
            else:
                ax = fig.add_subplot(1, 1, 1)
                _draw_calibration_axes(ax, fig, draw_xlabel=True)
                if plot_title:
                    ax.set_title(plot_title)
            # constrained_layout handles spacing automatically — no
            # tight_layout() (which warns + mis-shapes colorbar).
            fig.savefig(plot_file)
            return fig

        # Interactive path (show_plots=True) — keep pyplot so the GUI window is managed.
        # 2026-05-11: layout="constrained" -> layout=None (same rationale as the
        # save-only path above: 1.67x faster, visually equivalent on this
        # 12x6 figsize + multi-axis colorbar + 2-line title geometry).
        if show_prob_histogram:
            _subplots_kwargs = dict(
                nrows=2, ncols=1,
                figsize=figsize,
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
                layout=None,
            )
            if dpi is not None:
                _subplots_kwargs["dpi"] = dpi
            fig, (ax_main, ax_hist) = plt.subplots(**_subplots_kwargs)
            # Colorbar spans both subplots — see _draw_calibration_axes
            # docstring for why (X-axis alignment under sharex).
            _draw_calibration_axes(ax_main, fig, draw_xlabel=False,
                                   cbar_ax=[ax_main, ax_hist])
            _draw_histogram_axes(ax_hist)
            plt.setp(ax_main.get_xticklabels(), visible=False)
            if plot_title:
                ax_main.set_title(plot_title)
        else:
            _fig_kwargs2 = {"figsize": figsize, "layout": None}
            if dpi is not None:
                _fig_kwargs2["dpi"] = dpi
            fig = plt.figure(**_fig_kwargs2)
            ax = fig.add_subplot(1, 1, 1)
            _draw_calibration_axes(ax, fig, draw_xlabel=True)
            if plot_title:
                ax.set_title(plot_title)

        # Default geometry fits this layout (verified via visual A/B
        # against constrained_layout, see bench_calibration_layout.py).

        if plot_file:
            fig.savefig(plot_file)

        # 2026-05-09: ``show_plots=True`` previously ran ``plt.ion();
        # plt.show()`` and left the figure open. In an automated /
        # headless / fuzz / CI run (no REPL, no Jupyter kernel),
        # ``plt.show()`` is either a no-op (Agg canvas, with
        # ``UserWarning: FigureCanvasAgg is non-interactive``) or
        # opens an unowned Qt window the script cannot dismiss; in
        # both cases the figure stayed alive in pyplot's registry,
        # accumulating across the 12+ cal-plot calls per multilabel
        # fit and tripping ``More than 20 figures have been opened``
        # plus a real per-fit slowdown (each new ``plt.subplots`` has
        # to track all live figures). Detect interactive consumer
        # (Python REPL / IPython / Jupyter kernel) by ``sys.ps1`` and
        # ``IPython`` in ``sys.modules``; if neither is present, close
        # the figure right after the (no-op or fire-and-forget)
        # ``plt.show()``. Inline-display backends (Jupyter inline
        # ``%matplotlib inline``, ipympl) register the figure with the
        # display hooks during ``plt.show()`` BEFORE we close it, so
        # the rendered output is preserved.
        if show_plots:
            _show_plots_unless_agg()
        _close_unless_interactive(fig, was_shown=show_plots)

    return fig
