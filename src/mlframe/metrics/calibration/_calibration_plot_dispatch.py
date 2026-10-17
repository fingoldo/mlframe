"""Dispatch / render helpers split out of ``show_calibration_plot`` to keep that function under the length gate.

Helpers that need names defined in ``_calibration_plot`` import them lazily at call time (the parent imports this
module at the top, so a module-level import back would be circular).
"""
from __future__ import annotations

import sys
from typing import Any, Optional

import numpy as np

from mlframe._output_paths import ensure_parent_dir


def render_dsl_calibration_if_requested(
    freqs_predicted: np.ndarray,
    freqs_true: np.ndarray,
    hits: np.ndarray,
    plot_title: str,
    show_prob_histogram: bool,
    show_inline_population_labels: bool,
    label_freq: str,
    label_prob: str,
    label_histogram: str,
    colorbar_label: Optional[str],
    figsize: tuple,
    prob_histogram_yscale: str,
    show_wilson_ci: bool,
    reliability_smoothed: bool,
    raw_probs: Optional[np.ndarray],
    raw_labels: Optional[np.ndarray],
    plot_outputs: Optional[str],
    base_path: Optional[str],
    plot_file: str,
    backend: str,
    show_plots: bool,
) -> bool:
    """Render through the DSL path when it is requested; returns True when it rendered (the caller then returns None)."""
    # DSL render path (single source of truth for the reliability diagram via
    # build_calibration_spec). Checked BEFORE the no-consumer short-circuit so a
    # caller that supplies plot_outputs + base_path always gets disk artifacts
    # (PNG + plotly HTML) regardless of show_plots / plot_file / session kind.
    # backend="plotly" is also routed here so the legacy inline-plotly branch
    # below never runs -- one PlotlyRenderer, no duplicated styling.
    #
    # A bare ``show_plots=True`` (the DEFAULT) must NOT alone count as "save to disk": the
    # matplotlib branch below already treats ``show_plots`` with no real ``plot_file`` as a no-op
    # outside an interactive session (``plt.show()`` is a documented no-op there); the plotly
    # branch used to skip that check entirely, so ``show_calibration_plot(..., backend="plotly")``
    # with every OTHER argument at its default silently wrote a stray ``calibration.html`` into the
    # CURRENT WORKING DIRECTORY every time (falling back to the literal ``"calibration"`` base name
    # when ``plot_file`` is empty) -- confirmed: this reached the repo root via a non-tmp_path test
    # (test_show_calibration_plot_plotly_no_consumer.py), landed on master via `git add -A`, and is
    # the artifact audits/ci_review_2026-09-08/_TRACKER.md's X4 named but never traced to a producer.
    # An explicit ``plot_file`` still forces a save regardless of session, matching every other
    # branch in this function.
    from ._calibration_plot import _show_calibration_p_plot_outputs_base_path

    try:
        _is_interactive_session = bool(__IPYTHON__)  # type: ignore[name-defined]
    except NameError:
        _is_interactive_session = hasattr(sys, "ps1")
    if (plot_outputs and base_path) or (backend == "plotly" and (plot_file or (show_plots and _is_interactive_session))):
        from mlframe.reporting.charts.calibration import build_calibration_spec
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
            show_wilson_ci=show_wilson_ci,
            reliability_smoothed=reliability_smoothed,
            raw_probs=raw_probs,
            raw_labels=raw_labels,
        )
        _base, _outputs = _show_calibration_p_plot_outputs_base_path(plot_outputs, base_path, plot_file)
        render_and_save(spec, _outputs, _base)
        return True
    return False


def no_plot_consumer(show_plots: bool, plot_file: str) -> bool:
    """True when nothing consumes the figure (``show_plots`` set, no ``plot_file``, and no interactive session), so rendering can be skipped."""
    # short-circuit when there is NO plot consumer. The
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
            _is_interactive = bool(__IPYTHON__)  # type: ignore[name-defined]
        except NameError:
            _is_interactive = hasattr(sys, "ps1")
        if not _is_interactive:
            return True
    return False


def save_only_calibration_fig(
    figsize: tuple,
    dpi: Optional[int],
    plot_file: str,
    show_prob_histogram: bool,
    draw_calibration_axes: Any,
    draw_histogram_axes: Any,
    plot_title: str,
) -> Any:
    """Render the figure with ``Figure`` + ``FigureCanvasAgg`` directly (no pyplot / GUI backend), save it to ``plot_file`` and return it."""
    from ._calibration_plot import _show_calibration_p_show_prob_histogram

    # Save-only fast path: bypass pyplot + GUI backend (Qt init costs ~1.7s per call).
    # Using Figure + FigureCanvasAgg directly drops this to ~0.2s. Also thread-safe,
    # which matters for parallel val/test evaluation.
    #
    # tried extending the guard to also fire for
    # ``show_plots=True`` outside an IPython kernel, hypothesising
    # that the GUI-bound path's pyplot+Qt overhead was wasted in
    # script / CI runs. Two A/B benches on c0104 (5 fits each)
    # showed 2733 +- 236 ms and 3145 +- 723 ms vs the
    # close-fix-only baseline of 2275 +- 185 ms -- a regression,
    # not a win. Revert; keep the original guard. Closing figures
    # in the interactive path (already done above) is enough.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    # layout=None: matplotlib's default geometry already fits the
    # multi-axis colorbar + 2-line title with no clipping; the
    # constrained-layout solver adds ~170 ms per call without visual
    # benefit. Honour the dpi kwarg so ReportingConfig.plot_dpi
    # propagates; None defers to matplotlib's default.
    _fig_kwargs: dict[str, Any] = {"figsize": figsize, "layout": None}
    if dpi is not None:
        _fig_kwargs["dpi"] = dpi
    fig = Figure(**_fig_kwargs)
    FigureCanvasAgg(fig)
    _show_calibration_p_show_prob_histogram(show_prob_histogram, fig, draw_calibration_axes, draw_histogram_axes, plot_title)
    # constrained_layout handles spacing automatically -- no
    # tight_layout() (which warns + mis-shapes colorbar).
    fig.savefig(ensure_parent_dir(plot_file))
    return fig


def finish_interactive_calibration_fig(fig: Any, plot_file: str, show_plots: bool) -> None:
    """Save the pyplot figure when ``plot_file`` is set, show it when requested and close it unless the session is interactive."""
    from ._calibration_plot import _close_unless_interactive, _show_plots_unless_agg

    # Default geometry fits this layout (verified via visual A/B
    # against constrained_layout, see bench_calibration_layout.py).

    if plot_file:
        fig.savefig(ensure_parent_dir(plot_file))

    # ``show_plots=True`` previously ran ``plt.ion;
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
