"""Stages of ``_report_regression__carry_per_output_aggregated``."""

from __future__ import annotations

# --- imports (managed) ---
import numpy as np
import os
from mlframe._output_paths import ensure_parent_dir
from mlframe.training.reporting._reporting_common import DEFAULT_RANDOM_SEED
from mlframe.training.reporting._reporting_regression import plt
from typing import Any
# --- end imports ---


def _render_per_output_plots(plot_outputs, plot_file, reporting_config, targets, preds, _audit, header_str, metrics_str, plot_sample_size, plot_dpi, figsize, plot_marker, show_perf_chart):
    """Stage of ``_report_regression__carry_per_output_aggregated``."""
    from mlframe.training.targets import (
        plot_residual_diagnostics as _plot_residual_diagnostics,
    )

    if plot_outputs and plot_file:
        # G6: per-call regression panel template override from ReportingConfig (SCATTER / RESID_HIST /
        # RESID_VS_PRED / ERR_BY_DECILE); None falls back to the composer default.
        _regression_panels = getattr(reporting_config, "regression_panels", None)
        _plot_residual_diagnostics(
            targets,
            preds,
            audit=_audit,
            plot_outputs=plot_outputs,
            base_path=plot_file,
            header_str=header_str,
            metrics_str=metrics_str,
            plot_sample_size=plot_sample_size,
            seed=DEFAULT_RANDOM_SEED,
            dpi=plot_dpi,
            panels_template=_regression_panels,
        )
    else:
        # Legacy matplotlib-only path. Kept for callers that
        # still want axes injection (e.g. notebooks composing
        # the chart into a larger figure).
        #
        # Extremes-preserving subsample so the MaxError point quoted in the title (and the axis-anchoring
        # range endpoints) are actually plotted; a uniform draw silently drops exactly those points.
        _preds_arr = np.asarray(preds, dtype=np.float64).ravel()
        _targ_arr = np.asarray(targets, dtype=np.float64).ravel()
        if _preds_arr.size > plot_sample_size:
            from mlframe.reporting.charts import subsample_preserving_extremes

            _resid_for_extremes = _targ_arr - _preds_arr
            idx = subsample_preserving_extremes(
                _preds_arr,
                _targ_arr,
                sample_size=plot_sample_size,
                extreme_values=_resid_for_extremes,
                rng=DEFAULT_RANDOM_SEED,
            )
        else:
            idx = np.arange(_preds_arr.size)
        idx = idx[np.argsort(_preds_arr[idx])]

        # Two-panel figure: scatter | residuals histogram.
        # removed the "Residuals vs predicted"
        # panel; its heteroscedasticity diagnostic (spearman
        # of |resid|, y_hat) now lands inside the scatter
        # title. The remaining two panels each get ~50% wider
        # at the same total figsize.
        # constrained_layout cached solver state -- ~13s saved
        # vs tight_layout per-chart on multi-chart reports.
        # Honour plot_dpi when caller set it.
        _reg_subplots_kwargs: dict[str, Any] = dict(
            figsize=(figsize[0] * 3 / 2, figsize[1]),
            layout="constrained",
        )
        if plot_dpi is not None:
            _reg_subplots_kwargs["dpi"] = plot_dpi
        fig, axes = plt.subplots(1, 2, **_reg_subplots_kwargs)
        ax_scatter, ax_hist = axes

        # y=1.02 puts the suptitle ABOVE the
        # axes region so constrained_layout auto-extends the
        # top margin. Previously y=0.995 placed the suptitle
        # inside the axes row, causing collision with the
        # multiline subplot titles (hist panel carries a
        # 2-line title for hypothesis + suggested loss).
        fig.suptitle(header_str, fontsize=11, y=1.02)

        # Scatter title: metrics + Spearman/heteroscedasticity diagnostic
        # (moved from the dropped 3rd panel so the signal stays visible).
        _scatter_title = metrics_str
        if _audit is not None:
            _het_marker = "(!) heteroscedastic" if _audit.hetero_significant else "homoscedastic"
            if np.isfinite(_audit.hetero_spearman):
                _scatter_title = f"{metrics_str}\nspearman(|resid|, y_hat) = " f"{_audit.hetero_spearman:+.3f} ({_het_marker})"

        ax_scatter.scatter(
            preds[idx],
            targets[idx],
            marker=plot_marker,
            alpha=0.3,
        )
        ax_scatter.plot(
            preds[idx],
            preds[idx],
            linestyle="--",
            color="green",
            label="Perfect fit",
        )
        ax_scatter.set_xlabel("Predictions")
        ax_scatter.set_ylabel("True values")
        ax_scatter.set_title(_scatter_title)
        ax_scatter.grid(True, alpha=0.3)
        ax_scatter.legend(loc="best", fontsize=8, framealpha=0.7)

        if _audit is not None:
            # The residual-diagnostics helper now needs ONLY the hist
            # axis; passing ax_resid_vs_pred=None silences the legacy
            # 3rd-panel render path.
            _plot_residual_diagnostics(
                targets,
                preds,
                audit=_audit,
                ax_hist=ax_hist,
                ax_resid_vs_pred=None,
            )
        else:
            ax_hist.set_visible(False)

        if plot_file:
            # Default the extension to .png when the caller passed an extension-less path -- savefig with no
            # extension writes a file matplotlib can't infer a format for / opens as raw bytes downstream.
            _save_path = plot_file
            if os.path.splitext(plot_file)[1].lower() not in (
                ".png",
                ".pdf",
                ".svg",
                ".jpg",
                ".jpeg",
                ".tif",
                ".tiff",
                ".webp",
            ):
                # Resolved, not concatenated: a flat name lands beside the png/ directory the rest of
                # the run writes into.
                from mlframe.reporting.renderers.save import resolve_output_path

                _save_path = resolve_output_path(plot_file, "matplotlib", "png", multi_output=False)
            fig.savefig(ensure_parent_dir(_save_path))

        if show_perf_chart:
            # block=False (not plt.ion()) so the process-global interactive flag is not leaked into the user session.
            plt.show(block=False)
        # Leak fix: close unless interactive (Jupyter inline).
        from mlframe.metrics.shared import close_unless_interactive as _close_unless_interactive

        _close_unless_interactive(fig, was_shown=show_perf_chart)
