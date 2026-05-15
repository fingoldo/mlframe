"""Regression-report 3-panel chart spec builder.

Layout (single row, 3 columns):
- left: ScatterPanelSpec (predictions vs true values, with perfect-fit diagonal)
- middle: HistogramPanelSpec (residuals + fitted Normal overlay; carries
  the noise-distribution hypothesis + suggested loss in its title)
- right: ScatterPanelSpec (residuals vs predicted -- heteroscedasticity panel)

Figure-level ``suptitle`` carries the model identity (split / model_name +
[features/rows]) per the 2026-05-08 layout split.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import numpy as np

from mlframe.reporting.colors import CALIBRATION, NORMAL_OVERLAY, ZERO_LINE
from mlframe.reporting.spec import (
    FigureSpec, HistogramPanelSpec, ScatterPanelSpec,
)


def build_regression_panel_spec(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    audit: Any,                       # ResidualAudit instance (duck-typed)
    header_str: str = "",             # figure suptitle
    metrics_str: str = "",            # left-panel title (MAE/RMSE/MaxError/R2)
    figsize: Tuple[float, float] = (18.0, 4.0),
    plot_sample_size: int = 5000,
    seed: int = 42,
) -> FigureSpec:
    """Build the 3-panel regression FigureSpec.

    ``audit`` is a duck-typed ResidualAudit; we read ``mean``, ``std``,
    ``skew``, ``excess_kurt``, ``hypothesis``, ``suggested_loss``,
    ``hetero_significant``, ``hetero_spearman``.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_f = y_true[finite_mask]
    y_pred_f = y_pred[finite_mask]
    residuals = y_true_f - y_pred_f

    # Subsample for plotting so 9M-row datasets don't render 9M points.
    if residuals.size > plot_sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(residuals.size, size=plot_sample_size, replace=False)
        plot_resid = residuals[idx]
        plot_pred = y_pred_f[idx]
        plot_true = y_true_f[idx]
    else:
        plot_resid = residuals
        plot_pred = y_pred_f
        plot_true = y_true_f

    # Sort by predicted for the scatter so the diagonal line draws cleanly.
    sort_order = np.argsort(plot_pred)
    sorted_pred = plot_pred[sort_order]
    sorted_true = plot_true[sort_order]

    scatter = ScatterPanelSpec(
        x=sorted_pred,
        y=sorted_true,
        title=metrics_str,
        xlabel="Predictions",
        ylabel="True values",
        perfect_fit_line=True,
        point_color="steelblue",
        point_alpha=0.3,
        point_size=10.0,
    )

    # Histogram title: skew/kurt + hypothesis + suggested loss.
    hist_n_bins = max(20, min(80, int(math.sqrt(plot_resid.size) if plot_resid.size > 0 else 20)))
    suggested = ""
    if audit is not None and getattr(audit, "suggested_loss", None):
        suggested = audit.suggested_loss.split("(")[0].strip()
    hyp_line = f"hypothesis: {audit.hypothesis}" if audit is not None else ""
    if suggested:
        hyp_line += f" (suggested: {suggested})"
    hist_title = (
        f"Residuals (skew={audit.skew:+.2f}, excess_kurt={audit.excess_kurt:+.2f})"
        if audit is not None else "Residuals"
    )

    hist = HistogramPanelSpec(
        values=plot_resid,
        bins=hist_n_bins,
        title=hist_title + ("\n" + hyp_line if hyp_line else ""),
        xlabel="Residual (y_true - y_pred)",
        ylabel="Density",
        density=True,
        overlay_normal=(audit.mean, audit.std) if (audit is not None and audit.std > 0) else None,
    )

    het_marker = "(!) heteroscedastic" if (audit is not None and audit.hetero_significant) else "homoscedastic"
    spearman_line = (
        f"spearman(|resid|, y_hat) = {audit.hetero_spearman:+.3f}"
        if audit is not None else ""
    )
    resid_vs_pred = ScatterPanelSpec(
        x=plot_pred,
        y=plot_resid,
        title=f"Residuals vs predicted ({het_marker})\n{spearman_line}",
        xlabel="Predicted (y_hat)",
        ylabel="Residual",
        point_color="steelblue",
        point_alpha=0.3,
        point_size=10.0,
        # Perfect_fit_line would be y=x; for resid-vs-pred we want y=0 line
        # instead. Renderers lack a generic axhline spec field; add one in
        # PR2 if needed. For now the zero-line is implicit via gridlines.
    )

    return FigureSpec(
        suptitle=header_str,
        panels=((scatter, hist, resid_vs_pred),),
        figsize=figsize,
        suptitle_fontsize=11,
    )


__all__ = ["build_regression_panel_spec"]
