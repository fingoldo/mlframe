"""Calibration-report chart spec builder.

Produces a 2-row FigureSpec:
- top: ScatterPanelSpec (reliability scatter + perfect-fit diagonal +
  per-bin colormap-driven population colors + inline population labels)
- bottom (optional): HistogramPanelSpec (bin populations as colored bars
  matching the scatter colormap)

Reusable by both backends; replaces the matplotlib-only renderer in
``mlframe.metrics::show_calibration_plot`` (which stays as a back-compat
wrapper).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from mlframe.reporting.colors import CALIBRATION
from mlframe.reporting.spec import (
    FigureSpec, HistogramPanelSpec, ScatterPanelSpec,
)


def _format_population(n: float) -> str:
    """Compact thousands/millions/billions for inline scatter labels."""
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return f"{n:.0f}"


def build_calibration_spec(
    freqs_predicted: np.ndarray,
    freqs_true: np.ndarray,
    hits: np.ndarray,
    *,
    plot_title: str = "",
    show_prob_histogram: bool = True,
    show_inline_population_labels: bool = True,
    label_freq: str = "Observed Frequency",
    label_prob: str = "Predicted Probability",
    label_histogram: str = "Bin population",
    colorbar_label: str = "Bin population",
    figsize: Tuple[float, float] = (12.0, 6.0),
) -> FigureSpec:
    """Build a calibration-report FigureSpec.

    Parameters mirror the legacy ``show_calibration_plot`` signature for
    back-compat. The histogram bottom panel uses the same colormap as
    the scatter so the colorbar reads consistently across both subplots.
    """
    freqs_predicted = np.asarray(freqs_predicted, dtype=np.float64)
    freqs_true = np.asarray(freqs_true, dtype=np.float64)
    hits = np.asarray(hits)

    if len(freqs_predicted) > 1:
        bar_width = float(np.mean(np.diff(np.sort(freqs_predicted))))
    else:
        bar_width = 0.05

    inline_labels: Optional[Tuple[Tuple[float, float, str], ...]] = None
    if show_inline_population_labels and len(hits) > 0:
        inline_labels = tuple(
            (float(x), float(y), _format_population(float(h)))
            for x, y, h in zip(freqs_predicted, freqs_true, hits)
        )

    # 5000 * h / sum(h) is the same scaling as the legacy ``show_calibration_plot``
    # (renders bigger circles for high-population bins).
    total_hits = float(hits.sum()) if len(hits) > 0 else 1.0
    point_size = 5000.0 * np.asarray(hits, dtype=np.float64) / max(total_hits, 1.0)

    scatter = ScatterPanelSpec(
        x=freqs_predicted,
        y=freqs_true,
        title=plot_title,
        xlabel=label_prob if not show_prob_histogram else "",
        ylabel=label_freq,
        perfect_fit_line=True,
        point_color=hits.astype(np.float64),
        colormap=CALIBRATION,
        point_alpha=0.7,
        point_size=point_size,
        inline_labels=inline_labels,
        colorbar_label=colorbar_label,
    )

    if not show_prob_histogram:
        return FigureSpec(
            suptitle="",
            panels=((scatter,),),
            figsize=figsize,
        )

    hist = HistogramPanelSpec(
        values=hits,            # heights = bin populations
        bin_centers=freqs_predicted,
        bin_width=bar_width,
        bar_colors=hits.astype(np.float64),
        colormap=CALIBRATION,
        title="",
        xlabel=label_prob,
        ylabel=label_histogram,
        yscale="linear",
        density=False,
    )

    return FigureSpec(
        suptitle="",
        panels=((scatter,), (hist,)),
        figsize=figsize,
        row_height_ratios=(3.0, 1.0),
        sharex=True,
    )


__all__ = ["build_calibration_spec"]
