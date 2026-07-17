"""Regression: interval-width stats must stay non-negative under quantile crossing.

Pre-fix ``_width_dist_panel`` computed width as ``P[:, -1] - P[:, 0]`` (assumes ascending alpha). Under quantile
crossing (non-monotone columns) this went negative -> nonsense widths, and ``widths.mean()`` / ``widths.max()`` ran
before NaN handling. Post-fix width is the column-wise span (max - min across alpha columns) and stats use
``np.nanmean`` / ``np.nanmax``.
"""

import re

import numpy as np

from mlframe.reporting.charts.quantile import _width_dist_panel


def test_width_dist_panel_handles_crossed_quantiles():
    """Width dist panel handles crossed quantiles."""
    rng = np.random.default_rng(0)
    n = 100
    # Three alphas; deliberately CROSS them so the last column sits BELOW the first column for every row.
    alphas = [0.1, 0.5, 0.9]
    base = np.linspace(0.0, 10.0, n)
    span = rng.uniform(1.0, 3.0, n)  # per-row varying width so the histogram path isn't degenerate
    P = np.column_stack([base + span, base + span / 2.0, base + 0.0])  # descending -> crossed
    y_true = base

    panel = _width_dist_panel(y_true, P, alphas)

    # Span-based width (max-min across columns) = `span` per row, always positive despite the crossing.
    m = re.search(r"mean=([-\d.]+), max=([-\d.]+)", panel.title)
    assert m, panel.title
    mean_w, max_w = float(m.group(1)), float(m.group(2))
    assert mean_w >= 0.0 and max_w >= 0.0
    # Title formats stats to 3 decimals, so compare at that precision.
    assert np.isclose(mean_w, float(np.mean(span)), atol=1e-3)
    assert np.isclose(max_w, float(np.max(span)), atol=1e-3)


def test_width_dist_panel_nan_safe_stats():
    """Width dist panel nan safe stats."""
    rng = np.random.default_rng(1)
    n = 20
    alphas = [0.1, 0.9]
    lo = np.zeros(n)
    hi = rng.uniform(2.0, 4.0, n)
    P = np.column_stack([lo, hi])
    P[0, :] = np.nan  # an entirely-NaN row -> its width is NaN
    y_true = np.zeros(n)
    panel = _width_dist_panel(y_true, P, alphas)
    m = re.search(r"mean=([-\d.]+), max=([-\d.]+)", panel.title)
    assert m, panel.title
    mean_w, max_w = float(m.group(1)), float(m.group(2))
    # nanmean/nanmax must ignore the all-NaN row; pre-fix widths.mean()/.max() returned NaN.
    assert np.isfinite(mean_w) and np.isfinite(max_w)
    assert np.isclose(mean_w, float(np.nanmean(hi[1:])), atol=1e-3)
    assert np.isclose(max_w, float(np.max(hi[1:])), atol=1e-3)
