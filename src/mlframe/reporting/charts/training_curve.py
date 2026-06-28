"""Training-curve panels: train vs validation metric over boosting iterations (INV-24).

``compose_training_curve_figure`` turns the generic per-metric history extracted from a
fitted gradient-booster (lgb ``evals_result_`` / xgb ``evals_result()`` /
catboost ``get_evals_result()``) into a FigureSpec with one LinePanelSpec per metric:
train and validation curves vs iteration, the early-stopping iteration marked with a
vline, and the post-ES iterations shaded. Train/val divergence after the ES point is the
overfitting signal the panel exists to expose.

History shape (backend-agnostic)::

    {metric_name: {"train": [...], "val": [...]}}

Each split list is the per-iteration metric. ``train`` / ``val`` are the only recognised
split keys (case-insensitive aliases ``valid`` / ``validation`` / ``test`` map to ``val``);
a metric may carry either or both. The integrator normalises a raw ``evals_result_`` into
this shape before calling -- the composer only consumes the normalised dict so it is
trivially testable on synthetic histories.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from mlframe.reporting.charts._layout import figsize_for_grid, pack_panels
from mlframe.reporting.spec import (
    AnnotationPanelSpec, FigureSpec, LinePanelSpec, PanelSpec,
)

_TRAIN_KEYS = frozenset({"train", "training", "learn"})
_VAL_KEYS = frozenset({"val", "valid", "validation", "test", "eval", "holdout"})


def normalize_history(
    history: Mapping[str, Mapping[str, Sequence[float]]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Collapse split-key aliases to canonical ``train`` / ``val`` and coerce to float arrays.

    Unknown split keys are dropped (the booster sometimes emits an extra eval set the report
    does not care about); a metric left with no recognised split is dropped entirely.
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for metric, splits in history.items():
        norm: Dict[str, np.ndarray] = {}
        for raw_key, series in splits.items():
            key = str(raw_key).strip().lower()
            if key in _TRAIN_KEYS:
                canon = "train"
            elif key in _VAL_KEYS:
                canon = "val"
            else:
                continue
            # First alias wins: a booster never emits two train-like keys for one metric, and
            # silently overwriting would hide a caller bug rather than surface it.
            norm.setdefault(canon, np.asarray(series, dtype=np.float64).ravel())
        if norm:
            out[str(metric)] = norm
    return out


def _metric_panel(
    metric: str,
    splits: Mapping[str, np.ndarray],
    es_iteration: Optional[int],
) -> LinePanelSpec:
    """One metric's train/val curves vs iteration, with the ES point marked + post-ES shaded."""
    series: List[np.ndarray] = []
    labels: List[str] = []
    styles: List[str] = []
    colors: List[str] = []
    n_iter = 0
    if "train" in splits:
        series.append(splits["train"])
        labels.append("train")
        styles.append("-")
        colors.append("steelblue")
        n_iter = max(n_iter, splits["train"].shape[0])
    if "val" in splits:
        series.append(splits["val"])
        labels.append("val")
        styles.append("-")
        colors.append("darkorange")
        n_iter = max(n_iter, splits["val"].shape[0])

    x = np.arange(n_iter, dtype=np.float64)
    # Shared x requires every series to span n_iter; a booster that early-stops one eval set leaves train/val ragged.
    # Right-pad the short series with NaN (renders as a gap) so the shared-x panel stays valid instead of crashing.
    if any(s.shape[0] != n_iter for s in series):
        series = [s if s.shape[0] == n_iter else np.concatenate([s, np.full(n_iter - s.shape[0], np.nan)]) for s in series]
    vlines = None
    vspans = None
    title = f"{metric} vs iteration"
    if es_iteration is not None and 0 <= es_iteration < n_iter:
        vlines = ((float(es_iteration), "firebrick", f"early stop @ {es_iteration}"),)
        if es_iteration < n_iter - 1:
            # Shade the iterations a non-early-stopping fit would have wasted past the ES point.
            vspans = ((float(es_iteration), float(n_iter - 1), "firebrick", 0.08),)
        title = f"{metric} vs iteration (ES @ {es_iteration})"

    return LinePanelSpec(
        x=x,
        y=tuple(series) if len(series) > 1 else series[0],
        series_labels=tuple(labels),
        title=title,
        xlabel="Iteration",
        ylabel=metric,
        line_styles=tuple(styles),
        colors=tuple(colors),
        vlines=vlines,
        vspans=vspans,
    )


def compose_training_curve_figure(
    history: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    es_iteration: Optional[int] = None,
    metrics: Optional[Sequence[str]] = None,
    suptitle: str = "",
    max_cols: int = 2,
    cell_width: float = 9.0,
    cell_height: float = 4.5,
) -> FigureSpec:
    """Build a train-vs-val training-curve FigureSpec, one panel per metric.

    Inputs:
    - ``history``: ``{metric_name: {"train": [...], "val": [...]}}`` (aliases tolerated).
    - ``es_iteration``: the early-stopping iteration; marked with a vline + post-ES shading on
      every panel. ``None`` (no early stopping) draws plain curves. Out-of-range values are
      ignored gracefully (no marker) rather than raising.
    - ``metrics``: optional explicit ordering / subset of metric names; default = history order.
    """
    norm = normalize_history(history)
    if not norm:
        empty = AnnotationPanelSpec(
            text="No train/val history to plot",
            title="Training curves",
        )
        return FigureSpec(
            suptitle=suptitle, panels=((empty,),),
            figsize=figsize_for_grid(1, 1, cell_width=cell_width, cell_height=cell_height),
        )

    order = list(metrics) if metrics is not None else list(norm.keys())
    panels: List[PanelSpec] = [
        _metric_panel(m, norm[m], es_iteration) for m in order if m in norm
    ]
    grid = pack_panels(panels, max_cols=max_cols)
    n_rows = len(grid)
    n_cols = max_cols if n_rows > 1 else len(panels)
    return FigureSpec(
        suptitle=suptitle,
        panels=grid,
        figsize=figsize_for_grid(n_rows, n_cols, cell_width=cell_width, cell_height=cell_height),
    )


__all__ = ["compose_training_curve_figure", "normalize_history"]
