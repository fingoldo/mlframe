"""Training-curve panels: train vs validation metric over boosting iterations.

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


# Metrics where a HIGHER validation value is the optimum; everything else (loss, error, deviance) is minimised.
_HIGHER_IS_BETTER_METRICS = frozenset({"auc", "roc_auc", "aucpr", "pr_auc", "average_precision", "ap", "accuracy",
                                       "f1", "map", "ndcg", "r2", "precision", "recall", "balanced_accuracy"})


def _sampled_positions(length: int, n_iter: int, metric_period: Optional[int]) -> Optional[np.ndarray]:
    """Iteration positions of a series the booster recorded only every ``metric_period`` iterations, else None.

    CatBoost with ``metric_period=k`` logs the learn metric at iterations 0, k, 2k, ... (plus, in some versions, the
    final one), while the validation metric is logged EVERY iteration (early stopping needs it), so the two arrays have
    different lengths. The positions are returned only when ``length`` matches that grid exactly, with or without the
    final iteration; a series that is short for another reason (an eval set that genuinely stopped early) returns None
    and must not be stretched. A production CatBoost chart (metric_period=5, 259 iterations) drew the train curve
    ending at iteration 51 because its 52 points lacked the final iteration and only the 53-point grid was accepted.
    """
    if length < 2 or n_iter < 2:
        return None
    if not metric_period or metric_period <= 1:
        return None
    for k in (int(metric_period),):
        pos = list(range(0, n_iter, k))
        if len(pos) == length:
            return np.asarray(pos, dtype=np.float64)
        if pos[-1] != n_iter - 1 and len(pos) + 1 == length:
            return np.asarray([*pos, n_iter - 1], dtype=np.float64)
    return None


def _metric_panel(
    metric: str,
    splits: Mapping[str, np.ndarray],
    es_iteration: Optional[int],
    metric_period: Optional[int] = None,
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
    # Shared x requires every series to span n_iter. A series recorded every ``metric_period`` iterations is placed at
    # its real iterations and interpolated onto the grid; plotting it against its array index instead squashed it
    # k-fold (the CatBoost train curve "ended" at iteration ~n/5) and paired train[i] with val[i] from different
    # iterations in the gap numbers. A series that genuinely stopped early is right-padded with NaN (renders as a gap).
    if any(s.shape[0] != n_iter for s in series):
        _aligned = []
        for s in series:
            if s.shape[0] == n_iter:
                _aligned.append(s)
                continue
            _pos = _sampled_positions(s.shape[0], n_iter, metric_period)
            if _pos is not None and np.isfinite(s).all():
                _aligned.append(np.interp(x, _pos, s))
            else:
                _aligned.append(np.concatenate([s, np.full(n_iter - s.shape[0], np.nan)]))
        series = _aligned
    vlines = None
    vspans = None
    # The early-stop iteration is carried by the legend (vline label), so the title does not repeat it.
    title = str(metric)
    if es_iteration is not None and 0 <= es_iteration < n_iter:
        vlines = ((float(es_iteration), "firebrick", f"early stop @ {es_iteration}"),)
        if es_iteration < n_iter - 1:
            # Shade the iterations a non-early-stopping fit would have wasted past the ES point.
            vspans = ((float(es_iteration), float(n_iter - 1), "firebrick", 0.08),)

    point_markers = None
    if "train" in splits and "val" in splits:
        _tr, _va = series[labels.index("train")], series[labels.index("val")]
        _both = np.isfinite(_tr) & np.isfinite(_va)
        if _both.any():
            _last = int(np.flatnonzero(_both)[-1])
            _at_stop = int(es_iteration) if (es_iteration is not None and 0 <= es_iteration <= _last and _both[es_iteration]) else _last
            _gap_stop = float(_va[_at_stop] - _tr[_at_stop])
            _gap_last = float(_va[_last] - _tr[_last])
            # The gap WIDENING between the stop and the last iteration is the signal: it means the extra rounds
            # bought train fit that validation never saw. A gap that stays flat is a model that is merely imperfect.
            _widened = abs(_gap_last) - abs(_gap_stop)
            # Kept to one line at the default width; the figure caption explains how to read a widening gap.
            _verdict = "widening" if _widened > 0 else "not widening"
            _where = "early stop" if _at_stop != _last else f"iter {_last}"
            title += "\n" + f"val-train gap {_gap_stop:+.3g} at {_where}, {_gap_last:+.3g} at iter {_last} ({_verdict})"
        # Mark the validation optimum: the iteration the early-stopping rule was trying to find.
        if np.isfinite(_va).any():
            _lower_is_better = str(metric).lower() not in _HIGHER_IS_BETTER_METRICS
            _best = int(np.nanargmin(_va) if _lower_is_better else np.nanargmax(_va))
            point_markers = ((float(_best), float(_va[_best]), f"val optimum @ {_best}", "darkorange", "*"),)

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
        point_markers=point_markers,
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
    metric_period: Optional[int] = None,
) -> FigureSpec:
    """Build a train-vs-val training-curve FigureSpec, one panel per metric.

    Inputs:
    - ``history``: ``{metric_name: {"train": [...], "val": [...]}}`` (aliases tolerated).
    - ``es_iteration``: the early-stopping iteration; marked with a vline + post-ES shading on
      every panel. ``None`` (no early stopping) draws plain curves. Out-of-range values are
      ignored gracefully (no marker) rather than raising.
    - ``metrics``: optional explicit ordering / subset of metric names; default = history order.
    - ``metric_period``: the booster's metric-logging period; a series logged every k-th iteration is placed at its
      real iterations instead of its array index.
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
    panels: List[PanelSpec] = [_metric_panel(m, norm[m], es_iteration, metric_period) for m in order if m in norm]
    # Pack to no more columns than there are panels: padding a lone panel to (panel, None) made the renderer lay out a
    # 2-wide grid inside a figure sized for 1, so the plot filled half the width and its title wrapped into narrow shards.
    grid = pack_panels(panels, max_cols=max(1, min(max_cols, len(panels))))
    n_rows = len(grid)
    n_cols = len(grid[0]) if grid else 1
    return FigureSpec(
        suptitle=suptitle,
        panels=grid,
        figsize=figsize_for_grid(n_rows, n_cols, cell_width=cell_width, cell_height=cell_height),
        caption=(
            "How to read: the train curve says what the model can fit, the validation curve what it can generalise. "
            "The gap between them at the stopping point is the overfitting signal; a validation curve that turns "
            "back up while train keeps falling means the extra rounds are memorising. Validation is also the split "
            "early stopping optimised against, so its final value is optimistic -- quote the test split instead."
        ),
    )


__all__ = ["compose_training_curve_figure", "normalize_history"]
