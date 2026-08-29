"""Confusion-matrix family for the multiclass composer: the grid, its margins, and the confused-pairs ranking.

Carved out of ``multiclass.py``, which had grown past the house carve band. These three panels answer WHERE the
errors land, from one shared count matrix; the rest of the module answers how well the scores rank and calibrate
per class, from the probability matrix. They share only the class list, which is passed in.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from mlframe.reporting.colors import HEATMAP_CMAP
from mlframe.reporting.spec import (
    AnnotationPanelSpec, BarPanelSpec, ConfusionMarginsPanelSpec, HeatmapPanelSpec, PanelSpec,
)

# Above this K the confusion heatmap K^2 cell-text turns to unreadable soup.
_CONFUSION_TEXT_MAX_K: int = 15
# A confusion RATE needs a denominator: below this many true samples a single misrouted row produces a 50-100% cell
# that outranks every genuine, well-measured confusion on the chart.
_CONFUSED_PAIRS_MIN_SUPPORT: int = 20

def _resolve_pred(y_pred, y_proba) -> np.ndarray:
    """Return ``y_pred`` if supplied, else the nan-safe positional argmax of ``y_proba``.

    The composer computes the hard prediction once and threads it in; this fallback keeps
    each builder independently callable (direct tests / future callers) without it.
    """
    if y_pred is not None:
        return np.asarray(y_pred)
    from ...utils.nan_safe import argmax_classes_safe
    return argmax_classes_safe(np.asarray(y_proba), context="reporting.charts.multiclass")


def _confusion_counts(y_true, y_pred, K: int) -> np.ndarray:
    """K x K raw confusion counts (float64); out-of-range true/pred rows excluded.

    Vectorised tally: flatten (true, pred) into a single linear code and bincount it.
    ``compose_multiclass_figure`` maps unseen true labels to -1 ("excluded") and argmax
    may return a fallback, so out-of-range pairs are masked rather than indexed (the old
    loop silently wrapped -1 into the last row via negative indexing).
    """
    ti = np.asarray(y_true).astype(np.intp)
    pi = np.asarray(y_pred).astype(np.intp)
    valid = (ti >= 0) & (ti < K) & (pi >= 0) & (pi < K)
    return np.bincount(ti[valid] * K + pi[valid], minlength=K * K).reshape(K, K).astype(np.float64)


def _confusion_panel(y_true, y_proba, classes, *, y_pred=None, normalize: bool = True) -> HeatmapPanelSpec:
    """Confusion matrix heatmap.

    ``normalize=True`` (default) row-normalises so each row reads as P(pred | true);
    raw counts hide minority-class confusion because a frequent class dominates the
    color scale. Cell text is suppressed past ``_CONFUSION_TEXT_MAX_K`` classes where
    K^2 annotations turn to soup.

    Counts / row-rates are unsigned magnitudes, so the colormap is the CB-safe sequential
    viridis -- a diverging red/blue map would imply a meaningful zero-centre that does not exist.
    """
    K = len(classes)
    matrix = _confusion_counts(y_true, _resolve_pred(y_pred, y_proba), K)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        display = matrix / row_sums
        title = "Confusion (row-normalised)"
        cbar = "P(pred | true)"
        fmt = ".2f"
    else:
        display = matrix
        title = "Confusion (counts)"
        cbar = "count"
        fmt = ".0f"
    labels = tuple(str(c) for c in classes)
    return HeatmapPanelSpec(
        matrix=display,
        row_labels=labels,
        col_labels=labels,
        title=title,
        xlabel="Predicted",
        ylabel="True",
        colormap=HEATMAP_CMAP,
        cell_text=display if K <= _CONFUSION_TEXT_MAX_K else None,
        text_format=fmt,
        colorbar_label=cbar,
    )


def _confusion_margins_panel(y_true, y_proba, classes, *, y_pred=None, normalize: bool = True) -> PanelSpec:
    """Confusion heatmap flanked by class-support marginal bars.

    The heatmap is identical to ``CONFUSION``; the right bar is per-true-class support (``matrix.sum(axis=1)`` --
    how many samples actually belong to each class) and the top bar is per-predicted-class volume
    (``matrix.sum(axis=0)`` -- how many the model routed there). The two margins are pure row/column sums of the
    already-computed confusion matrix (O(K^2) on the small matrix, no extra full-n pass), and equal
    ``bincount(y_true)`` / ``bincount(y_pred)`` over the in-range pairs. A dominant right-bar reveals imbalance; a
    top-bar exceeding the matching right-bar reveals the model over-predicting that class.
    """
    K = len(classes)
    matrix = _confusion_counts(y_true, _resolve_pred(y_pred, y_proba), K)
    row_margin = matrix.sum(axis=1)  # true-class support
    col_margin = matrix.sum(axis=0)  # predicted-class volume
    total = float(matrix.sum())
    note: Optional[str] = None
    if K <= 1:
        note = "single-class problem"
    elif total == 0:
        note = "no in-range samples"
    elif total < 10:
        note = f"tiny n ({int(total)}) -- margins noisy"
    if normalize and total > 0:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        display = matrix / row_sums
        title = "Confusion + class-support margins (row-normalised)"
        cbar = "P(pred | true)"
        fmt = ".2f"
    else:
        display = matrix
        title = "Confusion + class-support margins (counts)"
        cbar = "count"
        fmt = ".0f"
    labels = tuple(str(c) for c in classes)
    return ConfusionMarginsPanelSpec(
        matrix=display,
        row_margin=row_margin,
        col_margin=col_margin,
        row_labels=labels,
        col_labels=labels,
        title=title,
        colormap=HEATMAP_CMAP,
        cell_text=display if K <= _CONFUSION_TEXT_MAX_K else None,
        text_format=fmt,
        colorbar_label=cbar,
        note=note,
    )


def _confused_pairs_panel(y_true, y_proba, classes, *, y_pred=None, top_n: int = 15) -> PanelSpec:
    """Top-N most-confused (true -> pred) class pairs as a horizontal bar.

    Ranks off-diagonal cells of the ROW-NORMALISED confusion matrix (so a 40%
    misroute of a rare class outranks a 2% leak of a frequent one). Bars read
    "A -> B: x%" with the highest-confusion pair on top.

    Rows below :data:`_CONFUSED_PAIRS_MIN_SUPPORT` true samples are excluded, and every surviving bar carries its
    true-class support. A rate is meaningless without its denominator here: 1 of 2 rows misrouted topped the chart
    at "50%", outranking a 12% leak measured over 40000 rows.
    """
    K = len(classes)
    matrix = _confusion_counts(y_true, _resolve_pred(y_pred, y_proba), K)
    row_totals = matrix.sum(axis=1)
    row_sums = row_totals.reshape(-1, 1).copy()
    row_sums[row_sums == 0] = 1.0
    norm = matrix / row_sums
    off = norm.copy()
    np.fill_diagonal(off, 0.0)
    flat_order = np.argsort(off.ravel())[::-1]
    pairs: List[str] = []
    vals: List[float] = []
    for code in flat_order:
        v = float(off.ravel()[code])
        if v <= 0.0:
            break
        i, j = divmod(int(code), K)
        support = int(row_totals[i])
        if support < _CONFUSED_PAIRS_MIN_SUPPORT:
            continue
        pairs.append(f"{classes[i]} -> {classes[j]} (n={support:,} true)")
        vals.append(v)
        if len(pairs) >= top_n:
            break
    if not pairs:
        return AnnotationPanelSpec(
            text=(
                "No off-diagonal confusion to rank: predictions are perfect, single-class, or every misrouted class "
                f"has fewer than {_CONFUSED_PAIRS_MIN_SUPPORT} true samples."
            ),
            title="Most-confused class pairs",
        )
    # Horizontal bars: long "A -> B" labels read cleanly on the y-axis and the highest-confusion pair sits on top.
    categories = tuple(pairs)
    values = np.asarray(vals, dtype=np.float64)
    return BarPanelSpec(
        categories=categories,
        values=values,
        title=f"Most-confused class pairs (top {len(pairs)}; classes under {_CONFUSED_PAIRS_MIN_SUPPORT} true rows excluded)",
        xlabel="P(pred | true)",
        ylabel="true -> pred",
        orientation="horizontal",
    )


__all__ = []
