"""Own-implementation confusion-matrix plotter -- a drop-in replacement for ``sklearn.metrics.ConfusionMatrixDisplay``.

Unlike the declarative ``compose_multiclass_figure`` CONFUSION panel (which emits a ``HeatmapPanelSpec`` for the
spec/renderer pipeline), this is a standalone, imperative plotter that returns a matplotlib ``(fig, ax)`` directly --
matching the ergonomics of sklearn's ``ConfusionMatrixDisplay.from_predictions`` for callers that just want a figure.

The matrix itself is computed from OUR confusion kernels (``_multiclass_confusion_kernel``), not sklearn, so the
whole path is sklearn-free at runtime. Rendering uses the same Agg-safe ``Figure`` + ``FigureCanvasAgg`` pattern as
``MatplotlibRenderer`` so it is headless / parallel-safe (no GUI backend, no global ``matplotlib.use``).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from mlframe.reporting.colors import HEATMAP_CMAP, auto_text_color

# Above this many cells the per-cell annotation turns to unreadable soup; suppress the text (matrix still renders).
_CELL_TEXT_MAX = 400


def confusion_matrix_counts(
    y_true, y_pred, *, labels: Optional[Sequence] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(matrix, labels)`` where ``matrix[i, j]`` = count of true==labels[i] predicted as labels[j].

    Uses our njit ``_multiclass_confusion_kernel`` after remapping the (possibly non-contiguous / string) labels to
    positions 0..K-1. Matches ``sklearn.metrics.confusion_matrix`` orientation (rows = true, cols = predicted) and,
    when ``labels`` is None, its label ordering (sorted unique of the union of y_true and y_pred).
    """
    from mlframe.metrics.classification import _multiclass_confusion_kernel

    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if labels is None:
        labels_arr = np.unique(np.concatenate([yt.ravel(), yp.ravel()])) if yt.size or yp.size else np.array([])
    else:
        labels_arr = np.asarray(labels)
    K = len(labels_arr)
    if K == 0:
        return np.zeros((0, 0), dtype=np.int64), labels_arr

    # Map raw labels -> positions 0..K-1 via searchsorted on the sorted label set; unseen labels map to -1 (excluded),
    # mirroring sklearn's ``confusion_matrix`` which drops rows whose label is outside ``labels``.
    sorter = np.argsort(labels_arr, kind="stable")
    sorted_labels = labels_arr[sorter]

    def _to_pos(a: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(sorted_labels, a)
        pos_clipped = np.clip(pos, 0, K - 1)
        matched = sorted_labels[pos_clipped] == a
        return np.where(matched, sorter[pos_clipped], -1).astype(np.int64)

    yt_pos = _to_pos(yt.ravel())
    yp_pos = _to_pos(yp.ravel())
    matrix = _multiclass_confusion_kernel(
        np.ascontiguousarray(yt_pos), np.ascontiguousarray(yp_pos), K,
    )
    return matrix, labels_arr


def _normalize_matrix(matrix: np.ndarray, normalize: Optional[str]) -> np.ndarray:
    """Return a float matrix normalised per sklearn's ``normalize`` in {None, 'true', 'pred', 'all'}."""
    if normalize is None:
        return matrix.astype(np.float64)
    m = matrix.astype(np.float64)
    with np.errstate(all="ignore"):
        if normalize == "true":
            denom = m.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            denom = m.sum(axis=0, keepdims=True)
        elif normalize == "all":
            denom = m.sum()
        else:
            raise ValueError(f"normalize must be one of None, 'true', 'pred', 'all'; got {normalize!r}")
        denom = np.where(denom == 0, 1.0, denom)
        return m / denom


def plot_confusion_matrix(
    y_true,
    y_pred,
    *,
    labels: Optional[Sequence] = None,
    display_labels: Optional[Sequence] = None,
    normalize: Optional[str] = None,
    ax=None,
    cmap: str = HEATMAP_CMAP,
    colorbar: bool = True,
    values_format: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: str = "Predicted label",
    ylabel: str = "True label",
):
    """Plot an annotated confusion-matrix heatmap -- own-implementation replacement for
    ``sklearn.metrics.ConfusionMatrixDisplay.from_predictions``.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n,)
        Ground-truth and predicted class labels (any hashable/orderable dtype; ints or strings).
    labels : sequence, optional
        Class labels to index the matrix (order + subset). Defaults to the sorted unique union of y_true/y_pred,
        matching ``sklearn.metrics.confusion_matrix``.
    display_labels : sequence, optional
        Tick labels shown on the axes; defaults to ``labels``.
    normalize : {None, 'true', 'pred', 'all'}, optional
        Row- / column- / grand-total normalisation, identical to sklearn's ``normalize``.
    ax : matplotlib Axes, optional
        Draw onto this axes (its figure is returned). When None an Agg-backed figure is created headlessly.
    cmap : str
        Colormap name (default CB-safe viridis).
    colorbar : bool
        Draw the colorbar (default True).
    values_format : str, optional
        Format spec for cell text; defaults to ``'d'`` for raw counts and ``'.2f'`` when normalised.
    title, xlabel, ylabel : str
        Axis / figure text.

    Returns
    -------
    (fig, ax) : the matplotlib Figure and Axes.
    """
    import matplotlib

    matrix, labels_arr = confusion_matrix_counts(y_true, y_pred, labels=labels)
    K = len(labels_arr)
    display = _normalize_matrix(matrix, normalize)

    if display_labels is None:
        tick_labels = [str(c) for c in labels_arr]
    else:
        tick_labels = [str(c) for c in display_labels]

    if values_format is None:
        values_format = "d" if normalize is None else ".2f"

    if ax is None:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig = Figure(figsize=(max(4.0, 0.6 * K + 2.0), max(3.5, 0.6 * K + 1.5)), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.figure

    cm = matplotlib.colormaps[cmap]
    im = ax.imshow(display, cmap=cm, aspect="auto")

    ax.set_xticks(range(K))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(K))
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)

    # Per-cell annotations: colour each label for contrast against its own cell (auto_text_color), suppressed past
    # the readability ceiling. Raw counts print as ints even when the display matrix is normalised, matching sklearn's
    # ConfusionMatrixDisplay which annotates the (normalised) displayed value.
    if K and display.size <= _CELL_TEXT_MAX:
        finite = display[np.isfinite(display)]
        if finite.size:
            vmin, vmax = float(finite.min()), float(finite.max())
            for i in range(K):
                for j in range(K):
                    val = float(display[i, j])
                    color = auto_text_color(val if np.isfinite(val) else vmin, cmap, vmin=vmin, vmax=vmax)
                    # Raw-count cells format as ints ('d'); the display matrix is float64 either way, so cast the
                    # value to int for an integer format spec (sklearn prints counts as ints too).
                    cell_val = int(round(val)) if values_format.endswith(("d", "n")) else display[i, j]
                    ax.text(j, i, format(cell_val, values_format), ha="center", va="center", fontsize=8, color=color)

    if colorbar:
        fig.colorbar(im, ax=ax)
    return fig, ax


__all__ = ["plot_confusion_matrix", "confusion_matrix_counts"]
