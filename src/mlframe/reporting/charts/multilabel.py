"""Multilabel quality-visualisation panels.

Each panel builder takes ``(y_true, y_pred_proba, label_names)`` and
returns one ``PanelSpec``. Inputs are 2-D ``(N, K)`` arrays.

Token catalogue (all 7):
- ``PR_F1``        — per-label precision/recall/F1 grouped bar
- ``ROC``          — per-label ROC curves overlaid
- ``CALIB_GRID``   — per-label reliability curves overlaid
- ``COOCCURRENCE`` — true × predicted label co-occurrence heatmap
                     (off-diagonal = labels co-predicted; diagonal = own
                     label recall)
- ``CARDINALITY``  — distribution of #labels per row (pred vs true grouped bar)
- ``JACCARD_DIST`` — per-row Jaccard score histogram
- ``HAMMING_DIST`` — per-row Hamming distance histogram
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np

from mlframe.reporting.charts._layout import (
    figsize_for_grid, pack_panels, parse_panel_template,
)
from mlframe.reporting.colors import HEATMAP_GENERIC, line_color
from mlframe.reporting.spec import (
    BarPanelSpec, FigureSpec, HeatmapPanelSpec, HistogramPanelSpec,
    LinePanelSpec, PanelSpec,
)


# ----------------------------------------------------------------------------
# Per-token panel builders
# ----------------------------------------------------------------------------


def _per_label_prf1(y_true: np.ndarray, y_pred: np.ndarray):
    """Vectorised per-label precision / recall / F1 for the positive class.

    Replaces K separate ``precision_recall_fscore_support`` calls with one
    pass of per-column confusion counts. ``y_true`` / ``y_pred`` are (N, K)
    {0,1}; the 2-bit code ``2*true + pred`` per column maps to TN/FP/FN/TP,
    tallied with a single bincount over the flattened (column, code) index.
    ``zero_division=0`` semantics (sklearn default) for empty denominators.
    """
    yt = (np.asarray(y_true) == 1).astype(np.intp)
    yp = (np.asarray(y_pred) == 1).astype(np.intp)
    K = yt.shape[1]
    code = (yt * 2 + yp)                                   # (N, K) in {0,1,2,3}
    flat = (np.arange(K) * 4 + code).ravel()               # column-offset code
    counts = np.bincount(flat, minlength=K * 4).reshape(K, 4)
    tp = counts[:, 3].astype(np.float64)
    fp = counts[:, 1].astype(np.float64)
    fn = counts[:, 2].astype(np.float64)
    pred_pos = tp + fp
    true_pos = tp + fn
    precision = np.divide(tp, pred_pos, out=np.zeros(K), where=pred_pos > 0)
    recall = np.divide(tp, true_pos, out=np.zeros(K), where=true_pos > 0)
    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros(K), where=denom > 0)
    return precision, recall, f1


def _pr_f1_panel(y_true, y_proba, labels) -> BarPanelSpec:
    """Per-label precision / recall / F1 bar."""
    y_pred = (y_proba >= 0.5).astype(np.int8)
    p_arr, r_arr, f_arr = _per_label_prf1(y_true, y_pred)
    return BarPanelSpec(
        categories=tuple(str(l) for l in labels),
        values=(p_arr, r_arr, f_arr),
        series_labels=("precision", "recall", "F1"),
        title="Per-label P / R / F1",
        xlabel="Label",
        ylabel="Score",
        xtick_rotation=30.0,
    )


def _roc_panel(y_true, y_proba, labels) -> LinePanelSpec:
    """Per-label ROC curves overlaid."""
    from sklearn.metrics import roc_curve, auc

    K = y_true.shape[1]
    x_grid = np.linspace(0.0, 1.0, 200)
    series: List[np.ndarray] = []
    series_labels: List[str] = []
    for k in range(K):
        bin_y = y_true[:, k].astype(np.int8)
        if bin_y.sum() == 0 or bin_y.sum() == len(bin_y):
            series.append(np.full_like(x_grid, np.nan))
            series_labels.append(f"{labels[k]} (n/a)")
            continue
        fpr, tpr, _ = roc_curve(bin_y, y_proba[:, k])
        roc_auc = auc(fpr, tpr)
        series.append(np.interp(x_grid, fpr, tpr))
        series_labels.append(f"{labels[k]} (AUC={roc_auc:.3f})")
    # Chance diagonal (TPR == FPR) anchors AUC=0.5 so curves below it read as worse-than-random.
    chance = x_grid.copy()
    return LinePanelSpec(
        x=x_grid,
        y=tuple([chance] + series),
        series_labels=tuple(["chance"] + series_labels),
        title="Per-label ROC",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        line_styles=tuple([":"] + ["-"] * K),
        colors=tuple(["gray"] + [line_color(i) for i in range(K)]),
    )


def _calib_grid_panel(y_true, y_proba, labels) -> LinePanelSpec:
    """Per-label reliability curves overlaid."""
    K = y_true.shape[1]
    n_bins = 10
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    x_grid = (edges[:-1] + edges[1:]) / 2

    series: List[np.ndarray] = []
    series_labels: List[str] = []
    for k in range(K):
        proba_k = y_proba[:, k]
        true_k = y_true[:, k].astype(np.float64)
        bin_idx = np.clip(np.digitize(proba_k, edges[1:-1]), 0, n_bins - 1)
        # Per-bin observed mean via two bincounts (sum / count) instead of an inner n_bins x O(N) mask loop.
        counts = np.bincount(bin_idx, minlength=n_bins)
        sums = np.bincount(bin_idx, weights=true_k, minlength=n_bins)
        observed = np.full(n_bins, np.nan)
        nz = counts > 0
        observed[nz] = sums[nz] / counts[nz]
        series.append(observed)
        series_labels.append(str(labels[k]))
    diag = x_grid.copy()
    return LinePanelSpec(
        x=x_grid,
        y=tuple([diag] + series),
        series_labels=tuple(["perfect"] + series_labels),
        title="Per-label reliability",
        xlabel="Mean predicted P(label)",
        ylabel="Observed P(label | bin)",
        line_styles=tuple([":"] + ["-"] * K),
        colors=tuple(["green"] + [line_color(i) for i in range(K)]),
    )


def _cooccurrence_panel(y_true, y_proba, labels) -> HeatmapPanelSpec:
    """Co-occurrence: how often each label was predicted (cols)
    when each label was true (rows). Diagonal = recall; off-diagonal =
    label confusion."""
    K = y_true.shape[1]
    # counts[i, j] = #rows where label i is true AND label j is predicted
    # = (true^T @ pred)[i, j]; a single GEMM replaces the K x K x O(N) double
    # loop (5x on 200k rows / K=10). Rows with no true samples stay zero.
    # Counts are integers <= N (exactly representable in float32, N << 2**24), so the cast +
    # GEMM run in float32 -- ~1.55x vs float64 by halving (N, K) memory traffic; bit-identical
    # result (the K x K ratio divide is promoted back to float64 for an exact P(pred|true)).
    y_pred = (y_proba >= 0.5).astype(np.float32)
    yt = (y_true == 1).astype(np.float32)
    counts = (yt.T @ y_pred).astype(np.float64)
    n_true = yt.sum(axis=0, dtype=np.float64)
    matrix = np.zeros((K, K), dtype=np.float64)
    nz = n_true > 0
    matrix[nz] = counts[nz] / n_true[nz, None]
    label_strs = tuple(str(l) for l in labels)
    return HeatmapPanelSpec(
        matrix=matrix,
        row_labels=label_strs,
        col_labels=label_strs,
        title="Label co-occurrence (true → predicted)",
        xlabel="Predicted label",
        ylabel="True label",
        colormap=HEATMAP_GENERIC,
        cell_text=matrix,
        text_format=".2f",
        colorbar_label="P(predicted | true)",
    )


def _cardinality_panel(y_true, y_proba, labels) -> BarPanelSpec:
    """Bar plot of label cardinality distribution: how many labels per
    row, predicted vs true. Reveals over- / under-prediction."""
    y_pred = (y_proba >= 0.5).astype(np.int8)
    K = y_true.shape[1]
    # Cardinality (#labels per row) is in 0..K by construction, so bincount
    # tallies both histograms directly -- replaces two O(N) Python loops (14x).
    true_card = y_true.sum(axis=1).astype(np.intp)
    pred_card = y_pred.sum(axis=1).astype(np.intp)
    true_counts = np.bincount(true_card, minlength=K + 1)[: K + 1].astype(np.int64)
    pred_counts = np.bincount(pred_card, minlength=K + 1)[: K + 1].astype(np.int64)
    return BarPanelSpec(
        categories=tuple(str(c) for c in range(K + 1)),
        values=(true_counts.astype(np.float64), pred_counts.astype(np.float64)),
        series_labels=("true", "predicted"),
        title="Label cardinality distribution",
        xlabel="# labels per row",
        ylabel="Row count",
    )


def _jaccard_dist_panel(y_true, y_proba, labels) -> HistogramPanelSpec:
    """Per-row Jaccard score distribution.

    History:
    - v1: Python row-loop over N. ~15 s / panel on N=1M K=10.
    - v2: numpy vectorised AND/OR + axis-1 sum + ``np.where``. ~80 ms.
    - v3 (current): numba parallel kernel. ~8 ms on a 6-core box
      (bit-exact equivalent of v2; A/B/C benched on 1M K=10).

    The numba path materialises one ``out`` buffer and walks rows in
    parallel via ``prange``; ``y_true`` is coerced to int8 and
    ``y_proba`` to float32 (matches the input dtypes already produced
    upstream) so the kernel JIT-compiles once and caches.
    """
    from ._jaccard_kernel import jaccard_rows

    y_t_arr = np.ascontiguousarray(np.asarray(y_true), dtype=np.int8)
    y_p_arr = np.ascontiguousarray(np.asarray(y_proba), dtype=np.float32)
    jaccards = jaccard_rows(y_t_arr, y_p_arr)
    return HistogramPanelSpec(
        values=jaccards,
        bins=20,
        title=f"Per-row Jaccard (mean={jaccards.mean():.3f})",
        xlabel="Jaccard score",
        ylabel="Density",
        density=True,
    )


def _hamming_dist_panel(y_true, y_proba, labels) -> HistogramPanelSpec:
    """Per-row Hamming distance distribution."""
    y_pred = (y_proba >= 0.5).astype(np.int8)
    hamming = (y_true != y_pred).sum(axis=1).astype(np.float64)
    return HistogramPanelSpec(
        values=hamming,
        bins=int(y_true.shape[1]) + 1,
        title=f"Per-row Hamming distance (mean={hamming.mean():.3f})",
        xlabel="Hamming distance (# disagreements)",
        ylabel="Density",
        density=True,
    )


# ----------------------------------------------------------------------------
# Token registry + composer
# ----------------------------------------------------------------------------


_TOKEN_BUILDERS: Dict[str, Callable] = {
    "PR_F1": _pr_f1_panel,
    "ROC": _roc_panel,
    "CALIB_GRID": _calib_grid_panel,
    "COOCCURRENCE": _cooccurrence_panel,
    "CARDINALITY": _cardinality_panel,
    "JACCARD_DIST": _jaccard_dist_panel,
    "HAMMING_DIST": _hamming_dist_panel,
}

ALLOWED_MULTILABEL_PANEL_TOKENS = frozenset(_TOKEN_BUILDERS)


def compose_multilabel_figure(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    labels: Sequence,
    *,
    panels_template: str = "PR_F1 CALIB_GRID COOCCURRENCE CARDINALITY JACCARD_DIST",
    suptitle: str = "",
    max_cols: int = 2,
    cell_width: float = 6.0,
    cell_height: float = 4.0,
) -> FigureSpec:
    """Build a multilabel quality FigureSpec from a panel template.

    y_true / y_proba: (N, K) binary / probability matrices.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    if y_true.ndim != 2 or y_proba.ndim != 2:
        raise ValueError(
            f"multilabel panels require 2-D y_true/y_proba; "
            f"got shapes {y_true.shape}, {y_proba.shape}"
        )
    if y_true.shape != y_proba.shape:
        raise ValueError(
            f"y_true {y_true.shape} != y_proba {y_proba.shape}"
        )

    tokens = parse_panel_template(panels_template)
    unknown = [t for t in tokens if t not in _TOKEN_BUILDERS]
    if unknown:
        raise ValueError(
            f"Unknown multilabel panel tokens {unknown}. "
            f"Allowed: {sorted(ALLOWED_MULTILABEL_PANEL_TOKENS)}"
        )
    panels: List[PanelSpec] = [
        _TOKEN_BUILDERS[tok](y_true, y_proba, labels) for tok in tokens
    ]
    grid = pack_panels(panels, max_cols=max_cols)
    n_rows = len(grid)
    n_cols = max_cols if grid else 0
    return FigureSpec(
        suptitle=suptitle,
        panels=grid,
        figsize=figsize_for_grid(n_rows, n_cols,
                                 cell_width=cell_width, cell_height=cell_height),
    )


__all__ = [
    "ALLOWED_MULTILABEL_PANEL_TOKENS",
    "compose_multilabel_figure",
]
