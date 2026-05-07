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


def _pr_f1_panel(y_true, y_proba, labels) -> BarPanelSpec:
    """Per-label precision / recall / F1 bar."""
    from sklearn.metrics import precision_recall_fscore_support

    y_pred = (y_proba >= 0.5).astype(np.int8)
    K = y_true.shape[1]
    p_arr = np.zeros(K)
    r_arr = np.zeros(K)
    f_arr = np.zeros(K)
    for k in range(K):
        p, r, f, _ = precision_recall_fscore_support(
            y_true[:, k], y_pred[:, k], average="binary",
            zero_division=0, labels=[0, 1],
        )
        p_arr[k], r_arr[k], f_arr[k] = p, r, f
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
    return LinePanelSpec(
        x=x_grid,
        y=tuple(series),
        series_labels=tuple(series_labels),
        title="Per-label ROC",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        colors=tuple(line_color(i) for i in range(K)),
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
        observed = np.full(n_bins, np.nan)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.any():
                observed[b] = float(true_k[mask].mean())
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
    y_pred = (y_proba >= 0.5).astype(np.int8)
    K = y_true.shape[1]
    matrix = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        true_mask = y_true[:, i] == 1
        n_true = float(true_mask.sum())
        if n_true == 0:
            continue
        for j in range(K):
            matrix[i, j] = float(y_pred[true_mask, j].sum()) / n_true
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
    true_card = y_true.sum(axis=1).astype(np.int32)
    pred_card = y_pred.sum(axis=1).astype(np.int32)
    bins = np.arange(0, K + 2)  # 0..K
    true_counts = np.zeros(K + 1, dtype=np.int64)
    pred_counts = np.zeros(K + 1, dtype=np.int64)
    for c in true_card:
        if 0 <= c <= K:
            true_counts[c] += 1
    for c in pred_card:
        if 0 <= c <= K:
            pred_counts[c] += 1
    return BarPanelSpec(
        categories=tuple(str(c) for c in range(K + 1)),
        values=(true_counts.astype(np.float64), pred_counts.astype(np.float64)),
        series_labels=("true", "predicted"),
        title="Label cardinality distribution",
        xlabel="# labels per row",
        ylabel="Row count",
    )


def _jaccard_dist_panel(y_true, y_proba, labels) -> HistogramPanelSpec:
    """Per-row Jaccard score distribution."""
    y_pred = (y_proba >= 0.5).astype(np.int8)
    n = y_true.shape[0]
    jaccards = np.zeros(n)
    for i in range(n):
        intersection = int(((y_true[i] == 1) & (y_pred[i] == 1)).sum())
        union = int(((y_true[i] == 1) | (y_pred[i] == 1)).sum())
        jaccards[i] = (intersection / union) if union > 0 else 1.0
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
