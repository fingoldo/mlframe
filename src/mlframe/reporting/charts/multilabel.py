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
- ``THRESHOLD_SWEEP`` — per-label F1 across a shared threshold grid as a
                     label x threshold heatmap; the F1-optimal threshold per
                     label is marked in the row label. Picking one global 0.5
                     cutoff across labels is wrong when labels have different
                     base rates; this surfaces the per-label optimum at a
                     glance. Vectorised via a per-label probability histogram +
                     reverse-cumsum (no per-threshold recompute, no per-label
                     argsort) over a shared <=200-point grid.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np

from mlframe.reporting.charts._layout import (
    figsize_for_grid, pack_panels, parse_panel_template,
)
from mlframe.reporting.colors import HEATMAP_GENERIC, line_color
from mlframe.reporting.spec import (
    AnnotationPanelSpec, BarPanelSpec, FigureSpec, HeatmapPanelSpec,
    HistogramPanelSpec, LinePanelSpec, PanelSpec,
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


# Shared threshold-grid resolution for the per-label sweep. 200 points reads as a smooth heatmap row and
# is the cap the diagnostic guidance specifies; the optimum-finding is over this grid (not exhaustive).
_SWEEP_N_THRESHOLDS: int = 200


def _uniform_unit_grid(thresholds: np.ndarray) -> bool:
    """True when ``thresholds`` is the uniform ``linspace(0, 1, T)`` grid the njit fast path assumes."""
    T = thresholds.shape[0]
    return bool(T >= 2 and thresholds[0] == 0.0 and thresholds[-1] == 1.0)


def _grid_fire_index(pk: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Per-probability count of grid thresholds it fires at (proba >= t), i.e. searchsorted-right.

    For the uniform ``linspace(0, 1, T)`` grid the index is ``floor(p*(T-1)) + 1`` with a one-step
    comparison correction for the floating-point rounding at exact grid points, making it BIT-IDENTICAL
    to ``np.searchsorted(thresholds, pk, 'right')`` -- the index picks the F1-optimal threshold, so an
    off-by-one column shift would silently move the chosen cutoff and is not acceptable. The hot sweep
    uses the fused njit kernel below; this stays as the readable reference + the non-uniform fallback.
    """
    T = thresholds.shape[0]
    if _uniform_unit_grid(thresholds):
        cand = np.clip(np.floor(pk * (T - 1)).astype(np.int64), 0, T - 1)
        cand -= (thresholds[cand] > pk).astype(np.int64)
        cand = np.clip(cand, -1, T - 1)
        up = (cand + 1 < T) & (thresholds[np.clip(cand + 1, 0, T - 1)] <= pk)
        cand = cand + up.astype(np.int64)
        return np.clip(cand + 1, 0, T)
    return np.clip(np.searchsorted(thresholds, pk, side="right"), 0, T)


def _f1_sweep_uniform_grid(yt: np.ndarray, P: np.ndarray, T: int):
    """Fused single-pass (K, T) F1 sweep on the uniform unit grid via the njit kernel (or numpy fallback).

    One typed O(N) pass per label accumulates the positive/negative grid-fire histograms with the exact
    fire index inline (no length-n temporaries), then reverse-cumsums to TP(t)/FP(t). Bit-identical to the
    ``_grid_fire_index`` reference by construction (same integer index). Falls back to the vectorised numpy
    path when numba is unavailable.
    """
    try:
        from ._threshold_sweep_kernel import f1_sweep_kernel
    except Exception:
        return None
    return f1_sweep_kernel(
        np.ascontiguousarray(yt, dtype=np.uint8),
        np.ascontiguousarray(P, dtype=np.float64),
        int(T),
    )


def _per_label_f1_sweep(y_true: np.ndarray, y_proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """(K, T) F1 matrix: F1 of each label at each shared threshold, fully vectorised.

    For label k and threshold grid edge ``t``, the operating point predicts positive where
    ``proba_k >= t``. TP(t) / FP(t) are decreasing step functions of t; rather than recompute per
    threshold (or argsort per label), each label's positive- and negative-class probabilities are
    histogrammed into the ``T`` grid bins ONCE and reverse-cumsummed, so TP(t) = #positives with proba in
    [t, 1] and FP(t) = #negatives with proba in [t, 1] fall straight out. One O(N) pass per label, no sort.

    On the standard uniform unit grid the per-label pass is a fused njit kernel (no length-n temporaries);
    otherwise the readable numpy histogram path runs. ``thresholds`` is ascending in [0, 1]; F1 is reported
    at each grid point (predict-positive when proba >= threshold), zero-division -> 0 (sklearn default).
    """
    yt = (np.asarray(y_true) == 1)
    P = np.asarray(y_proba, dtype=np.float64)
    K = P.shape[1]
    T = thresholds.shape[0]
    if _uniform_unit_grid(thresholds):
        fast = _f1_sweep_uniform_grid(yt, P, T)
        if fast is not None:
            return fast
    f1 = np.zeros((K, T), dtype=np.float64)
    n_pos = yt.sum(axis=0).astype(np.float64)
    for k in range(K):
        pk = P[:, k]
        pos_mask = yt[:, k]
        # Index of the highest grid threshold each proba still fires at (proba >= t): bins 0..idx-1.
        idx = _grid_fire_index(pk, thresholds)
        pos_idx = idx[pos_mask]
        neg_idx = idx[~pos_mask]
        # bincount over 0..T then drop the final "fires nowhere" overflow bin, reverse-cumsum to get
        # cumulative "fires at grid point >= j" counts.
        pos_hist = np.bincount(pos_idx, minlength=T + 1)[1:].astype(np.float64)
        neg_hist = np.bincount(neg_idx, minlength=T + 1)[1:].astype(np.float64)
        tp = np.cumsum(pos_hist[::-1])[::-1]
        fp = np.cumsum(neg_hist[::-1])[::-1]
        pred_pos = tp + fp
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = pred_pos + n_pos[k]
            f1[k] = np.where(denom > 0, 2.0 * tp / denom, 0.0)
    return f1


def _threshold_sweep_panel(y_true, y_proba, labels) -> PanelSpec:
    """Per-label F1 across a shared threshold grid as a label x threshold heatmap.

    Each row is one label's F1 curve over the threshold grid; the cell color is the F1. The F1-optimal
    threshold per label is folded into the row label (``name @t*=0.37``) so the operator reads off the
    per-label cutoff directly -- a single global 0.5 is wrong when labels have different base rates. The
    sweep is vectorised per label via a probability histogram + reverse-cumsum (no per-threshold recompute,
    no per-label argsort); the grid is capped at ``_SWEEP_N_THRESHOLDS`` points.
    """
    P = np.asarray(y_proba, dtype=np.float64)
    K = P.shape[1]
    if K == 0:
        return AnnotationPanelSpec(
            text="Threshold sweep skipped: no labels",
            title="Per-label F1 threshold sweep",
        )
    thresholds = np.linspace(0.0, 1.0, _SWEEP_N_THRESHOLDS)
    f1 = _per_label_f1_sweep(y_true, P, thresholds)
    best_col = np.argmax(f1, axis=1)
    best_t = thresholds[best_col]
    best_f1 = f1[np.arange(K), best_col]
    row_labels = tuple(
        f"{labels[k]} @t*={best_t[k]:.2f} (F1={best_f1[k]:.2f})" for k in range(K)
    )
    # Column labels: a sparse subset of the grid so the axis stays readable at 200 thresholds.
    n_ticks = min(11, _SWEEP_N_THRESHOLDS)
    tick_pos = np.linspace(0, _SWEEP_N_THRESHOLDS - 1, n_ticks).astype(int)
    col_labels = tuple(
        f"{thresholds[j]:.2f}" if j in set(tick_pos.tolist()) else "" for j in range(_SWEEP_N_THRESHOLDS)
    )
    return HeatmapPanelSpec(
        matrix=f1,
        row_labels=row_labels,
        col_labels=col_labels,
        title="Per-label F1 threshold sweep (row label marks the F1-optimal t*)",
        xlabel="Threshold (predict positive when proba >= t)",
        ylabel="Label",
        colormap=HEATMAP_GENERIC,
        colorbar_label="F1",
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
    "THRESHOLD_SWEEP": _threshold_sweep_panel,
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
