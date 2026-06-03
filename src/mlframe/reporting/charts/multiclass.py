"""Multiclass quality-visualisation panels.

Each panel builder takes ``(y_true, y_pred_proba, classes)`` and returns
a single ``PanelSpec`` instance. ``compose_multiclass_figure`` parses
the panel template (DSL from ``ReportingConfig.multiclass_panels``) and
packs the selected panels into a row-major grid.

Token catalogue (all 7):
- ``CONFUSION``  — row-normalised confusion matrix heatmap
- ``PR_F1``      — per-class precision/recall/F1 grouped bar
- ``ROC``        — per-class ROC curves overlaid
- ``PR_CURVES``  — per-class precision-recall curves overlaid
- ``CALIB_GRID`` — per-class reliability curves overlaid
- ``PROB_DIST``  — per-true-class predicted-probability violins
                   (one violin per true class, showing the distribution
                   of P(y=true_class | x))
- ``TOP_K_ACC``  — top-k accuracy curve (k=1..K)
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mlframe.reporting.charts._layout import (
    figsize_for_grid, pack_panels, parse_panel_template,
)
from mlframe.reporting.colors import CONFUSION as CONFUSION_CMAP, line_color
from mlframe.reporting.spec import (
    BarPanelSpec, FigureSpec, HeatmapPanelSpec, LinePanelSpec, PanelSpec,
    ScatterPanelSpec, ViolinPanelSpec,
)


# ----------------------------------------------------------------------------
# Per-token panel builders
# ----------------------------------------------------------------------------


def _confusion_panel(y_true, y_proba, classes) -> HeatmapPanelSpec:
    """Row-normalised confusion matrix heatmap.

    Cells show the percentage of true-class samples assigned to each
    predicted class; diagonal = correct. Cell text overlays the
    fraction.
    """
    # Wave 21 P2: nan-safe argmax so NaN-proba rows don't silently collapse
    # to class-0 in confusion matrix / per-class metrics.
    from ...utils.nan_safe import argmax_classes_safe
    y_pred = argmax_classes_safe(y_proba, context="reporting.charts.multiclass")
    K = len(classes)
    # Vectorised tally: flatten (true, pred) into a single linear code and
    # bincount it -- replaces the per-sample Python loop (50x on 200k rows).
    # Only in-range 0..K-1 pairs are counted: compose_multiclass_figure maps
    # unseen true labels to -1 (documented "excluded") and argmax may return a
    # fallback, so mask out-of-range rather than indexing them (the old loop
    # silently wrapped -1 into the last row via negative indexing).
    ti = np.asarray(y_true).astype(np.intp)
    pi = np.asarray(y_pred).astype(np.intp)
    valid = (ti >= 0) & (ti < K) & (pi >= 0) & (pi < K)
    matrix = np.bincount(ti[valid] * K + pi[valid], minlength=K * K).reshape(K, K).astype(np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    matrix_norm = matrix / row_sums
    labels = tuple(str(c) for c in classes)
    return HeatmapPanelSpec(
        matrix=matrix_norm,
        row_labels=labels,
        col_labels=labels,
        title="Confusion (row-normalised)",
        xlabel="Predicted",
        ylabel="True",
        colormap=CONFUSION_CMAP,
        cell_text=matrix_norm,
        text_format=".2f",
        colorbar_label="P(pred | true)",
    )


def _pr_f1_panel(y_true, y_proba, classes) -> BarPanelSpec:
    """Per-class precision / recall / F1 grouped bar."""
    from sklearn.metrics import precision_recall_fscore_support

    # Wave 21 P2: nan-safe argmax so NaN-proba rows don't silently collapse
    # to class-0 in confusion matrix / per-class metrics.
    from ...utils.nan_safe import argmax_classes_safe
    y_pred = argmax_classes_safe(y_proba, context="reporting.charts.multiclass")
    K = len(classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(K)), average=None, zero_division=0,
    )
    return BarPanelSpec(
        categories=tuple(str(c) for c in classes),
        values=(np.asarray(precision), np.asarray(recall), np.asarray(f1)),
        series_labels=("precision", "recall", "F1"),
        title="Per-class P / R / F1",
        xlabel="Class",
        ylabel="Score",
    )


def _roc_panel(y_true, y_proba, classes) -> LinePanelSpec:
    """Per-class ROC curves overlaid (one-vs-rest)."""
    from sklearn.metrics import roc_curve, auc

    K = len(classes)
    fprs: List[np.ndarray] = []
    tprs: List[np.ndarray] = []
    labels: List[str] = []
    # Common x-grid for plotting; interpolate each curve onto it so the
    # LinePanelSpec can store one shared x.
    x_grid = np.linspace(0.0, 1.0, 200)
    interpolated: List[np.ndarray] = []
    for k in range(K):
        bin_y = (np.asarray(y_true) == k).astype(np.int8)
        if bin_y.sum() == 0 or bin_y.sum() == len(bin_y):
            # Degenerate class -> flat NaN curve (keeps legend slot).
            interpolated.append(np.full_like(x_grid, np.nan))
            labels.append(f"{classes[k]} (n/a)")
            continue
        fpr, tpr, _ = roc_curve(bin_y, y_proba[:, k])
        roc_auc = auc(fpr, tpr)
        # Interpolate tpr onto the shared fpr grid.
        interp_tpr = np.interp(x_grid, fpr, tpr)
        interpolated.append(interp_tpr)
        labels.append(f"{classes[k]} (AUC={roc_auc:.3f})")
    return LinePanelSpec(
        x=x_grid,
        y=tuple(interpolated),
        series_labels=tuple(labels),
        title="Per-class ROC (one-vs-rest)",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        colors=tuple(line_color(i) for i in range(K)),
    )


def _pr_curves_panel(y_true, y_proba, classes) -> LinePanelSpec:
    """Per-class precision-recall curves overlaid (one-vs-rest)."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    K = len(classes)
    x_grid = np.linspace(0.0, 1.0, 200)
    interpolated: List[np.ndarray] = []
    labels: List[str] = []
    for k in range(K):
        bin_y = (np.asarray(y_true) == k).astype(np.int8)
        if bin_y.sum() == 0:
            interpolated.append(np.full_like(x_grid, np.nan))
            labels.append(f"{classes[k]} (n/a)")
            continue
        precision, recall, _ = precision_recall_curve(bin_y, y_proba[:, k])
        ap = average_precision_score(bin_y, y_proba[:, k])
        # precision-recall: x = recall (descending), y = precision.
        # Interpolate precision onto common ascending recall grid.
        order = np.argsort(recall)
        interp_p = np.interp(x_grid, recall[order], precision[order])
        interpolated.append(interp_p)
        labels.append(f"{classes[k]} (AP={ap:.3f})")
    return LinePanelSpec(
        x=x_grid,
        y=tuple(interpolated),
        series_labels=tuple(labels),
        title="Per-class precision-recall",
        xlabel="Recall",
        ylabel="Precision",
        colors=tuple(line_color(i) for i in range(K)),
    )


def _calib_grid_panel(y_true, y_proba, classes) -> LinePanelSpec:
    """Per-class reliability curves overlaid (small-multiples-as-overlay).

    For each class k, bin predictions into deciles, plot mean predicted
    P(y=k) vs observed P(y=k|bin). Perfect calibration = y=x diagonal.
    """
    K = len(classes)
    n_bins = 10
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    x_grid = (edges[:-1] + edges[1:]) / 2

    series: List[np.ndarray] = []
    labels: List[str] = []
    yt = np.asarray(y_true)
    for k in range(K):
        proba_k = y_proba[:, k]
        true_k = (yt == k).astype(np.float64)
        bin_idx = np.clip(np.digitize(proba_k, edges[1:-1]), 0, n_bins - 1)
        # Per-bin observed mean via two bincounts (sum / count) instead of an
        # inner n_bins x O(N) mask loop.
        counts = np.bincount(bin_idx, minlength=n_bins)
        sums = np.bincount(bin_idx, weights=true_k, minlength=n_bins)
        observed = np.full(n_bins, np.nan)
        nz = counts > 0
        observed[nz] = sums[nz] / counts[nz]
        series.append(observed)
        labels.append(str(classes[k]))
    # Plus the perfect-calibration diagonal as the first series.
    diag = x_grid.copy()
    return LinePanelSpec(
        x=x_grid,
        y=tuple([diag] + series),
        series_labels=tuple(["perfect"] + labels),
        title="Per-class reliability curves",
        xlabel="Mean predicted P(y=k)",
        ylabel="Observed P(y=k | bin)",
        line_styles=tuple([":"] + ["-"] * K),
        colors=tuple(["green"] + [line_color(i) for i in range(K)]),
    )


def _prob_dist_panel(y_true, y_proba, classes) -> ViolinPanelSpec:
    """Per-true-class violin: distribution of P(y=true_class | x).

    Concentration near 1 = high confidence; spread across [0,1] = calibrated
    uncertainty. Always-near-0 violin = model collapse on that class.

    Sampling: on a 1M-row multiclass run the un-sampled per-class slice is
    ~N/K points (e.g. 333k for K=3). Matplotlib's ``violinplot`` runs
    ``gaussian_kde`` per group, which is O(N) in the data and at 333k
    points spends seconds on a single chart — measured at ~10-15s for the
    full PROB_DIST panel on K=3 / N=1M / 28 reports. We cap each group
    at ``DEFAULT_VIOLIN_SAMPLE_CAP`` (5000) which keeps KDE bandwidth
    within Scott's-rule plateau and renders visually-identical violins.
    Per-class population in the label still reflects the FULL group size.
    """
    from ._sampling import subsample_for_density

    K = len(classes)
    groups: List[np.ndarray] = []
    labels: List[str] = []
    for k in range(K):
        mask = np.asarray(y_true) == k
        if not mask.any():
            groups.append(np.array([0.0]))   # placeholder so the violin slot exists
            labels.append(f"{classes[k]} (n=0)")
        else:
            full = y_proba[mask, k]
            # Cap KDE-bound rendering cost. Label reflects true group size.
            groups.append(subsample_for_density(full, seed=k))
            labels.append(f"{classes[k]} (n={int(mask.sum())})")
    return ViolinPanelSpec(
        groups=tuple(groups),
        group_labels=tuple(labels),
        title="P(y=true_class) per true class",
        xlabel="True class",
        ylabel="Predicted P(y = true_class)",
    )


def _top_k_acc_panel(y_true, y_proba, classes) -> LinePanelSpec:
    """Top-k accuracy curve: probability that the true class is in
    the top-k predicted classes (by score), for k=1..K."""
    K = len(classes)
    y_arr = np.asarray(y_true)
    n = len(y_arr)
    if n == 0:
        x = np.arange(1, K + 1)
        return LinePanelSpec(x=x, y=np.zeros(K), title="Top-k accuracy",
                             xlabel="k", ylabel="Top-k accuracy")
    # For each row, rank classes by descending probability.
    sorted_idx = np.argsort(-y_proba, axis=1)
    accs = np.zeros(K)
    for k in range(1, K + 1):
        in_top_k = (sorted_idx[:, :k] == y_arr[:, None]).any(axis=1)
        accs[k - 1] = float(in_top_k.mean())
    return LinePanelSpec(
        x=np.arange(1, K + 1),
        y=accs,
        title="Top-k accuracy",
        xlabel="k",
        ylabel="Top-k accuracy",
    )


# ----------------------------------------------------------------------------
# Token registry + composer
# ----------------------------------------------------------------------------


_TOKEN_BUILDERS: Dict[str, Callable] = {
    "CONFUSION": _confusion_panel,
    "PR_F1": _pr_f1_panel,
    "ROC": _roc_panel,
    "PR_CURVES": _pr_curves_panel,
    "CALIB_GRID": _calib_grid_panel,
    "PROB_DIST": _prob_dist_panel,
    "TOP_K_ACC": _top_k_acc_panel,
}

ALLOWED_MULTICLASS_PANEL_TOKENS = frozenset(_TOKEN_BUILDERS)


def compose_multiclass_figure(
    y_true,
    y_proba: np.ndarray,
    classes: Sequence,
    *,
    panels_template: str = "CONFUSION PR_F1 ROC CALIB_GRID PROB_DIST TOP_K_ACC",
    suptitle: str = "",
    max_cols: int = 2,
    cell_width: float = 6.0,
    cell_height: float = 4.0,
) -> FigureSpec:
    """Build a multiclass quality FigureSpec from a panel template.

    Parameters
    ----------
    y_true : (N,) array of int class labels
    y_proba : (N, K) array of predicted class probabilities (rows sum to 1)
    classes : sequence of K class identifiers (e.g. [0, 1, 2] or strings)
    panels_template : space-separated token list. See the module docstring
        for the full vocabulary.
    suptitle : figure suptitle (model identity).
    max_cols : grid width (default 2).
    """
    tokens = parse_panel_template(panels_template)
    unknown = [t for t in tokens if t not in _TOKEN_BUILDERS]
    if unknown:
        raise ValueError(
            f"Unknown multiclass panel tokens {unknown}. "
            f"Allowed: {sorted(ALLOWED_MULTICLASS_PANEL_TOKENS)}"
        )
    # ``y_true`` carries the RAW class labels (commonly ``model.classes_``
    # values), which are NOT guaranteed to be the positions 0..K-1 -- the
    # target is never label-encoded upstream. Every panel builder indexes a
    # K-sized structure positionally (``matrix[int(t)]``, ``y_true == k``,
    # ``labels=range(K)``), so a raw label like 5 or "low" would IndexError
    # (the figure is then silently dropped by the dispatcher) or land in the
    # wrong cell. Remap the true labels to their position in ``classes`` once,
    # up front, so every panel is correct; ``classes`` still supplies the
    # display labels. Unseen labels map to -1 (matched by no class -> excluded).
    y_true_arr = np.asarray(y_true)
    if len(classes):
        try:
            # Vectorised label->position remap: 2-4x faster than the per-element
            # dict.get listcomp at n=100k (cost scales with len(y_true), so it
            # surfaces only at large n). argsort + searchsorted maps each raw label
            # to its index in ``classes`` (-1 for unseen), bit-identical to the dict
            # lookup for the homogeneous, orderable, unique class arrays sklearn
            # produces (verified). Falls back to the dict listcomp for pathological
            # unorderable / mixed-dtype ``classes`` (TypeError) so robustness is kept.
            _classes_arr = np.asarray(classes)
            _sorter = np.argsort(_classes_arr, kind="stable")
            _sorted_c = _classes_arr[_sorter]
            _pos = np.clip(np.searchsorted(_sorted_c, y_true_arr), 0, len(_sorted_c) - 1)
            _matched = _sorted_c[_pos] == y_true_arr
            y_true_pos = np.where(_matched, _sorter[_pos], -1).astype(np.int64)
        except (TypeError, ValueError):
            _label_to_pos = {lbl: i for i, lbl in enumerate(classes)}
            y_true_pos = np.array(
                [_label_to_pos.get(t, -1) for t in y_true_arr.tolist()],
                dtype=np.int64,
            )
    else:
        y_true_pos = np.full(y_true_arr.shape, -1, dtype=np.int64)
    # A total mismatch (every label unseen) silently empties every one-vs-rest
    # panel -- the usual cause is y_true holding positional indices 0..K-1 while
    # classes are display strings. Surface it loudly instead of rendering blanks.
    if y_true_pos.size and not (y_true_pos >= 0).any():
        warnings.warn(
            f"compose_multiclass_figure: none of the {y_true_pos.size} y_true "
            f"values matched any entry in classes={list(classes)!r}; every "
            "sample was excluded and the panels will be empty. y_true must hold "
            "the actual class identifiers (the same domain as classes), not "
            "positional indices.",
            UserWarning, stacklevel=2,
        )
    panels: List[PanelSpec] = [
        _TOKEN_BUILDERS[tok](y_true_pos, y_proba, classes) for tok in tokens
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
    "ALLOWED_MULTICLASS_PANEL_TOKENS",
    "compose_multiclass_figure",
]
