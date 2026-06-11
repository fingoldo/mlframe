"""Multiclass quality-visualisation panels.

Each panel builder takes ``(y_true, y_pred_proba, classes)`` and returns
a single ``PanelSpec`` instance. ``compose_multiclass_figure`` parses
the panel template (DSL from ``ReportingConfig.multiclass_panels``) and
packs the selected panels into a row-major grid.

Token catalogue (all 8):
- ``CONFUSION``  — row-normalised confusion matrix heatmap
- ``CONFUSED_PAIRS`` — top-N most-confused (true -> pred) class pairs as a horizontal bar
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
from mlframe.reporting.colors import HEATMAP_CMAP
from mlframe.reporting.spec import (
    AnnotationPanelSpec, BarPanelSpec, FigureSpec, HeatmapPanelSpec,
    LinePanelSpec, PanelSpec, ScatterPanelSpec, ViolinPanelSpec,
)

# Curves drawn on a 200-pt display grid cannot resolve more than this many
# raw points; sklearn roc_curve / precision_recall_curve cost is what we cap.
_CURVE_SUBSAMPLE_CAP: int = 200_000
# Above this K the confusion heatmap K^2 cell-text turns to unreadable soup.
_CONFUSION_TEXT_MAX_K: int = 15

# matplotlib tab20: extends the 10-color LINE_PALETTE so two classes never share
# a color until K > 20 (per-class ROC / PR / calib overlays go well past 10).
_TAB20: Tuple[str, ...] = (
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
    "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
    "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
    "#17becf", "#9edae5",
)


def _class_color(idx: int) -> str:
    """Per-class line color; uses tab20 before recycling so up to 20 classes are distinct."""
    return _TAB20[idx % len(_TAB20)]


def _stratified_subsample(y_pos: np.ndarray, cap: int, seed: int = 0) -> np.ndarray:
    """Indices of a class-stratified subsample of size ~``cap`` (all rows if n <= cap).

    Proportional allocation per class keeps each one-vs-rest curve's positive/negative
    ratio intact, so a subsampled roc_curve / precision_recall_curve matches the full-n
    shape that a 200-pt display grid can resolve. Deterministic via a fixed-seed RNG.
    """
    n = y_pos.shape[0]
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    frac = cap / n
    out: List[np.ndarray] = []
    for c in np.unique(y_pos):
        idx_c = np.flatnonzero(y_pos == c)
        take = max(1, int(round(len(idx_c) * frac)))
        take = min(take, len(idx_c))
        out.append(rng.choice(idx_c, size=take, replace=False))
    return np.sort(np.concatenate(out)).astype(np.int64)


# ----------------------------------------------------------------------------
# Per-token panel builders
# ----------------------------------------------------------------------------


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


def _confused_pairs_panel(y_true, y_proba, classes, *, y_pred=None, top_n: int = 15) -> PanelSpec:
    """Top-N most-confused (true -> pred) class pairs as a horizontal bar.

    Ranks off-diagonal cells of the ROW-NORMALISED confusion matrix (so a 40%
    misroute of a rare class outranks a 2% leak of a frequent one). Bars read
    "A -> B: x%" with the highest-confusion pair on top.
    """
    K = len(classes)
    matrix = _confusion_counts(y_true, _resolve_pred(y_pred, y_proba), K)
    row_sums = matrix.sum(axis=1, keepdims=True)
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
        pairs.append(f"{classes[i]} -> {classes[j]}")
        vals.append(v)
        if len(pairs) >= top_n:
            break
    if not pairs:
        return AnnotationPanelSpec(
            text="No off-diagonal confusion\n(perfect or single-class predictions)",
            title="Most-confused class pairs",
        )
    # Horizontal bars: long "A -> B" labels read cleanly on the y-axis and the highest-confusion pair sits on top.
    categories = tuple(pairs)
    values = np.asarray(vals, dtype=np.float64)
    return BarPanelSpec(
        categories=categories,
        values=values,
        title=f"Most-confused class pairs (top {len(pairs)})",
        xlabel="P(pred | true)",
        ylabel="true -> pred",
        orientation="horizontal",
    )


def _pr_f1_panel(y_true, y_proba, classes, *, y_pred=None) -> BarPanelSpec:
    """Per-class precision / recall / F1 grouped bar."""
    from sklearn.metrics import precision_recall_fscore_support

    K = len(classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, _resolve_pred(y_pred, y_proba), labels=list(range(K)), average=None, zero_division=0,
    )
    return BarPanelSpec(
        categories=tuple(str(c) for c in classes),
        values=(np.asarray(precision), np.asarray(recall), np.asarray(f1)),
        series_labels=("precision", "recall", "F1"),
        title="Per-class P / R / F1",
        xlabel="Class",
        ylabel="Score",
    )


def _roc_panel(y_true, y_proba, classes, *, y_pred=None, sub=None, show_auc_ci: bool = True) -> LinePanelSpec:
    """Per-class ROC curves overlaid (one-vs-rest).

    Curve vertices AND the legend AUC come from one class-stratified subsample (cap
    ``_CURVE_SUBSAMPLE_CAP``). A full-n ``roc_auc_score`` argsorts the whole column per
    class (~5s @2M/K=10) for a number the 200-pt display can't distinguish from the
    stratified estimate, so AUC is taken via ``auc(fpr, tpr)`` on the same subsample.
    ``sub`` may be a precomputed shared subsample index (composer passes one for all panels).

    ``show_auc_ci`` (default on) appends a 95% DeLong confidence interval to each class's
    AUC legend label. DeLong is the closed-form O(n log n) AUC-variance estimator -- no
    bootstrap -- so the CI is essentially free on top of the AUC the panel already computes
    (it reads the same stratified subsample). A wide bracket on a rare class flags that its
    AUC is not yet pinned down by the data, which a single point estimate hides.
    """
    from sklearn.metrics import auc, roc_curve
    from mlframe.reporting.charts.calibration import delong_auc_ci

    K = len(classes)
    labels: List[str] = []
    x_grid = np.linspace(0.0, 1.0, 200)
    interpolated: List[np.ndarray] = []
    yt = np.asarray(y_true)
    # One shared class-stratified subsample for all K curves -- recomputing it per class
    # (full-n unique/flatnonzero x K) dominated the panel; stratifying on the true class
    # keeps every one-vs-rest ratio intact.
    if sub is None:
        sub = _stratified_subsample(yt, _CURVE_SUBSAMPLE_CAP, seed=0)
    yt_s = yt[sub]
    proba_s = y_proba[sub]
    for k in range(K):
        bin_y = (yt_s == k).astype(np.int8)
        if bin_y.sum() == 0 or bin_y.sum() == len(bin_y):
            interpolated.append(np.full_like(x_grid, np.nan))
            labels.append(f"{classes[k]} (n/a)")
            continue
        fpr, tpr, _ = roc_curve(bin_y, proba_s[:, k])
        roc_auc = auc(fpr, tpr)
        interpolated.append(np.interp(x_grid, fpr, tpr))
        if show_auc_ci:
            # DeLong CI from the same stratified subsample (the displayed AUC's data); closed-form, no extra sort cost
            # beyond two midrank argsorts the panel would not otherwise pay.
            _, lo, hi = delong_auc_ci(bin_y, proba_s[:, k])
            labels.append(f"{classes[k]} (AUC={roc_auc:.3f} [{lo:.3f}, {hi:.3f}])")
        else:
            labels.append(f"{classes[k]} (AUC={roc_auc:.3f})")
    chance = x_grid.copy()
    return LinePanelSpec(
        x=x_grid,
        y=tuple([chance] + interpolated),
        series_labels=tuple(["chance"] + labels),
        title="Per-class ROC (one-vs-rest)",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        line_styles=tuple([":"] + ["-"] * K),
        colors=tuple(["gray"] + [_class_color(i) for i in range(K)]),
    )


def _pr_curves_panel(y_true, y_proba, classes, *, y_pred=None, sub=None) -> LinePanelSpec:
    """Per-class precision-recall curves overlaid (one-vs-rest).

    Curve vertices AND the legend AP come from one class-stratified subsample (the full-n
    average_precision_score argsorts every column per class for a number the 200-pt grid
    can't resolve). Each class gets a dotted no-skill (prevalence) baseline at P =
    positive-rate, so a curve hugging its own prevalence line reads as no better than
    always-positive.
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    K = len(classes)
    x_grid = np.linspace(0.0, 1.0, 200)
    interpolated: List[np.ndarray] = []
    baselines: List[np.ndarray] = []
    labels: List[str] = []
    baseline_labels: List[str] = []
    yt = np.asarray(y_true)
    n_valid = int((yt >= 0).sum())
    # Shared class-stratified subsample for all K curves (see _roc_panel).
    if sub is None:
        sub = _stratified_subsample(yt, _CURVE_SUBSAMPLE_CAP, seed=0)
    yt_s = yt[sub]
    proba_s = y_proba[sub]
    for k in range(K):
        bin_full = int((yt == k).sum())                              # full-n prevalence numerator
        bin_y = (yt_s == k).astype(np.int8)
        if bin_full == 0:
            interpolated.append(np.full_like(x_grid, np.nan))
            labels.append(f"{classes[k]} (n/a)")
            baselines.append(np.full_like(x_grid, np.nan))
            baseline_labels.append("")
            continue
        ap = average_precision_score(bin_y, proba_s[:, k])           # stratified-subsample AP
        precision, recall, _ = precision_recall_curve(bin_y, proba_s[:, k])
        order = np.argsort(recall)
        interpolated.append(np.interp(x_grid, recall[order], precision[order]))
        labels.append(f"{classes[k]} (AP={ap:.3f})")
        prevalence = float(bin_full) / max(1, n_valid)               # no-skill precision (full-n)
        baselines.append(np.full_like(x_grid, prevalence))
        baseline_labels.append("")
    return LinePanelSpec(
        x=x_grid,
        y=tuple(interpolated + baselines),
        series_labels=tuple(labels + baseline_labels),
        title="Per-class precision-recall",
        xlabel="Recall",
        ylabel="Precision",
        line_styles=tuple(["-"] * K + [":"] * K),
        colors=tuple([_class_color(i) for i in range(K)] + [_class_color(i) for i in range(K)]),
    )


def _calib_grid_panel(y_true, y_proba, classes, *, y_pred=None, sub=None) -> LinePanelSpec:
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
    # Reliability curves are per-bin means, so a shared class-stratified subsample
    # estimates each bin within display precision and bounds the per-class O(N) digitize.
    if sub is None:
        sub = _stratified_subsample(yt, _CURVE_SUBSAMPLE_CAP, seed=0)
    yt_s = yt[sub]
    proba_s = y_proba[sub]
    for k in range(K):
        proba_k = proba_s[:, k]
        true_k = (yt_s == k).astype(np.float64)
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
        colors=tuple(["green"] + [_class_color(i) for i in range(K)]),
    )


def _prob_dist_panel(y_true, y_proba, classes, *, y_pred=None, sub=None) -> PanelSpec:
    """Per-true-class violin: distribution of P(y=true_class | x).

    Concentration near 1 = high confidence; spread across [0,1] = calibrated
    uncertainty. Always-near-0 violin = model collapse on that class.

    Empty classes are dropped (no fake ``[0.0]`` violin); if no class has any
    sample the panel becomes an honest annotation placeholder.

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
    yt = np.asarray(y_true)
    # Per-class full-n counts (for the label) via one bincount; the violin data is
    # then drawn from a shared stratified subsample so the per-class mask is over
    # ~200k rows instead of full n (the violins KDE-cap at 5000 anyway).
    valid = yt >= 0
    full_counts = np.bincount(yt[valid], minlength=K)[:K] if valid.any() else np.zeros(K, dtype=np.int64)
    if sub is None:
        sub = _stratified_subsample(yt, _CURVE_SUBSAMPLE_CAP, seed=0)
    yt_s = yt[sub]
    proba_s = y_proba[sub]
    for k in range(K):
        if full_counts[k] == 0:
            continue   # drop empty class rather than planting a fake [0.0] violin
        mask = yt_s == k
        groups.append(subsample_for_density(proba_s[mask, k], seed=k))
        labels.append(f"{classes[k]} (n={int(full_counts[k])})")
    if not groups:
        return AnnotationPanelSpec(
            text="P(y=true_class): no true-class samples\n(every y_true was excluded)",
            title="P(y=true_class) per true class",
        )
    return ViolinPanelSpec(
        groups=tuple(groups),
        group_labels=tuple(labels),
        title="P(y=true_class) per true class",
        xlabel="True class",
        ylabel="Predicted P(y = true_class)",
    )


def _top_k_acc_panel(y_true, y_proba, classes, *, y_pred=None) -> LinePanelSpec:
    """Top-k accuracy curve: probability that the true class is in
    the top-k predicted classes (by score), for k=1..K.

    The per-row argsort over the full (N, K) proba dominates at large N; top-k accuracy
    is a row-mean so a uniform subsample of ~``_CURVE_SUBSAMPLE_CAP`` rows estimates each
    point within display precision. Rows with no matched true class (-1) are dropped.
    """
    K = len(classes)
    y_arr = np.asarray(y_true)
    valid = y_arr >= 0
    y_arr = y_arr[valid]
    proba = y_proba[valid]
    n = len(y_arr)
    if n == 0:
        x = np.arange(1, K + 1)
        return LinePanelSpec(x=x, y=np.zeros(K), title="Top-k accuracy",
                             xlabel="k", ylabel="Top-k accuracy")
    if n > _CURVE_SUBSAMPLE_CAP:
        rng = np.random.default_rng(0)
        sub = rng.choice(n, size=_CURVE_SUBSAMPLE_CAP, replace=False)
        y_arr = y_arr[sub]
        proba = proba[sub]
    # For each row, rank classes by descending probability.
    sorted_idx = np.argsort(-proba, axis=1)
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
    "CONFUSED_PAIRS": _confused_pairs_panel,
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
            # Unorderable / mixed-dtype ``classes``: resolve each DISTINCT label once
            # (pd.factorize collapses to ~K uniques + a hash-based inverse index, no sort)
            # instead of a per-row dict.get listcomp over the full n. pd.factorize is the
            # right tool here because np.unique would re-trip the same unorderable comparison.
            import pandas as pd
            _label_to_pos = {lbl: i for i, lbl in enumerate(classes)}
            _codes, _uniq = pd.factorize(np.asarray(y_true_arr).ravel(), sort=False)
            _uniq_pos = np.array([_label_to_pos.get(u, -1) for u in list(_uniq)], dtype=np.int64)
            # pd.factorize codes NaN/missing as -1; map those to -1 (excluded) too.
            y_true_pos = np.where(_codes >= 0, _uniq_pos[_codes], -1).reshape(y_true_arr.shape)
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
    # Hard prediction (positional argmax) computed ONCE here and threaded into every
    # builder; previously each of CONFUSION / PR_F1 recomputed it (duplicate full-n argmax).
    from ...utils.nan_safe import argmax_classes_safe
    y_pred_pos = argmax_classes_safe(np.asarray(y_proba), context="reporting.charts.multiclass")
    # One shared class-stratified subsample index for every curve / violin / reliability
    # panel; the per-panel default would recompute the full-n unique once per token.
    shared_sub = _stratified_subsample(np.asarray(y_true_pos), _CURVE_SUBSAMPLE_CAP, seed=0)
    _SUB_TOKENS = {"ROC", "PR_CURVES", "CALIB_GRID", "PROB_DIST"}
    panels: List[PanelSpec] = []
    for tok in tokens:
        kw = {"y_pred": y_pred_pos}
        if tok in _SUB_TOKENS:
            kw["sub"] = shared_sub
        panels.append(_TOKEN_BUILDERS[tok](y_true_pos, y_proba, classes, **kw))
    grid = pack_panels(panels, max_cols=max_cols)
    n_rows = len(grid)
    n_cols = max_cols if grid else 0
    # Scale cell width with K so K-class confusion / per-class legends stay legible;
    # the fixed 6.0 squeezes labels past ~6 classes. Capped so very large K stays bounded.
    K = len(classes)
    eff_cell_width = max(cell_width, min(12.0, cell_width + 0.5 * max(0, K - 6)))
    return FigureSpec(
        suptitle=suptitle,
        panels=grid,
        figsize=figsize_for_grid(n_rows, n_cols,
                                 cell_width=eff_cell_width, cell_height=cell_height),
    )


__all__ = [
    "ALLOWED_MULTICLASS_PANEL_TOKENS",
    "compose_multiclass_figure",
]
