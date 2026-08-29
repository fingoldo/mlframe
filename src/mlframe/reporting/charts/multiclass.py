"""Multiclass quality-visualisation panels.

Each panel builder takes ``(y_true, y_pred_proba, classes)`` and returns
a single ``PanelSpec`` instance. ``compose_multiclass_figure`` parses
the panel template (DSL from ``ReportingConfig.multiclass_panels``) and
packs the selected panels into a row-major grid.

Token catalogue (all 9):
- ``CONFUSION``  — row-normalised confusion matrix heatmap
- ``CONFUSION_MARGINS`` — confusion heatmap flanked by per-true-class support (right bar) and
                   per-predicted-class volume (top bar), so imbalance + over/under-prediction read at a glance
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
from typing import Callable, Dict, List, Sequence

import numpy as np

from ._captions import caption_for_tokens

from mlframe.reporting.charts._layout import (
    figsize_for_grid, pack_panels, parse_panel_template,
)
from mlframe.reporting.charts._rank_stats import ovr_rank_auc
from mlframe.reporting.colors import line_color, line_style
from mlframe.reporting.spec import (
    AnnotationPanelSpec, BarPanelSpec, FigureSpec,
    LinePanelSpec, PanelSpec, ViolinPanelSpec,
)

# Curves drawn on a 200-pt display grid cannot resolve more than this many
# raw points; sklearn roc_curve / precision_recall_curve cost is what we cap.
_CURVE_SUBSAMPLE_CAP: int = 200_000

from ._multiclass_confusion import (
    _confused_pairs_panel,
    _confusion_counts,  # noqa: F401 -- re-export: tests/reporting/test_charts_confusion_margins.py imports it from here
    _confusion_margins_panel,
    _confusion_panel,
    _resolve_pred,
)

def _class_color(idx: int) -> str:
    """Per-class line colour from the shared palette (paired with ``line_style``'s per-wrap dash past 10 classes)."""
    return line_color(idx)


# Above this K the per-class ROC/PR/reliability overlays put K curves in one panel: slow to compute (K sklearn fits)
# and unreadable spaghetti. Past it the composer renders only the OVERLAY_TOP_N worst-by-AUC classes + a macro-average.
_OVERLAY_MAX_CLASSES: int = 12
_OVERLAY_TOP_N: int = 8


def _ova_auc_all_classes(yt_pos: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """One-vs-rest AUC for every class at once via the rank (Mann-Whitney) identity -- one argsort per column.

    Selecting the worst-N classes needs an AUC for all K, but K separate sklearn ``roc_auc_score`` calls are the very
    cost the top-N switch exists to avoid. The rank-sum AUC over the shared subsample is a single vectorised pass
    that matches sklearn within display precision; classes with no positives/negatives return NaN (sorted last as
    "best").
    """
    return ovr_rank_auc(np.asarray(yt_pos), proba, int(proba.shape[1]))


def _select_overlay_classes(yt_pos: np.ndarray, proba: np.ndarray, top_n: int) -> np.ndarray:
    """Indices of the ``top_n`` WORST classes by one-vs-rest AUC (lowest AUC = hardest = most worth showing).

    NaN AUC (degenerate one-vs-rest split) sorts as "best" so a class with no defined curve never displaces a class
    with a genuine low AUC. Returned ascending in class index so the legend reads in a stable order.
    """
    auc = _ova_auc_all_classes(yt_pos, proba)
    order = np.argsort(np.where(np.isnan(auc), np.inf, auc), kind="stable")
    chosen = order[: min(top_n, len(order))]
    return np.sort(chosen)


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
        take = max(1, round(len(idx_c) * frac))
        take = min(take, len(idx_c))
        out.append(rng.choice(idx_c, size=take, replace=False))
    return np.sort(np.concatenate(out)).astype(np.int64)


def _pr_f1_panel(y_true, y_proba, classes, *, y_pred=None) -> BarPanelSpec:
    """Per-class precision / recall / F1 grouped bar."""
    from sklearn.metrics import precision_recall_fscore_support

    K = len(classes)
    yt = np.asarray(y_true)
    yp = _resolve_pred(y_pred, y_proba)
    # sklearn raises "Found empty input array" on n==0; an all-zeros P/R/F1 bar is the honest empty-data reading.
    if yt.size == 0:
        precision = recall = f1 = np.zeros(K, dtype=np.float64)
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            yt, yp, labels=list(range(K)), average=None, zero_division=0,
        )
    return BarPanelSpec(
        categories=tuple(str(c) for c in classes),
        values=(np.asarray(precision), np.asarray(recall), np.asarray(f1)),
        series_labels=("precision", "recall", "F1"),
        title=f"Per-class P / R / F1 (macro-F1={float(np.mean(f1)):.3f} over {K} classes)",
        xlabel="Class",
        ylabel="Score",
    )


def _roc_panel(y_true, y_proba, classes, *, y_pred=None, sub=None, show_auc_ci: bool = True, class_subset=None) -> LinePanelSpec:
    """Per-class ROC curves overlaid (one-vs-rest).

    Curve vertices AND the legend AUC come from one class-stratified subsample (cap
    ``_CURVE_SUBSAMPLE_CAP``). A full-n ``roc_auc_score`` argsorts the whole column per
    class (~5s @2M/K=10) for a number the 200-pt display can't distinguish from the
    stratified estimate, so AUC is taken via ``auc(fpr, tpr)`` on the same subsample.
    ``sub`` may be a precomputed shared subsample index (composer passes one for all panels).

    ``class_subset`` (composer passes one past the large-K threshold) restricts the drawn curves to those class
    indices and appends a macro-average curve over the SAME subset, so K=100 shows ~8 readable curves + macro instead
    of 100 overlapping ones -- a speed win (8 sklearn fits, not 100) and a readability win.

    ``show_auc_ci`` (default on) appends a 95% DeLong confidence interval to each class's
    AUC legend label. DeLong is the closed-form O(n log n) AUC-variance estimator -- no
    bootstrap -- so the CI is essentially free on top of the AUC the panel already computes
    (it reads the same stratified subsample). A wide bracket on a rare class flags that its
    AUC is not yet pinned down by the data, which a single point estimate hides.
    """
    from sklearn.metrics import auc
    from mlframe.metrics.core import fast_roc_curve
    from mlframe.reporting.charts.calibration import delong_auc_ci

    K = len(classes)
    draw_idx = list(range(K)) if class_subset is None else [int(k) for k in class_subset]
    labels: List[str] = []
    x_grid = np.linspace(0.0, 1.0, 200)
    interpolated: List[np.ndarray] = []
    colors: List[str] = []
    yt = np.asarray(y_true)
    # One shared class-stratified subsample for all curves -- recomputing it per class
    # (full-n unique/flatnonzero x K) dominated the panel; stratifying on the true class
    # keeps every one-vs-rest ratio intact.
    if sub is None:
        sub = _stratified_subsample(yt, _CURVE_SUBSAMPLE_CAP, seed=0)
    yt_s = yt[sub]
    proba_s = y_proba[sub]
    valid_curves: List[np.ndarray] = []
    for k in draw_idx:
        bin_y = (yt_s == k).astype(np.int8)
        col = proba_s[:, k]
        # roc_curve rejects non-finite scores; a single class or an all-NaN proba column has no defined curve.
        if bin_y.sum() == 0 or bin_y.sum() == len(bin_y) or not np.isfinite(col).any():
            interpolated.append(np.full_like(x_grid, np.nan))
            labels.append(f"{classes[k]} (n/a)")
            colors.append(_class_color(k))
            continue
        finite = np.isfinite(col)
        bin_y, col = bin_y[finite], col[finite]
        if bin_y.sum() == 0 or bin_y.sum() == len(bin_y):
            interpolated.append(np.full_like(x_grid, np.nan))
            labels.append(f"{classes[k]} (n/a)")
            colors.append(_class_color(k))
            continue
        fpr, tpr, _ = fast_roc_curve(bin_y, col)
        roc_auc = auc(fpr, tpr)
        curve = np.interp(x_grid, fpr, tpr)
        interpolated.append(curve)
        valid_curves.append(curve)
        colors.append(_class_color(k))
        if show_auc_ci:
            # DeLong CI from the same stratified subsample (the displayed AUC's data); closed-form, no extra sort cost
            # beyond two midrank argsorts the panel would not otherwise pay.
            _, lo, hi = delong_auc_ci(bin_y, col)
            labels.append(f"{classes[k]} (AUC={roc_auc:.3f} [{lo:.3f}, {hi:.3f}])")
        else:
            labels.append(f"{classes[k]} (AUC={roc_auc:.3f})")
    chance = x_grid.copy()
    series = [chance, *interpolated]
    series_labels = ["chance", *labels]
    # Past the palette's 10 hues the colour repeats, so the dash pattern is what keeps class 0 and class 10 apart --
    # and it is the only separation that survives a red-green colour vision deficiency at any K.
    styles = [":"] + [line_style(int(k)) for k in draw_idx]
    line_colors = ["gray", *colors]
    if class_subset is not None:
        # Macro-average TPR over the drawn classes anchors the truncated overlay to a population-level summary.
        macro = np.nanmean(np.vstack(valid_curves), axis=0) if valid_curves else np.full_like(x_grid, np.nan)
        macro_auc = float(auc(x_grid, macro)) if valid_curves else float("nan")
        series.append(macro)
        series_labels.append(f"macro-avg (AUC={macro_auc:.3f})")
        styles.append("--")
        line_colors.append("black")
    title = "Per-class ROC (one-vs-rest)"
    if class_subset is not None:
        title = f"Per-class ROC: {len(draw_idx)} of {K} classes (worst by AUC) + macro-avg"
    return LinePanelSpec(
        x=x_grid,
        y=tuple(series),
        series_labels=tuple(series_labels),
        title=title,
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        line_styles=tuple(styles),
        colors=tuple(line_colors),
    )


def _pr_curves_panel(y_true, y_proba, classes, *, y_pred=None, sub=None, class_subset=None) -> LinePanelSpec:
    """Per-class precision-recall curves overlaid (one-vs-rest).

    Curve vertices AND the legend AP come from one class-stratified subsample (the full-n
    average_precision_score argsorts every column per class for a number the 200-pt grid
    can't resolve). Each class gets a dotted no-skill (prevalence) baseline at P =
    positive-rate, so a curve hugging its own prevalence line reads as no better than
    always-positive.

    ``class_subset`` (composer passes one past the large-K threshold) restricts the drawn curves to those class
    indices and appends a macro-average PR curve, so large K shows ~8 readable curves + macro instead of K.
    """
    from sklearn.metrics import precision_recall_curve
    from mlframe.metrics.core import average_precision_score

    K = len(classes)
    draw_idx = list(range(K)) if class_subset is None else [int(k) for k in class_subset]
    x_grid = np.linspace(0.0, 1.0, 200)
    interpolated: List[np.ndarray] = []
    baselines: List[np.ndarray] = []
    labels: List[str] = []
    baseline_labels: List[str] = []
    draw_colors: List[str] = []
    valid_curves: List[np.ndarray] = []
    yt = np.asarray(y_true)
    # Shared class-stratified subsample for all K curves (see _roc_panel).
    if sub is None:
        sub = _stratified_subsample(yt, _CURVE_SUBSAMPLE_CAP, seed=0)
    yt_s = yt[sub]
    proba_s = y_proba[sub]
    for k in draw_idx:
        bin_full = int((yt == k).sum())  # full-n prevalence numerator
        bin_y = (yt_s == k).astype(np.int8)
        col = proba_s[:, k]
        finite = np.isfinite(col)
        draw_colors.append(_class_color(k))
        # No positives, or an all-NaN proba column -> no defined PR curve (sklearn rejects non-finite scores).
        if bin_full == 0 or not finite.any() or int(bin_y[finite].sum()) == 0:
            interpolated.append(np.full_like(x_grid, np.nan))
            labels.append(f"{classes[k]} (n/a)")
            baselines.append(np.full_like(x_grid, np.nan))
            baseline_labels.append("")
            continue  # no curve, so no baseline to label either
        bin_yf, colf = bin_y[finite], col[finite]
        ap = average_precision_score(bin_yf, colf)  # stratified-subsample AP
        precision, recall, _ = precision_recall_curve(bin_yf, colf)
        order = np.argsort(recall)
        curve = np.interp(x_grid, recall[order], precision[order])
        interpolated.append(curve)
        valid_curves.append(curve)
        labels.append(f"{classes[k]} (AP={ap:.3f})")
        # No-skill precision baseline must use the SAME population the AP was computed on (the finite stratified
        # subsample), otherwise AP-vs-baseline compares inconsistent prevalences.
        prevalence = float(int(bin_yf.sum())) / max(1, bin_yf.size)
        baselines.append(np.full_like(x_grid, prevalence))
        # An empty label put K unexplained dotted lines on the panel. Each one is the no-skill precision for ONE
        # class -- its prevalence -- so it has to name the class and the number it draws.
        baseline_labels.append(f"{classes[k]} no-skill ({prevalence:.3f})")
    n_drawn = len(draw_idx)
    series = list(interpolated)
    series_labels = list(labels)
    styles = ["-"] * n_drawn
    line_colors = list(draw_colors)
    if class_subset is not None:
        macro = np.nanmean(np.vstack(valid_curves), axis=0) if valid_curves else np.full_like(x_grid, np.nan)
        series.append(macro)
        series_labels.append("macro-avg")
        styles.append("--")
        line_colors.append("black")
    series += baselines
    series_labels += baseline_labels
    styles += [":"] * n_drawn
    line_colors += list(draw_colors)
    title = "Per-class precision-recall"
    if class_subset is not None:
        title = f"Per-class PR: {n_drawn} of {K} classes (worst by AUC) + macro-avg"
    return LinePanelSpec(
        x=x_grid,
        y=tuple(series),
        series_labels=tuple(series_labels),
        title=title,
        xlabel="Recall",
        ylabel="Precision",
        line_styles=tuple(styles),
        colors=tuple(line_colors),
    )


def _calib_grid_panel(y_true, y_proba, classes, *, y_pred=None, sub=None, class_subset=None) -> LinePanelSpec:
    """Per-class reliability curves overlaid (small-multiples-as-overlay).

    For each class k, bin predictions into deciles, plot mean predicted
    P(y=k) vs observed P(y=k|bin). Perfect calibration = y=x diagonal.

    ``class_subset`` (composer passes one past the large-K threshold) restricts the drawn curves to those class
    indices and appends a macro-average reliability curve, so large K stays readable.
    """
    K = len(classes)
    draw_idx = list(range(K)) if class_subset is None else [int(k) for k in class_subset]
    n_bins = 10
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    x_grid = (edges[:-1] + edges[1:]) / 2

    series: List[np.ndarray] = []
    labels: List[str] = []
    colors: List[str] = []
    yt = np.asarray(y_true)
    # Reliability curves are per-bin means, so a shared class-stratified subsample
    # estimates each bin within display precision and bounds the per-class O(N) digitize.
    if sub is None:
        sub = _stratified_subsample(yt, _CURVE_SUBSAMPLE_CAP, seed=0)
    yt_s = yt[sub]
    proba_s = y_proba[sub]
    pooled_sums = np.zeros(n_bins, dtype=np.float64)
    pooled_counts = np.zeros(n_bins, dtype=np.float64)
    for k in draw_idx:
        proba_k = proba_s[:, k]
        true_k = (yt_s == k).astype(np.float64)
        # np.digitize sorts NaN after every finite edge, so a non-finite proba value would silently
        # land in the LAST bin instead of being excluded from the reliability curve.
        finite_k = np.isfinite(proba_k)
        if not np.all(finite_k):
            proba_k = proba_k[finite_k]
            true_k = true_k[finite_k]
        bin_idx = np.clip(np.digitize(proba_k, edges[1:-1]), 0, n_bins - 1)
        # Per-bin observed mean via two bincounts (sum / count) instead of an
        # inner n_bins x O(N) mask loop.
        counts = np.bincount(bin_idx, minlength=n_bins)
        sums = np.bincount(bin_idx, weights=true_k, minlength=n_bins)
        observed = np.full(n_bins, np.nan)
        nz = counts > 0
        observed[nz] = sums[nz] / counts[nz]
        series.append(observed)
        pooled_sums += sums
        pooled_counts += counts
        labels.append(str(classes[k]))
        colors.append(_class_color(k))
    diag = x_grid.copy()
    all_series = [diag, *series]
    all_labels = ["perfect", *labels]
    styles = [":"] + ["-"] * len(series)
    all_colors = ["green", *colors]
    if class_subset is not None:
        # Pool the per-bin sums and counts rather than averaging the per-class RATES: an unweighted nanmean gave a
        # 3-row bin the same say as a 300k-row bin, and the counts were computed two lines above and thrown away.
        macro = np.full(n_bins, np.nan)
        nz_pool = pooled_counts > 0
        macro[nz_pool] = pooled_sums[nz_pool] / pooled_counts[nz_pool]
        all_series.append(macro)
        all_labels.append("pooled avg (weighted by bin support)")
        styles.append("--")
        all_colors.append("black")
    title = "Per-class reliability curves"
    if class_subset is not None:
        title = f"Per-class reliability: {len(draw_idx)} of {K} classes (worst by AUC) + macro-avg"
    return LinePanelSpec(
        x=x_grid,
        y=tuple(all_series),
        series_labels=tuple(all_labels),
        title=title,
        xlabel="Mean predicted P(y=k)",
        ylabel="Observed P(y=k | bin)",
        line_styles=tuple(styles),
        colors=tuple(all_colors),
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
            continue  # drop empty class rather than planting a fake [0.0] violin
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


def _top_k_acc_panel(y_true, y_proba, classes, *, y_pred=None) -> PanelSpec:
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
        # A line of zeros is indistinguishable from a model that gets every row wrong; say WHY instead.
        return AnnotationPanelSpec(
            text="Top-k accuracy unavailable: no row's true label matched any of the supplied class labels.",
            title="Top-k accuracy",
        )
    if n > _CURVE_SUBSAMPLE_CAP:
        rng = np.random.default_rng(0)
        sub = rng.choice(n, size=_CURVE_SUBSAMPLE_CAP, replace=False)
        y_arr = y_arr[sub]
        proba = proba[sub]
    # Rank the true class by how many classes STRICTLY outscore it, and split the tied block fairly instead of
    # letting argsort break ties by class index. A collapsed model emitting a uniform row is pure chance, but under
    # index tie-breaking it scored the lowest-indexed class every time: at prevalences [0.8, 0.1, 0.1] that reported
    # top-1 accuracy 0.798, and reordering the same classes to [0.1, 0.1, 0.8] reported 0.100 -- both should be 1/K.
    # Averaging over tie permutations (the closed form below) is exact and needs no RNG, and the whole loop is now
    # O(n*K) rather than the O(n*K^2) the per-k membership scan cost.
    p_true = np.take_along_axis(proba, y_arr[:, None], axis=1)
    strictly_greater = (proba > p_true).sum(axis=1).astype(np.float64)
    n_tied = np.maximum((proba == p_true).sum(axis=1), 1).astype(np.float64)
    accs = np.empty(K)
    for k in range(1, K + 1):
        accs[k - 1] = float(np.clip((k - strictly_greater) / n_tied, 0.0, 1.0).mean())
    tied_frac = float((n_tied > 1).mean())
    tie_note = f" -- {tied_frac:.0%} of rows tie at the true class (averaged over tie orderings)" if tied_frac else ""
    return LinePanelSpec(
        x=np.arange(1, K + 1),
        y=accs,
        title=f"Top-k accuracy (chance = k/{K}){tie_note}",
        xlabel="k",
        ylabel="Top-k accuracy",
    )


# ----------------------------------------------------------------------------
# Token registry + composer
# ----------------------------------------------------------------------------


_TOKEN_BUILDERS: Dict[str, Callable] = {
    "CONFUSION": _confusion_panel,
    "CONFUSION_MARGINS": _confusion_margins_panel,
    "CONFUSED_PAIRS": _confused_pairs_panel,
    "PR_F1": _pr_f1_panel,
    "ROC": _roc_panel,
    "PR_CURVES": _pr_curves_panel,
    "CALIB_GRID": _calib_grid_panel,
    "PROB_DIST": _prob_dist_panel,
    "TOP_K_ACC": _top_k_acc_panel,
}

ALLOWED_MULTICLASS_PANEL_TOKENS = frozenset(_TOKEN_BUILDERS)

# One sentence per token, joined for the tokens ACTUALLY rendered (see ``_captions.caption_for_tokens``). The
# figure-level caption used to describe the DEFAULT template, so a caller asking for a narrower mix read about
# panels that were not on their figure.
_TOKEN_CAPTIONS: Dict[str, str] = {
    "CONFUSION": ("The confusion matrix reads row-wise: each row is a true class and shows where its rows actually landed."),
    "CONFUSION_MARGINS": (
        "The margins beside the matrix give each true class its support and each predicted class its volume, so a dense-looking row can be told from a well-populated one."
    ),
    "CONFUSED_PAIRS": (
        "The confused-pairs panel ranks the specific class pairs the model mixes up, which is the actionable form of the same information the matrix holds."
    ),
    "ROC": ("The one-vs-rest ROC curves answer ranking per class; a class with few rows produces a jagged curve that is noise, not structure."),
    "PR_CURVES": (
        "The one-vs-rest PR curves each carry their own class prevalence as a baseline, so they can be compared to that line rather than to each other."
    ),
    "PR_F1": ("Per-class precision, recall and F1 turn the matrix into the three numbers a threshold decision actually uses."),
    "PROB_DIST": (
        "The predicted-probability distributions show how confident the model is per class; mass piled at the extremes with a poor calibration curve is over-confidence."
    ),
    "CALIB_GRID": ("The calibration grid asks, per class, whether a predicted 0.7 happens 70% of the time -- a question ranking metrics never answer."),
    "TOP_K_ACC": (
        "Top-k accuracy credits a prediction when the true class is anywhere in the k highest-scoring classes, which is the right metric when the output is a shortlist."
    ),
}


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
    overlay_max_classes: int = _OVERLAY_MAX_CLASSES,
    overlay_top_n: int = _OVERLAY_TOP_N,
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
    overlay_max_classes : above this K the per-class ROC / PR / reliability OVERLAY panels render only the
        ``overlay_top_n`` worst-by-AUC classes plus a macro-average curve instead of all K (both a speed win --
        fewer sklearn fits / artists -- and a readability win, since K=50/100 curves are unreadable spaghetti). At
        K <= this threshold every class still renders (no behaviour change for typical K).
    overlay_top_n : number of worst-by-AUC classes drawn on the overlay panels when K exceeds the threshold.
    """
    if np.asarray(y_true).shape[0] == 0:
        # Four of the six panels draw an empty or zero surface on no rows; only two annotate. One figure-level
        # guard states the cause once rather than leaving four panels to each imply a measurement.
        return FigureSpec(
            suptitle=suptitle,
            panels=((AnnotationPanelSpec(
                text=(f"Multiclass report unavailable: 0 rows were supplied for {len(classes)} classes, so no "
                      "confusion matrix, curve or calibration can be computed."),
                title="Multiclass quality",
            ),),),
            figsize=(cell_width, cell_height),
        )

    tokens = parse_panel_template(panels_template)
    unknown = [t for t in tokens if t not in _TOKEN_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown multiclass panel tokens {unknown}. " f"Allowed: {sorted(ALLOWED_MULTICLASS_PANEL_TOKENS)}")
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
    _OVERLAY_TOKENS = {"ROC", "PR_CURVES", "CALIB_GRID"}
    # Past the threshold the per-class overlay panels would draw K spaghetti curves (K sklearn fits); restrict them
    # to the worst-by-AUC classes + a macro-average. Selection is computed ONCE on the shared subsample and reused.
    K = len(classes)
    class_subset = None
    if K > overlay_max_classes and overlay_top_n < K:
        class_subset = _select_overlay_classes(np.asarray(y_true_pos)[shared_sub], y_proba[shared_sub], overlay_top_n)
    panels: List[PanelSpec] = []
    for tok in tokens:
        kw = {"y_pred": y_pred_pos}
        if tok in _SUB_TOKENS:
            kw["sub"] = shared_sub
        if tok in _OVERLAY_TOKENS and class_subset is not None:
            kw["class_subset"] = class_subset
        panels.append(_TOKEN_BUILDERS[tok](y_true_pos, y_proba, classes, **kw))
    grid = pack_panels(panels, max_cols=max_cols)
    n_rows = len(grid)
    n_cols = max_cols if grid else 0
    # Scale cell width with K so K-class confusion / per-class legends stay legible;
    # the fixed 6.0 squeezes labels past ~6 classes. Capped so very large K stays bounded.
    eff_cell_width = max(cell_width, min(12.0, cell_width + 0.5 * max(0, K - 6)))
    return FigureSpec(
        suptitle=suptitle,
        panels=grid,
        figsize=figsize_for_grid(n_rows, n_cols, cell_width=eff_cell_width, cell_height=cell_height),
        caption=caption_for_tokens(
            "How to read: the panels split into where the errors LAND and how well the scores rank and calibrate per class.",
            tokens,
            _TOKEN_CAPTIONS,
        ),
    )


__all__ = [
    "ALLOWED_MULTICLASS_PANEL_TOKENS",
    "compose_multiclass_figure",
]
