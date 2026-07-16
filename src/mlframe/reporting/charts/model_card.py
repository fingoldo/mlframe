"""One-glance per-model MODEL CARD: an executive-summary FigureSpec for fast triage.

``compose_model_card_figure`` builds a compact card for one (model, split): a header text block with the
task-appropriate headline metrics rendered as a clean key/value list, a one-line TRAFFIC-LIGHT verdict
(GREEN / AMBER / RED) derived from simple thresholds with a stated reason, and up to 3 mini sparkline
panels (decimated/subsampled) giving instant visual context. It is the panel a data scientist reads first
before opening the dense full diagnostic figures.

Classification headline: ROC_AUC, PR_AUC (average precision), ECE, Brier, KS, MCC; minis = ROC + score
distribution + cumulative gain. Regression headline: RMSE, MAE, R2, bias (mean residual), heteroscedasticity
flag; minis = residual-vs-pred + residual histogram + pred-vs-actual.

Metrics reuse existing kernels (binary ``_ScoreSort`` for ROC/PR/KS/gain, ``delong_auc_ci``,
``fast_brier_score_loss``, ``compute_ece_and_brier_decomposition``); the minis reuse the same shared sort so
the card adds no full recompute on top of metric extraction. Degenerate input degrades to an honest
AnnotationPanelSpec (single-class / no-finite-pairs / constant residuals) rather than a fake chart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from mlframe.reporting.charts._layout import figsize_for_grid
from mlframe.reporting.charts.binary import _ScoreSort, _decimate, _finite_binary
from mlframe.reporting.charts.regression import _finite_pair
from mlframe.reporting.spec import (
    AnnotationPanelSpec, BarPanelSpec, FigureSpec, HistogramPanelSpec, LinePanelSpec,
    PanelSpec, ScatterPanelSpec,
)

# Traffic-light thresholds. AUC >= 0.8 green / 0.65-0.8 amber / < 0.65 red is the common discrimination
# rule of thumb; ECE / |bias| add a calibration-side downgrade so a sharp-but-miscalibrated model is not
# waved through as green. Regression mirrors it on R2 (>=0.75 green / 0.4-0.75 amber / <0.4 red).
AUC_GREEN: float = 0.80
AUC_AMBER: float = 0.65
ECE_AMBER: float = 0.05
ECE_RED: float = 0.15
R2_GREEN: float = 0.75
R2_AMBER: float = 0.40
# |mean residual| relative to the target std above which regression bias downgrades the verdict.
BIAS_REL_AMBER: float = 0.10

# Mini sparklines are context, not full panels: a coarse vertex cap keeps them light.
_MINI_VERTEX_CAP: int = 400
_MINI_SCATTER_SAMPLE: int = 1_500
_MINI_HIST_BINS: int = 30
_ECE_NBINS: int = 15

_GREEN = "#2ca02c"
_AMBER = "#ff7f0e"
_RED = "#d62728"


@dataclass(frozen=True)
class ModelCardVerdict:
    """The traffic-light verdict + the reason that drove it (surfaced to callers/tests, not just drawn)."""

    color: str  # "green" / "amber" / "red"
    label: str  # short headline, e.g. "STRONG" / "USABLE" / "WEAK"
    reason: str  # human-readable justification
    headline: Dict[str, float]


def _mcc(tp: float, tn: float, fp: float, fn: float) -> float:
    """Matthews correlation coefficient from a 2x2 confusion count; 0.0 when a denominator factor is empty."""
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom <= 0.0:
        return 0.0
    return float((tp * tn - fp * fn) / np.sqrt(denom))


def _classification_metrics(sort: _ScoreSort, yt: np.ndarray, ys: np.ndarray, threshold: float) -> Dict[str, float]:
    """Headline classification metrics from the shared score sort (one O(n log n) pass already paid)."""
    from mlframe.metrics import fast_brier_score_loss
    from mlframe.metrics.calibration import compute_ece_and_brier_decomposition
    from mlframe.reporting.charts.calibration import delong_auc_ci
    from mlframe.reporting.charts.binary import _ks_curve

    out: Dict[str, float] = {}
    auc, _, _ = delong_auc_ci(yt, ys)
    out["ROC_AUC"] = float(auc)
    # Average precision (PR-AUC) from the per-distinct-threshold counts, matching sklearn's step-sum form.
    tps, fps, _ = sort.distinct_threshold_counts()
    precision = tps / np.maximum(tps + fps, 1.0)
    recall = tps / max(1, sort.n_pos)
    out["PR_AUC"] = float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))
    ece, _, _, _, _ = compute_ece_and_brier_decomposition(yt.astype(np.float64), ys, _ECE_NBINS)
    out["ECE"] = float(ece)
    out["Brier"] = float(fast_brier_score_loss(yt.astype(np.float64), ys))
    _, _, _, ks_stat, _ = _ks_curve(sort)
    out["KS"] = float(ks_stat)
    pred_pos = ys >= threshold
    tp = float(np.sum(pred_pos & (yt == 1)))
    fp = float(np.sum(pred_pos & (yt == 0)))
    fn = float(np.sum(~pred_pos & (yt == 1)))
    tn = float(np.sum(~pred_pos & (yt == 0)))
    out["MCC"] = _mcc(tp, tn, fp, fn)
    return out


def _classification_verdict(m: Dict[str, float]) -> ModelCardVerdict:
    """GREEN/AMBER/RED on ROC_AUC discrimination, downgraded one step when ECE flags miscalibration."""
    auc = m["ROC_AUC"]
    ece = m["ECE"]
    if auc >= AUC_GREEN:
        color, label = "green", "STRONG"
    elif auc >= AUC_AMBER:
        color, label = "amber", "USABLE"
    else:
        color, label = "red", "WEAK"
    reason = f"ROC_AUC={auc:.3f}"
    # Calibration downgrade: a discriminating but badly-miscalibrated model is not production-green.
    if ece >= ECE_RED and color == "green":
        color, label = "amber", "MISCALIBRATED"
        reason += f"; ECE={ece:.3f} (>= {ECE_RED:g}) drops it from green"
    elif ece >= ECE_AMBER and color == "green":
        reason += f"; calibration borderline (ECE={ece:.3f})"
    else:
        reason += f", ECE={ece:.3f}"
    return ModelCardVerdict(color=color, label=label, reason=reason, headline=m)


def _regression_metrics(yt: np.ndarray, yp: np.ndarray) -> Dict[str, float]:
    """Headline regression metrics: RMSE, MAE, R2, bias (mean residual), heteroscedasticity flag/score."""
    resid = yt - yp
    out: Dict[str, float] = {}
    out["RMSE"] = float(np.sqrt(np.mean(resid * resid)))
    out["MAE"] = float(np.mean(np.abs(resid)))
    var = float(np.var(yt))
    out["R2"] = float(1.0 - np.mean(resid * resid) / var) if var > 0 else 0.0
    out["bias"] = float(np.mean(resid))
    # Spearman(|resid|, y_hat) is the standard heteroscedasticity probe; rank-corr via argsort-of-argsort.
    out["hetero"] = _spearman(np.abs(resid), yp)
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation; NaN-safe, returns 0.0 on degenerate (constant) input."""
    n = a.size
    if n < 3:
        return 0.0
    def _ranks(v: np.ndarray) -> np.ndarray:
        """Ordinal ranks via single argsort + scatter (bit-identical to argsort(argsort(v)), ~1.7-1.9x faster)."""
        order = np.argsort(v)
        r = np.empty(v.size, dtype=np.float64)
        r[order] = np.arange(v.size, dtype=np.float64)
        return r

    ra = _ranks(a)
    rb = _ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt(np.sum(ra * ra) * np.sum(rb * rb))
    return float(np.sum(ra * rb) / denom) if denom > 0 else 0.0


def _regression_verdict(m: Dict[str, float], y_std: float) -> ModelCardVerdict:
    """GREEN/AMBER/RED on R2, downgraded one step when |bias| is large relative to the target spread."""
    r2 = m["R2"]
    if r2 >= R2_GREEN:
        color, label = "green", "STRONG"
    elif r2 >= R2_AMBER:
        color, label = "amber", "USABLE"
    else:
        color, label = "red", "WEAK"
    reason = f"R2={r2:.3f}"
    bias_rel = abs(m["bias"]) / y_std if y_std > 0 else 0.0
    if bias_rel >= BIAS_REL_AMBER and color == "green":
        color, label = "amber", "BIASED"
        reason += f"; |bias|/std={bias_rel:.2f} (>= {BIAS_REL_AMBER:g}) drops it from green"
    else:
        reason += f", bias={m['bias']:+.3g}"
    return ModelCardVerdict(color=color, label=label, reason=reason, headline=m)


def _verdict_color_hex(color: str) -> str:
    """Map a verdict color name ("green"/"amber"/"red") to its hex swatch; unknown names fall back to red."""
    return {"green": _GREEN, "amber": _AMBER, "red": _RED}.get(color, _RED)


def _is_dummy_name(model_name: str) -> bool:
    """A baseline/dummy estimator (model_name like "DummyBaseline:mean") -- tagged so its card is not mistaken for a real model's."""
    return "dummy" in str(model_name).lower()


def _header_panel(model_name: str, split: str, verdict: ModelCardVerdict, metric_fmt: List[Tuple[str, str]]) -> AnnotationPanelSpec:
    """Header text block: title + traffic-light verdict line + reason + headline metric key/value list."""
    dot = {"green": "[GREEN]", "amber": "[AMBER]", "red": "[RED]"}.get(verdict.color, "[RED]")
    # A DUMMY tag on the verdict line makes a baseline card visually distinct from a real model's card at a glance
    # (the two are otherwise near-identical), so operators don't confuse the reference floor for a trained model.
    tag = "[DUMMY] " if _is_dummy_name(model_name) else ""
    lines = [f"{model_name}  # --  {split}", "", f"{tag}{dot} {verdict.label}", verdict.reason, ""]
    lines.extend(f"{name:<10s} {val}" for name, val in metric_fmt)
    title = "MODEL CARD (DUMMY BASELINE)" if _is_dummy_name(model_name) else "MODEL CARD"
    return AnnotationPanelSpec(text="\n".join(lines), title=title, fontsize=11)


def _headline_bar(metric_fmt: List[Tuple[str, float, bool]], verdict_color: str) -> BarPanelSpec:
    """Headline metrics as a horizontal bar (higher-is-better metrics shown directly; lower-is-better inverted to a 0-1 'quality').

    Each ``(name, raw, higher_is_better)`` becomes a bar in [0, 1]: higher-is-better metrics clipped to [0,1] as-is,
    lower-is-better mapped to ``max(0, 1 - raw)`` so a tall bar always reads as 'good'. The verdict color tints the bars.
    """
    cats: List[str] = []
    vals: List[float] = []
    for name, raw, higher in metric_fmt:
        cats.append(name)
        q = raw if higher else (1.0 - raw)
        vals.append(float(np.clip(q, 0.0, 1.0)))
    color = _verdict_color_hex(verdict_color)
    return BarPanelSpec(
        categories=tuple(cats),
        values=np.asarray(vals, dtype=np.float64),
        title="Headline quality (taller = better, [0,1])",
        xlabel="quality",
        ylabel="",
        orientation="horizontal",
        colors=(color,) * len(cats),
        hline=(0.5, "gray", "midpoint"),
    )


def _mini_roc(sort: _ScoreSort) -> PanelSpec:
    """Decimated ROC-curve sparkline reused from the shared score sort; a text annotation when only one class is present."""
    if sort.n_pos == 0 or sort.n_neg == 0:
        return AnnotationPanelSpec(text="ROC n/a\n(one class)", title="mini ROC")
    tps, fps, _ = sort.distinct_threshold_counts()
    tpr = np.concatenate(([0.0], tps / sort.n_pos))
    fpr = np.concatenate(([0.0], fps / sort.n_neg))
    x_thin, (tpr_thin,) = _decimate(fpr, tpr, cap=_MINI_VERTEX_CAP)
    return LinePanelSpec(
        x=x_thin, y=(tpr_thin, x_thin.copy()), series_labels=("ROC", "chance"),
        title="mini ROC", xlabel="FPR", ylabel="TPR",
        line_styles=("-", ":"), colors=("#1f77b4", "gray"),
        fill_to_baseline=(True, False),
    )


def _mini_score_dist(sort: _ScoreSort, yt: np.ndarray, ys: np.ndarray) -> PanelSpec:
    """Score-distribution sparkline: overlaid density histograms of the positive- and negative-class scores."""
    if sort.n == 0:
        return AnnotationPanelSpec(text="score dist n/a", title="mini score dist")
    lo, hi = float(ys.min()), float(ys.max())
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, _MINI_HIST_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    h_neg, _ = np.histogram(ys[yt == 0], bins=edges, density=True)
    h_pos, _ = np.histogram(ys[yt == 1], bins=edges, density=True)
    return LinePanelSpec(
        x=centers, y=(h_neg, h_pos), series_labels=("y=0", "y=1"),
        title="mini score dist", xlabel="score", ylabel="density",
        line_styles=("-", "-"), colors=("#d62728", "#1f77b4"),
        fill_to_baseline=(True, True), step_fill=True,
    )


def _mini_gain(sort: _ScoreSort) -> PanelSpec:
    """Decimated cumulative-gain curve sparkline; a text annotation when there are no positives to capture."""
    if sort.n_pos == 0:
        return AnnotationPanelSpec(text="gain n/a\n(no positives)", title="mini gain")
    pop = np.arange(1, sort.n + 1, dtype=np.float64) / sort.n
    gain = sort.cum_tp.astype(np.float64) / sort.n_pos
    pop = np.concatenate(([0.0], pop))
    gain = np.concatenate(([0.0], gain))
    x_thin, (gain_thin,) = _decimate(pop, gain, cap=_MINI_VERTEX_CAP)
    return LinePanelSpec(
        x=x_thin, y=(gain_thin, x_thin.copy()), series_labels=("model", "baseline"),
        title="mini gain", xlabel="pop. frac", ylabel="pos. captured",
        line_styles=("-", ":"), colors=("#1f77b4", "gray"),
        fill_to_baseline=(True, False),
    )


def _mini_resid_vs_pred(yt: np.ndarray, yp: np.ndarray) -> PanelSpec:
    """Residual-vs-prediction scatter sparkline (subsampled to ``_MINI_SCATTER_SAMPLE`` points) for a quick heteroscedasticity glance."""
    resid = yt - yp
    n = resid.size
    if n == 0:
        return AnnotationPanelSpec(text="resid n/a", title="mini residual-vs-pred")
    if n > _MINI_SCATTER_SAMPLE:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=_MINI_SCATTER_SAMPLE, replace=False)
        sp, sr = yp[idx], resid[idx]
    else:
        sp, sr = yp, resid
    return ScatterPanelSpec(
        x=sp, y=sr, title="mini residual-vs-pred", xlabel="y_hat", ylabel="resid",
        point_color="steelblue", point_alpha=0.3, point_size=8.0,
    )


def _mini_resid_hist(yt: np.ndarray, yp: np.ndarray) -> PanelSpec:
    """Residual-distribution density histogram sparkline (subsampled above 20k points for plotting speed)."""
    resid = yt - yp
    if resid.size == 0:
        return AnnotationPanelSpec(text="resid n/a", title="mini residual hist")
    if resid.size > 20_000:
        rng = np.random.default_rng(0)
        resid = resid[rng.choice(resid.size, size=20_000, replace=False)]
    return HistogramPanelSpec(
        values=resid, bins=_MINI_HIST_BINS, title="mini residual hist",
        xlabel="resid", ylabel="density", density=True,
    )


def _mini_pred_vs_actual(yt: np.ndarray, yp: np.ndarray) -> PanelSpec:
    """Predicted-vs-actual scatter sparkline with a perfect-fit reference line (subsampled to ``_MINI_SCATTER_SAMPLE`` points)."""
    n = yt.size
    if n == 0:
        return AnnotationPanelSpec(text="pred-vs-actual n/a", title="mini pred-vs-actual")
    if n > _MINI_SCATTER_SAMPLE:
        rng = np.random.default_rng(1)
        idx = rng.choice(n, size=_MINI_SCATTER_SAMPLE, replace=False)
        sp, st = yp[idx], yt[idx]
    else:
        sp, st = yp, yt
    return ScatterPanelSpec(
        x=sp, y=st, title="mini pred-vs-actual", xlabel="y_hat", ylabel="y_true",
        perfect_fit_line=True, point_color="steelblue", point_alpha=0.3, point_size=8.0,
    )


def _degenerate_card(model_name: str, split: str, text: str, figsize: Tuple[float, float]) -> FigureSpec:
    """Single-panel fallback card that honestly reports why metrics could not be computed (e.g. single-class / no finite pairs), instead of drawing a misleading chart."""
    ann = AnnotationPanelSpec(text=f"{model_name}  # --  {split}\n\n{text}", title="MODEL CARD", fontsize=11)
    return FigureSpec(suptitle="", panels=((ann,),), figsize=figsize)


def compose_model_card_figure(
    *,
    task: str,
    y_true: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    model_name: str = "model",
    split: str = "test",
    threshold: float = 0.5,
    cell_width: float = 6.0,
    cell_height: float = 4.0,
) -> FigureSpec:
    """Build a one-glance MODEL CARD FigureSpec for one (model, split).

    Parameters
    ----------
    task : "classification" / "binary" -> classification card; "regression" -> regression card.
    y_true : (N,) labels (binary {0,1} for classification) or true values (regression).
    y_score : (N,) positive-class scores -- required for classification.
    y_pred : (N,) predictions -- required for regression.
    model_name / split : identity stamped into the header.
    threshold : operating point for the MCC confusion counts (classification, default 0.5).

    Layout: row 0 = [header key/value + verdict text | headline-quality bar], row 1 = up to 3 mini sparklines.
    Degenerate input (single class / no finite pairs / constant target) returns an honest 1-panel text card.

    The verdict (``ModelCardVerdict``) is also reachable standalone via ``model_card_verdict(...)`` for tests
    and downstream triage logic without rendering.
    """
    t = task.strip().lower()
    figsize = figsize_for_grid(2, 3, cell_width=cell_width, cell_height=cell_height)

    if t in ("classification", "binary"):
        if y_score is None:
            raise ValueError("classification model card requires y_score")
        yt, ys = _finite_binary(y_true, y_score)
        if yt.size == 0:
            return _degenerate_card(model_name, split, "no finite (label, score) pairs", figsize)
        if (yt == 1).sum() == 0 or (yt == 0).sum() == 0:
            return _degenerate_card(model_name, split, "only one class present: discrimination metrics undefined", figsize)
        sort = _ScoreSort(yt, ys)
        m = _classification_metrics(sort, yt, ys, threshold)
        verdict = _classification_verdict(m)
        metric_fmt = [
            ("ROC_AUC", f"{m['ROC_AUC']:.3f}"), ("PR_AUC", f"{m['PR_AUC']:.3f}"),
            ("ECE", f"{m['ECE']:.3f}"), ("Brier", f"{m['Brier']:.3f}"),
            ("KS", f"{m['KS']:.3f}"), ("MCC", f"{m['MCC']:+.3f}"),
        ]
        bar_fmt = [
            ("ROC_AUC", m["ROC_AUC"], True), ("PR_AUC", m["PR_AUC"], True),
            ("KS", m["KS"], True), ("MCC", max(0.0, m["MCC"]), True),
            ("1-ECE", m["ECE"], False), ("1-Brier", m["Brier"], False),
        ]
        header = _header_panel(model_name, split, verdict, metric_fmt)
        bar = _headline_bar(bar_fmt, verdict.color)
        minis = [_mini_roc(sort), _mini_score_dist(sort, yt, ys), _mini_gain(sort)]
        grid = ((header, bar, None), tuple(minis))
        return FigureSpec(
            suptitle=f"Model card -- {'[DUMMY] ' if _is_dummy_name(model_name) else ''}{model_name} ({split}) -- {verdict.label}",
            panels=grid, figsize=figsize, row_height_ratios=(1.2, 1.0),
        )

    if t == "regression":
        if y_pred is None:
            raise ValueError("regression model card requires y_pred")
        yt, yp = _finite_pair(y_true, y_pred)
        if yt.size == 0:
            return _degenerate_card(model_name, split, "no finite (y_true, y_pred) pairs", figsize)
        y_std = float(np.std(yt))
        m = _regression_metrics(yt, yp)
        verdict = _regression_verdict(m, y_std)
        het = "(!) heteroscedastic" if abs(m["hetero"]) >= 0.2 else "homoscedastic"
        metric_fmt = [
            ("RMSE", f"{m['RMSE']:.4g}"), ("MAE", f"{m['MAE']:.4g}"),
            ("R2", f"{m['R2']:.3f}"), ("bias", f"{m['bias']:+.4g}"),
            ("hetero", f"{m['hetero']:+.2f} {het}"),
        ]
        # Bars: R2 directly; RMSE/MAE/bias normalised by target std so a tall bar is good across scales.
        rmse_q = 1.0 - min(1.0, m["RMSE"] / y_std) if y_std > 0 else 0.0
        mae_q = 1.0 - min(1.0, m["MAE"] / y_std) if y_std > 0 else 0.0
        bias_q = 1.0 - min(1.0, abs(m["bias"]) / y_std) if y_std > 0 else 0.0
        bar_fmt = [
            ("R2", max(0.0, m["R2"]), True), ("1-RMSE/std", rmse_q, True),
            ("1-MAE/std", mae_q, True), ("1-|bias|/std", bias_q, True),
        ]
        header = _header_panel(model_name, split, verdict, metric_fmt)
        bar = _headline_bar(bar_fmt, verdict.color)
        minis = [_mini_resid_vs_pred(yt, yp), _mini_resid_hist(yt, yp), _mini_pred_vs_actual(yt, yp)]
        grid = ((header, bar, None), tuple(minis))
        return FigureSpec(
            suptitle=f"Model card -- {'[DUMMY] ' if _is_dummy_name(model_name) else ''}{model_name} ({split}) -- {verdict.label}",
            panels=grid, figsize=figsize, row_height_ratios=(1.2, 1.0),
        )

    raise ValueError(f"unknown task {task!r}; expected 'classification'/'binary'/'regression'")


def model_card_verdict(
    *,
    task: str,
    y_true: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> ModelCardVerdict:
    """Compute the traffic-light verdict + headline metrics without building a figure (triage / tests).

    Raises on degenerate classification input (single class) the same way the metric path would be undefined;
    regression always returns a verdict (R2 is defined whenever the target has spread).
    """
    t = task.strip().lower()
    if t in ("classification", "binary"):
        if y_score is None:
            raise ValueError("classification verdict requires y_score")
        yt, ys = _finite_binary(y_true, y_score)
        if yt.size == 0 or (yt == 1).sum() == 0 or (yt == 0).sum() == 0:
            raise ValueError("classification verdict undefined: need both classes present")
        sort = _ScoreSort(yt, ys)
        return _classification_verdict(_classification_metrics(sort, yt, ys, threshold))
    if t == "regression":
        if y_pred is None:
            raise ValueError("regression verdict requires y_pred")
        yt, yp = _finite_pair(y_true, y_pred)
        if yt.size == 0:
            raise ValueError("regression verdict undefined: no finite pairs")
        return _regression_verdict(_regression_metrics(yt, yp), float(np.std(yt)))
    raise ValueError(f"unknown task {task!r}")


__all__ = [
    "compose_model_card_figure",
    "model_card_verdict",
    "ModelCardVerdict",
    "AUC_GREEN",
    "AUC_AMBER",
    "ECE_AMBER",
    "ECE_RED",
    "R2_GREEN",
    "R2_AMBER",
]
