"""Risk-coverage (selective prediction / selective classification) chart builder.

Selective prediction answers a deployment question ROC / PR / calibration cannot:
if you let the model ABSTAIN on its least-confident cases and defer them to a human,
how much does accuracy on the cases you DO keep improve, and is it worth it?

For each coverage ``c`` (the fraction of cases retained, sorted by confidence
descending) we plot the accuracy (classification) or error (regression) on that
most-confident fraction. A confidence signal that genuinely ranks correctness makes
accuracy RISE as coverage shrinks; a useless confidence signal leaves the curve flat.

The RANDOM-rejection reference is the constant full-data accuracy: abstaining at
random keeps accuracy unchanged at every coverage, so the GAP between the
confidence-ranked curve and that flat line is exactly the value of the model's
confidence ranking. AURC (area under the risk-coverage curve, risk = 1 - accuracy or
the regression error) is the single headline: lower than the random-rejection AURC
means the confidence ranking buys real selective gain.

EFFICIENCY: ONE descending argsort of confidence, then a single cumulative pass --
``np.cumsum`` of the sorted correctness (classification) or sorted error (regression)
divided by the running count gives accuracy/error at every prefix in O(n). The sort
is O(n log n); the curve is O(n). For huge n the per-row curve is DECIMATED to <=2000
plotted points (the AURC is still integrated on the full curve).

cProfile (n=1e6, ``_benchmarks/profile_risk_coverage.py``): ~310 ms/call, of which the
argsort is ~73% and the cumsum ~5%; the curve is argsort-bound as designed -- no
actionable speedup beyond the sort floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from mlframe.reporting.spec import FigureSpec, LinePanelSpec

MAX_PLOT_POINTS: int = 2000

Task = Literal["binary", "multiclass", "regression"]


@dataclass(frozen=True)
class RiskCoverageResult:
    """Risk-coverage curve + spec + headline selective-gain metrics.

    ``coverage`` is the retained fraction grid (ascending); ``accuracy`` / ``risk`` are the metric on the most-confident
    ``coverage`` fraction (for regression ``accuracy`` is NaN and ``risk`` carries the retained error). ``aurc`` is the
    trapezoid area under risk-vs-coverage; ``aurc_random`` is the same for the flat random-rejection reference (= full
    risk). ``metric_at_80`` / ``metric_at_100`` are accuracy (classification) or error (regression) at 80% / full
    coverage; ``selective_gain`` = metric_at_80 - metric_at_100 (positive=accuracy rises / negative=error drops as you
    abstain). ``has_ranking_signal`` is False when confidence is constant (no abstention order -> flat curve).
    """

    figure: FigureSpec
    coverage: np.ndarray
    accuracy: np.ndarray
    risk: np.ndarray
    aurc: float
    aurc_random: float
    metric_at_80: float
    metric_at_100: float
    selective_gain: float
    has_ranking_signal: bool
    task: str


def _binary_confidence(y_score: np.ndarray) -> np.ndarray:
    """Confidence of a binary positive-class probability: distance from the 0.5 decision boundary, scaled to [0,1]."""
    return np.abs(y_score - 0.5) * 2.0


def _multiclass_confidence(proba: np.ndarray) -> np.ndarray:
    """Confidence of a multiclass proba matrix: top class probability (max over classes)."""
    return proba.max(axis=1)


def _prepare_classification(
    y_true, y_score, *, task: Task
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (confidence, correct) finite-only arrays; correct is 1.0 where the argmax/threshold prediction is right.

    Binary ``y_score`` is the positive-class probability (predict iff >=0.5). Multiclass ``y_score`` is an (n, K) proba
    matrix (predict argmax). Rows with any non-finite score or non-finite label are dropped.
    """
    yt = np.asarray(y_true).ravel()
    ys = np.asarray(y_score, dtype=np.float64)
    if task == "multiclass" or ys.ndim == 2:
        proba = ys if ys.ndim == 2 else ys.reshape(-1, 1)
        finite = np.isfinite(proba).all(axis=1) & np.isfinite(yt.astype(np.float64))
        proba = proba[finite]
        yt = yt[finite]
        conf = _multiclass_confidence(proba)
        pred = proba.argmax(axis=1)
        correct = (pred == yt.astype(pred.dtype)).astype(np.float64)
        return conf, correct
    ys = ys.ravel()
    yt_f = yt.astype(np.float64)
    finite = np.isfinite(ys) & np.isfinite(yt_f) & ((yt_f == 0.0) | (yt_f == 1.0))
    ys = ys[finite]
    yt_f = yt_f[finite]
    conf = _binary_confidence(ys)
    pred = (ys >= 0.5).astype(np.float64)
    correct = (pred == yt_f).astype(np.float64)
    return conf, correct


def compute_risk_coverage(
    y_true,
    y_score,
    *,
    task: Task = "binary",
    confidence: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, bool]:
    """Risk-coverage curve via one descending confidence sort + one cumulative pass.

    Returns ``(coverage, accuracy, risk, aurc, full_risk, has_ranking_signal)``. ``coverage[i]`` is the retained
    fraction; ``accuracy[i]`` / ``risk[i]`` are the metric on the most-confident ``coverage[i]`` of rows. For
    classification ``risk = 1 - accuracy``; for regression ``accuracy`` is NaN and ``risk`` is the running MAE.
    ``aurc`` integrates risk over coverage on the FULL curve; ``full_risk`` is the risk at coverage=1 (the flat
    random-rejection reference value). ``has_ranking_signal`` is False when confidence is constant.
    """
    if task == "regression":
        yt = np.asarray(y_true, dtype=np.float64).ravel()
        yp = np.asarray(y_score, dtype=np.float64).ravel()
        if confidence is None:
            raise ValueError("regression risk-coverage requires an explicit `confidence` array (higher = more certain)")
        conf = np.asarray(confidence, dtype=np.float64).ravel()
        finite = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(conf)
        conf = conf[finite]
        loss = np.abs(yt[finite] - yp[finite])
        is_regression = True
    else:
        conf, correct = _prepare_classification(y_true, y_score, task=task)
        loss = 1.0 - correct  # per-row 0/1 error; running mean = risk = 1 - accuracy
        is_regression = False

    n = conf.size
    if n == 0:
        cov = np.array([1.0])
        risk = np.array([np.nan])
        acc = np.array([np.nan])
        return cov, acc, risk, float("nan"), float("nan"), False

    has_signal = bool(np.ptp(conf) > 0.0)

    # Descending confidence sort; cumulative mean of loss over the most-confident prefix gives risk at each coverage.
    order = np.argsort(conf, kind="stable")[::-1]
    loss_sorted = loss[order]
    counts = np.arange(1, n + 1, dtype=np.float64)
    running_risk = np.cumsum(loss_sorted) / counts
    coverage = counts / n

    full_risk = float(running_risk[-1])
    # AURC on the full per-row curve (trapezoid over coverage). Coverage starts at 1/n, not 0; integrate as-is.
    aurc = float(np.trapezoid(running_risk, coverage)) if n > 1 else full_risk

    accuracy = np.full(n, np.nan) if is_regression else (1.0 - running_risk)
    return coverage, accuracy, running_risk, aurc, full_risk, has_signal


def build_risk_coverage_spec(
    y_true,
    y_score,
    *,
    task: Task = "binary",
    confidence: Optional[np.ndarray] = None,
    model_label: str = "model",
    title: str = "Risk-coverage (selective prediction)",
    figsize: Tuple[float, float] = (8.0, 5.8),
) -> RiskCoverageResult:
    """Risk-coverage FigureSpec: accuracy (or error) vs coverage with the flat random-rejection reference.

    Classification draws accuracy-vs-coverage (left axis, rising as you abstain when confidence ranks correctness) plus
    a risk-vs-coverage twin (right axis). Regression draws retained-error-vs-coverage. The flat random-rejection line
    (constant full-data accuracy / error) makes the selective-gain GAP visible. The 80%-coverage operating point is
    marked. Headline = AURC vs random AURC + accuracy@80% vs @100%.
    """
    coverage, accuracy, risk, aurc, full_risk, has_signal = compute_risk_coverage(
        y_true, y_score, task=task, confidence=confidence)
    is_regression = task == "regression"

    # Plain-language explainer wired into the title so a reader who has never seen a risk-coverage curve knows what it
    # shows and how to read it: sort by model confidence, drop the least-confident tail, and watch the error on what
    # is kept. A downward (regression error) / upward (classification accuracy) curve means confidence is informative
    # and you can defer the low-confidence tail; a flat curve means confidence carries no signal.
    how_to_read = (
        "Sort predictions by model confidence; x = coverage (fraction kept after dropping the least-confident tail), "
        "y = error/accuracy on the kept rows. Curve improving as coverage drops => confidence is informative: defer/"
        "flag the low-confidence tail to trade coverage for accuracy. Flat => confidence carries no signal."
    )
    # The random reference (constant full_risk) must be integrated over the SAME coverage domain as `aurc`
    # (np.trapezoid over coverage in [1/n, 1]), not over [0, 1]; otherwise "AURC vs random" is biased by the
    # domain mismatch. trapezoid of a constant c over [a, b] = c * (b - a) = full_risk * (coverage[-1] - coverage[0]).
    n = coverage.size
    if n > 1:
        aurc_random = full_risk * float(coverage[-1] - coverage[0])
    else:
        aurc_random = full_risk

    def _interp_at(cov_target: float) -> float:
        if n == 0 or not np.isfinite(coverage).any():
            return float("nan")
        src = risk if is_regression else accuracy
        return float(np.interp(cov_target, coverage, src))

    metric_at_80 = _interp_at(0.8)
    metric_at_100 = (float(risk[-1]) if is_regression else float(accuracy[-1])) if n else float("nan")
    # Selective gain: classification wants accuracy UP (positive good); regression wants error DOWN (negative good).
    selective_gain = metric_at_80 - metric_at_100

    # Decimate to <=MAX_PLOT_POINTS for huge n (AURC already computed on the full curve).
    if n > MAX_PLOT_POINTS:
        sel = np.unique(np.linspace(0, n - 1, MAX_PLOT_POINTS).astype(np.int64))
        cov_p, acc_p, risk_p = coverage[sel], accuracy[sel], risk[sel]
    else:
        cov_p, acc_p, risk_p = coverage, accuracy, risk

    note = "" if has_signal else "  [constant confidence: no ranking signal -> flat curve]"

    # Auto-verdict: quantify what deferring the worst 20% buys. Error at 100% vs 80% coverage is the concrete,
    # actionable number a reader wants -- "deferring the least-confident 20% cuts error by X%" (or "no measurable gain").
    err_at_100 = full_risk
    err_at_80 = (1.0 - metric_at_80) if (not is_regression and np.isfinite(metric_at_80)) else metric_at_80
    if not has_signal:
        verdict = "Verdict: confidence is constant -> no rows to defer, deferral cannot help."
    elif np.isfinite(err_at_100) and np.isfinite(err_at_80) and err_at_100 > 0:
        rel = (err_at_100 - err_at_80) / err_at_100 * 100.0
        if rel >= 1.0:
            verdict = f"Verdict: deferring the least-confident 20% cuts error {rel:.0f}% ({err_at_100:.3g} -> {err_at_80:.3g} on the kept 80%)."
        elif rel <= -1.0:
            verdict = f"Verdict: confidence is anti-informative -- deferring the 'worst' 20% RAISES error {abs(rel):.0f}%; do not gate on it."
        else:
            verdict = f"Verdict: deferring the worst 20% barely moves error ({err_at_100:.3g} -> {err_at_80:.3g}); confidence carries little signal."
    else:
        verdict = "Verdict: insufficient data to quantify a selective gain."

    # Title carries only the headline + the actionable verdict; the multi-sentence how-to-read goes to the figure
    # caption (bottom footnote) so it doesn't swallow the chart area.
    title_full = f"{title}{note}\n{verdict}"

    if is_regression:
        flat = np.full_like(cov_p, full_risk)
        markers = ()
        if np.isfinite(metric_at_80):
            markers = ((0.8, metric_at_80, f"err@80%={metric_at_80:.3g}", "#1f77b4", "*"),)
        line = LinePanelSpec(
            x=cov_p,
            y=(risk_p, flat),
            series_labels=(
                f"{model_label} retained MAE (AURC={aurc:.3g})",
                f"random rejection (MAE={full_risk:.3g})",
            ),
            line_styles=("-", "--"),
            colors=("#1f77b4", "#7f7f7f"),
            title=title_full,
            xlabel="Coverage (fraction retained, most-confident first)",
            ylabel="Retained error (MAE)",
            point_markers=markers,
        )
    else:
        flat_acc = np.full_like(cov_p, 1.0 - full_risk)
        markers = ()
        if np.isfinite(metric_at_80):
            markers = ((0.8, metric_at_80, f"acc@80%={metric_at_80:.3g}", "#2ca02c", "*"),)
        line = LinePanelSpec(
            x=cov_p,
            y=(acc_p, flat_acc, risk_p),
            series_labels=(
                f"{model_label} accuracy (AURC={aurc:.3g})",
                f"random rejection (acc={1.0 - full_risk:.3g})",
                "risk = 1 - accuracy",
            ),
            line_styles=("-", "--", ":"),
            colors=("#2ca02c", "#7f7f7f", "#d62728"),
            secondary_y=(False, False, True),
            secondary_ylabel="Risk (1 - accuracy)",
            title=title_full,
            xlabel="Coverage (fraction retained, most-confident first)",
            ylabel="Accuracy on retained",
            point_markers=markers,
        )

    fig = FigureSpec(suptitle="", panels=((line,),), figsize=figsize, caption=how_to_read)
    return RiskCoverageResult(
        figure=fig,
        coverage=coverage,
        accuracy=accuracy,
        risk=risk,
        aurc=aurc,
        aurc_random=aurc_random,
        metric_at_80=metric_at_80,
        metric_at_100=metric_at_100,
        selective_gain=selective_gain,
        has_ranking_signal=has_signal,
        task=task,
    )


__all__ = [
    "RiskCoverageResult",
    "compute_risk_coverage",
    "build_risk_coverage_spec",
    "MAX_PLOT_POINTS",
]
