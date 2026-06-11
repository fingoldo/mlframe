"""Render a visual gallery of every mlframe reporting chart to PNG on synthetic data.

Run: python scripts/render_gallery.py
Writes docs/gallery/<category>/<name>.png plus docs/gallery/index.md and (on failure) docs/gallery/_errors.log.

Each entry synthesizes small data that makes the chart MEANINGFUL (separable target for ROC,
heteroscedastic regression for residuals, graded relevance for LTR, drifting feature for PSI, etc.),
builds the composer's FigureSpec, and renders it via the matplotlib renderer to a real figure a human
can judge. Every render is wrapped so one failure never aborts the rest.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MLFRAME_PLOT_INLINE_DISPLAY", "0")

import traceback
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GALLERY = REPO / "docs" / "gallery"
HEARTBEAT = REPO / "audit" / "viz_followup_2026_06_11" / "HEARTBEAT_GALLERY.txt"
ERRORS_LOG = GALLERY / "_errors.log"

RNG = np.random.default_rng(20260611)


def heartbeat(msg: str) -> None:
    try:
        HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT.write_text(msg + "\n", encoding="ascii", errors="ignore")
    except Exception:
        pass


# Lazy: one renderer instance reused for every save.
_RENDERER = None


def _renderer():
    global _RENDERER
    if _RENDERER is None:
        from mlframe.reporting.renderers.base import get_renderer
        _RENDERER = get_renderer("matplotlib")
    return _RENDERER


def _save_spec(spec, category: str, name: str) -> str:
    """Save a builder's output as docs/gallery/<category>/<name>.png. Returns the relative path.

    Accepts either a ``FigureSpec`` (rendered via the active backend) or an already-built matplotlib ``Figure``
    (the direct-matplotlib builders, e.g. the decile table / SHAP-style panels, which return a Figure themselves)."""
    out_dir = GALLERY / category
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / name
    from mlframe.reporting.spec import FigureSpec
    if isinstance(spec, FigureSpec):
        rend = _renderer()
        fig = rend.render(spec)
        save_fn = lambda: rend.save(fig, str(base) + ".png", "png")
    else:
        fig = spec  # already a matplotlib Figure
        save_fn = lambda: fig.savefig(str(base) + ".png", bbox_inches="tight", pad_inches=0.15)
    try:
        save_fn()
    finally:
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass
    return f"{category}/{name}.png"


# Registry of (category, name, description, builder). Builder returns a FigureSpec.
ENTRIES: List[Tuple[str, str, str, Callable]] = []


def entry(category: str, name: str, description: str):
    def deco(fn: Callable):
        ENTRIES.append((category, name, description, fn))
        return fn
    return deco


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _binary_separable(n: int = 4000, sep: float = 1.3):
    """Separable binary target + a calibrated-ish score in [0,1]."""
    y = (RNG.random(n) < 0.4).astype(np.int8)
    raw = RNG.normal(loc=sep * y, scale=1.0, size=n)
    score = 1.0 / (1.0 + np.exp(-raw))
    return y, score


def _heteroscedastic_regression(n: int = 4000):
    """y_true / y_pred with funnel heteroscedasticity + a slight skew so residual panels are meaningful."""
    x = RNG.uniform(0.0, 10.0, size=n)
    y_true = 2.0 * x + 5.0 + RNG.normal(0.0, 0.5 + 0.4 * x, size=n)
    y_pred = 2.0 * x + 5.0 + RNG.normal(0.0, 0.2, size=n)
    return y_true, y_pred


# ---------------------------------------------------------------------------
# REGRESSION
# ---------------------------------------------------------------------------


@entry("regression", "regression_full", "Pred-vs-actual scatter, residual hist, residual-vs-pred, error-by-decile, worm, residual ACF.")
def _b():
    from mlframe.reporting.charts.regression import compose_regression_figure
    yt, yp = _heteroscedastic_regression()
    return compose_regression_figure(
        yt, yp, panels_template="SCATTER RESID_HIST RESID_VS_PRED ERR_BY_DECILE WORM RESID_ACF",
        suptitle="Regression diagnostics (heteroscedastic synthetic)",
    )


@entry("regression", "regression_hexbin_largen", "Large-n (>50k) pred-vs-actual drawn as a log-density 2-D histogram instead of a point cloud.")
def _b():
    from mlframe.reporting.charts.regression import compose_regression_figure
    n = 120_000
    x = RNG.uniform(0.0, 10.0, size=n)
    yt = 2.0 * x + 5.0 + RNG.normal(0.0, 1.0, size=n)
    yp = 2.0 * x + 5.0 + RNG.normal(0.0, 0.7, size=n)
    return compose_regression_figure(
        yt, yp, panels_template="SCATTER", suptitle=f"Pred-vs-actual log-density ({n:_} pts)",
    )


# ---------------------------------------------------------------------------
# BINARY
# ---------------------------------------------------------------------------


@entry("binary", "binary_full", "ROC, PR, score distribution, KS, threshold sweep, gain, PIT.")
def _b():
    from mlframe.reporting.charts.binary import compose_binary_figure
    y, score = _binary_separable()
    return compose_binary_figure(
        y, score, panels_template="ROC PR SCORE_DIST KS THRESHOLD GAIN PIT",
        cost_ratio=(1.0, 5.0), suptitle="Binary classification diagnostics",
    )


@entry("binary", "panel_emphasis_imbalanced",
       "Data-aware panel emphasis on a rare-event target (base rate ~0.03): the adaptive order leads with PR + threshold-sweep and "
       "drops the optimistic-under-imbalance ROC, so the operator sees the diagnostics that actually matter under skew first.")
def _b():
    from mlframe.reporting.charts.binary import compose_binary_figure
    from mlframe.reporting.auto_dispatch import select_binary_emphasis_panels
    y = (RNG.random(6000) < 0.03).astype(np.int8)
    raw = RNG.normal(loc=1.6 * y, scale=1.0, size=len(y))
    score = 1.0 / (1.0 + np.exp(-raw))
    template = select_binary_emphasis_panels(
        y, "ROC PR SCORE_DIST KS THRESHOLD GAIN PIT", emphasis="data_aware",
    )
    return compose_binary_figure(
        y, score, panels_template=template,
        suptitle=f"Data-aware emphasis (base rate ~0.03): {template}",
    )


@entry("binary", "panel_emphasis_balanced",
       "Data-aware panel emphasis on a balanced target (base rate ~0.5): the adaptive order leads with ROC, which is informative when "
       "the classes are even, ahead of PR / score-dist / KS / threshold.")
def _b():
    from mlframe.reporting.charts.binary import compose_binary_figure
    from mlframe.reporting.auto_dispatch import select_binary_emphasis_panels
    y = (RNG.random(6000) < 0.5).astype(np.int8)
    raw = RNG.normal(loc=1.3 * y, scale=1.0, size=len(y))
    score = 1.0 / (1.0 + np.exp(-raw))
    template = select_binary_emphasis_panels(
        y, "ROC PR SCORE_DIST KS THRESHOLD GAIN PIT", emphasis="data_aware",
    )
    return compose_binary_figure(
        y, score, panels_template=template,
        suptitle=f"Data-aware emphasis (base rate ~0.5): {template}",
    )


@entry("binary", "decile_table",
       "Credit-scoring decile gain/lift table: per-decile response / cumulative-gain / lift / cumulative-KS (top deciles highlighted, TOTAL row).")
def _b():
    from mlframe.reporting.charts.binary import binary_decile_table_figure
    y, score = _binary_separable()
    return binary_decile_table_figure(y, score, title="Decile gain / lift table (separable synthetic)")


@entry("binary", "decision_curve", "Decision-curve analysis: model net-benefit vs treat-all / treat-none policies.")
def _b():
    from mlframe.reporting.charts.decision_curve import build_decision_curve_spec
    y, score = _binary_separable()
    return build_decision_curve_spec(y, score, model_label="synthetic model").figure


@entry("binary", "calibration_reliability", "Reliability diagram with Wilson CI bands + binning-free smoothed (isotonic) overlay + bootstrap 95% band (significant-fraction annotation) + standard & debiased ECE annotation + population histogram.")
def _b():
    from mlframe.metrics.calibration._calibration_plot import fast_calibration_binning
    from mlframe.reporting.charts.calibration import build_calibration_spec
    y, score = _binary_separable(n=6000)
    fp, ftr, hits = fast_calibration_binning(y.astype(np.int64), score, nbins=15)
    return build_calibration_spec(fp, ftr, hits, plot_title="Reliability (Wilson CI + smoothed isotonic)",
                                  raw_probs=score, raw_labels=y.astype(np.int64))


# ---------------------------------------------------------------------------
# MODEL CARD
# ---------------------------------------------------------------------------


@entry("model_card", "model_card_binary", "One-glance executive model card: headline metrics + GREEN traffic-light verdict + mini ROC / score-dist / gain sparklines.")
def _b():
    from mlframe.reporting.charts.model_card import compose_model_card_figure
    # Strong AND calibrated: labels drawn from a peaked score so AUC is high and ECE is low -> GREEN.
    raw = RNG.normal(0.0, 2.5, 6000)
    p = 1.0 / (1.0 + np.exp(-raw))
    y = (RNG.random(6000) < p).astype(np.int8)
    return compose_model_card_figure(task="classification", y_true=y, y_score=p, model_name="lgbm_strong", split="test")


@entry("model_card", "model_card_regression", "Regression model card: RMSE/MAE/R2/bias headline + verdict + mini residual-vs-pred / residual-hist / pred-vs-actual sparklines.")
def _b():
    from mlframe.reporting.charts.model_card import compose_model_card_figure
    x = RNG.uniform(0.0, 10.0, 6000)
    yt = 2.0 * x + 5.0 + RNG.normal(0.0, 0.4, 6000)
    yp = 2.0 * x + 5.0 + RNG.normal(0.0, 0.4, 6000)
    return compose_model_card_figure(task="regression", y_true=yt, y_pred=yp, model_name="ridge", split="oof")


# ---------------------------------------------------------------------------
# MULTICLASS
# ---------------------------------------------------------------------------


def _multiclass_data(n: int = 4000, K: int = 4):
    y = RNG.integers(0, K, size=n)
    logits = RNG.normal(0.0, 1.0, size=(n, K))
    logits[np.arange(n), y] += 2.0  # signal so confusion / ROC are meaningful
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    proba = e / e.sum(axis=1, keepdims=True)
    return y, proba, list(range(K))


@entry("multiclass", "multiclass_full", "Normalized confusion, confused pairs, per-class P/R/F1, per-class ROC (DeLong CI), reliability, prob dist, top-k.")
def _b():
    from mlframe.reporting.charts.multiclass import compose_multiclass_figure
    y, proba, classes = _multiclass_data()
    return compose_multiclass_figure(
        y, proba, classes,
        panels_template="CONFUSION CONFUSED_PAIRS PR_F1 ROC CALIB_GRID PROB_DIST TOP_K_ACC",
        suptitle="Multiclass diagnostics (K=4)",
    )


@entry("multiclass", "confusion_margins",
       "Confusion heatmap flanked by class-support margins: right bar = per-true-class support, top bar = per-predicted-class volume. "
       "On an imbalanced + majority-biased synthetic the dominant class's support bar towers over the minorities and its predicted-volume "
       "bar exceeds its support, revealing imbalance + over-prediction at a glance.")
def _b():
    from mlframe.reporting.charts.multiclass import compose_multiclass_figure
    n, K = 6000, 4
    prevalence = np.array([0.6, 0.2, 0.13, 0.07])  # class 0 dominant
    y = RNG.choice(K, size=n, p=prevalence)
    proba = RNG.dirichlet([1.0] * K, size=n)
    for i, t in enumerate(y):
        proba[i, t] += 0.5      # genuine signal on the true class
        proba[i, 0] += 1.0      # majority-class bias so the model over-predicts class 0
    proba /= proba.sum(axis=1, keepdims=True)
    return compose_multiclass_figure(
        y, proba, list(range(K)),
        panels_template="CONFUSION_MARGINS",
        suptitle="Confusion + class-support margins (imbalanced 60/20/13/7, majority-biased model)",
        max_cols=1, cell_width=8.0, cell_height=6.0,
    )


@entry("multiclass", "multiclass_largeK",
       "Large-K (K=40): per-class ROC / PR / reliability overlays auto-switch to the 8 worst-by-AUC classes + a macro-average instead of 40 spaghetti curves.")
def _b():
    from mlframe.reporting.charts.multiclass import compose_multiclass_figure
    K = 40
    y, proba, classes = _multiclass_data(n=20_000, K=K)
    # Wipe the signal on a few classes so their one-vs-rest AUC is low and they surface as the drawn "worst" curves.
    for h in (4, 17, 29, 36):
        proba[:, h] = RNG.random(proba.shape[0])
    proba /= proba.sum(axis=1, keepdims=True)
    return compose_multiclass_figure(
        y, proba, classes,
        panels_template="CONFUSION ROC PR_CURVES CALIB_GRID",
        suptitle=f"Multiclass diagnostics (K={K}): overlays show 8 worst-by-AUC classes + macro-avg",
    )


# ---------------------------------------------------------------------------
# MULTILABEL
# ---------------------------------------------------------------------------


def _multilabel_data(n: int = 4000, K: int = 5):
    base = RNG.random((n, K))
    # Correlated labels so co-occurrence + cardinality are meaningful.
    y_true = np.zeros((n, K), dtype=np.int8)
    y_true[:, 0] = (base[:, 0] < 0.4).astype(np.int8)
    for k in range(1, K):
        y_true[:, k] = ((base[:, k] < 0.3) | (y_true[:, 0] & (base[:, k] < 0.6))).astype(np.int8)
    proba = np.clip(0.15 + 0.7 * y_true + RNG.normal(0.0, 0.18, size=(n, K)), 0.0, 1.0)
    return y_true, proba, [f"lbl{k}" for k in range(K)]


@entry("multilabel", "multilabel_full", "Per-label P/R/F1, reliability, co-occurrence, cardinality, Jaccard dist, threshold-sweep heatmap.")
def _b():
    from mlframe.reporting.charts.multilabel import compose_multilabel_figure
    yt, proba, labels = _multilabel_data()
    return compose_multilabel_figure(
        yt, proba, labels,
        panels_template="PR_F1 CALIB_GRID COOCCURRENCE CARDINALITY JACCARD_DIST THRESHOLD_SWEEP",
        suptitle="Multilabel diagnostics (K=5)",
    )


# ---------------------------------------------------------------------------
# LTR
# ---------------------------------------------------------------------------


def _ltr_data(n_queries: int = 300):
    rels: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    gids: List[np.ndarray] = []
    for q in range(n_queries):
        size = int(RNG.integers(2, 25))
        rel = RNG.integers(0, 4, size=size)  # graded 0..3
        score = rel + RNG.normal(0.0, 0.9, size=size)  # ranker correlates with relevance
        rels.append(rel)
        scores.append(score)
        gids.append(np.full(size, q))
    return (np.concatenate(rels), np.concatenate(scores), np.concatenate(gids))


@entry("ltr", "ltr_full", "NDCG@k, per-query NDCG dist, NDCG by query size, lift, MRR dist, score-by-relevance, top-1 by query size.")
def _b():
    from mlframe.reporting.charts.ltr import compose_ltr_figure
    rel, score, gids = _ltr_data()
    return compose_ltr_figure(
        rel, score, gids,
        panels_template="NDCG_K NDCG_DIST NDCG_BY_QSIZE LIFT MRR_DIST SCORE_BY_REL TOP1_BY_QSIZE",
        suptitle="Learning-to-rank diagnostics",
    )


# ---------------------------------------------------------------------------
# QUANTILE
# ---------------------------------------------------------------------------


def _quantile_data(n: int = 4000):
    from scipy.stats import norm
    alphas = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    x = RNG.uniform(0.0, 10.0, size=n)
    sigma = 0.5 + 0.4 * x  # heteroscedastic so interval width varies
    y = 2.0 * x + RNG.normal(0.0, sigma)
    mu = 2.0 * x
    preds = np.empty((n, len(alphas)), dtype=np.float64)
    for j, a in enumerate(alphas):
        preds[:, j] = mu + norm.ppf(a) * sigma
    return y, preds, alphas


@entry("quantile", "quantile_full", "Reliability, coverage, pinball-by-alpha, interval band, width dist, PIT, quantile reliability, pinball decomp, crossing, fan chart.")
def _b():
    from mlframe.reporting.charts.quantile import compose_quantile_figure
    y, preds, alphas = _quantile_data()
    return compose_quantile_figure(
        y, preds, alphas,
        panels_template="RELIABILITY COVERAGE PINBALL_BY_ALPHA INTERVAL_BAND WIDTH_DIST PIT_HIST "
                         "QUANTILE_RELIABILITY PINBALL_DECOMP QUANTILE_CROSSING FAN_CHART",
        suptitle="Quantile-regression diagnostics",
    )


# ---------------------------------------------------------------------------
# DRIFT
# ---------------------------------------------------------------------------


def _drift_frames(n: int = 5000, K: int = 6):
    """Train / test frames where a few features drift, plus timestamps."""
    names = [f"f{i}" for i in range(K)]
    train = RNG.normal(0.0, 1.0, size=(n, K))
    test = train.copy()
    test = RNG.normal(0.0, 1.0, size=(n, K))
    # Inject drift into 2 features on the test side.
    test[:, 1] += 1.5
    test[:, 3] *= 2.0
    return train, test, names


@entry("drift", "psi_heatmap", "Population Stability Index per feature per time bucket vs baseline (drifted features turn red).")
def _b():
    from mlframe.reporting.charts.drift import psi_heatmap
    n, K = 6000, 6
    names = [f"f{i}" for i in range(K)]
    ts = np.arange(n)
    feats = RNG.normal(0.0, 1.0, size=(n, K))
    # Make f1, f3 drift over time (later rows shifted/scaled).
    ramp = ts / n
    feats[:, 1] += 2.0 * ramp
    feats[:, 3] *= (1.0 + 2.0 * ramp)
    return psi_heatmap(feats, ts, feature_names=names, n_time_buckets=8)


@entry("drift", "residual_vs_time", "Regression residual mean +/- std per time bin: bias drift + variance drift over time.")
def _b():
    from mlframe.reporting.charts.drift import residual_vs_time
    n = 6000
    ts = np.arange(n)
    yt = RNG.normal(0.0, 1.0, size=n)
    # Residual bias drifts up and variance widens over time.
    ramp = ts / n
    yp = yt - (0.5 * ramp) + RNG.normal(0.0, 0.2 + 0.8 * ramp)
    return residual_vs_time(yt, yp, ts, x_is_time=False)


@entry("drift", "cusum_residual_drift", "Two-sided tabular CUSUM of standardized residuals: a sustained mean shift trips the control limit (change-point marked).")
def _b():
    from mlframe.reporting.charts.drift import cusum_residual_drift
    n = 6000
    shift = n // 2
    ts = np.arange(n)
    yt = RNG.normal(0.0, 1.0, size=n)
    # Residual is unbiased in the first half, then a sustained +1 sigma mean shift kicks in -- the CUSUM accumulates
    # the small persistent bias and crosses its control limit shortly after the break.
    resid = np.concatenate([RNG.normal(0.0, 1.0, shift), RNG.normal(1.0, 1.0, n - shift)])
    yp = yt - resid
    return cusum_residual_drift(yt, yp, ts, x_is_time=False, decision_h=10.0)


@entry("drift", "metric_over_time", "Rolling metric per time bucket with regime shading.")
def _b():
    import pandas as pd
    from mlframe.reporting.charts.drift import metric_over_time
    n = 4000
    days = pd.date_range("2025-01-01", periods=n, freq="h")
    y = (RNG.random(n) < 0.4).astype(np.int8)
    # AUC degrades over time: signal weakens.
    ramp = np.arange(n) / n
    score = 1.0 / (1.0 + np.exp(-(RNG.normal((1.2 - ramp) * y, 1.0))))
    regimes = [(days[0], days[n // 2], "tab:green", "regime A"),
               (days[n // 2], days[-1], "tab:red", "regime B")]
    return metric_over_time(y, score, days, metric="roc_auc", freq="D", min_samples=20, regimes=regimes)


@entry("drift", "adversarial_validation", "Train-vs-test LightGBM separability ROC + AUC + top drifting features.")
def _b():
    import pandas as pd
    from mlframe.reporting.charts.drift import adversarial_validation
    train, test, names = _drift_frames()
    return adversarial_validation(
        pd.DataFrame(train, columns=names), pd.DataFrame(test, columns=names), n_splits=3,
    )


# ---------------------------------------------------------------------------
# CALIBRATION DRIFT
# ---------------------------------------------------------------------------


@entry("calibration_drift", "calibration_drift", "ECE-over-time line + small-multiple per-window reliability curves.")
def _b():
    import pandas as pd
    from mlframe.reporting.charts.calibration_drift import build_calibration_drift_spec, calibration_drift
    n = 8000
    ts = pd.date_range("2025-01-01", periods=n, freq="h")
    y = (RNG.random(n) < 0.4).astype(np.int8)
    ramp = np.arange(n) / n
    # Score calibration decays over time (logits scaled by a growing factor -> over-confidence).
    logit = RNG.normal(1.2 * y - 0.5, 1.0)
    score = 1.0 / (1.0 + np.exp(-logit * (1.0 + 2.0 * ramp)))
    res = calibration_drift(y, score, ts, n_windows=10, n_bins=10)
    return build_calibration_drift_spec(res)


# ---------------------------------------------------------------------------
# ERROR ANALYSIS
# ---------------------------------------------------------------------------


def _error_data(n: int = 6000, K: int = 5):
    """Regression with an injected bad region in (f0, f1) space."""
    names = [f"f{i}" for i in range(K)]
    X = RNG.normal(0.0, 1.0, size=(n, K))
    yt = X @ RNG.normal(0.0, 1.0, size=K) + RNG.normal(0.0, 0.3, size=n)
    yp = yt + RNG.normal(0.0, 0.3, size=n)
    # Inject a bad region: where f0>1 and f1>1, predictions are badly off.
    bad = (X[:, 0] > 1.0) & (X[:, 1] > 1.0)
    yp[bad] += RNG.normal(4.0, 1.0, size=int(bad.sum()))
    return X, yt, yp, names


@entry("error_analysis", "weak_segment_heatmap", "FreaAI-style weak-segment grid: mean error by feature slice (injected bad region shows as a hot cell).")
def _b():
    import pandas as pd
    from mlframe.reporting.charts.error_analysis import weak_segment_heatmap
    X, yt, yp, names = _error_data()
    return weak_segment_heatmap(pd.DataFrame(X, columns=names), yt, yp, task="regression").figure


@entry("error_analysis", "error_bias_per_feature", "Evidently-style OVER/UNDER/MAJORITY feature-value distributions per feature.")
def _b():
    import pandas as pd
    from mlframe.reporting.charts.error_analysis import error_bias_per_feature
    X, yt, yp, names = _error_data()
    return error_bias_per_feature(pd.DataFrame(X, columns=names), yt, yp, max_features=4).figure


@entry("error_analysis", "target_dist_overlay", "Per-split overlaid density histograms of target and predictions (train p01/p99 envelope).")
def _b():
    from mlframe.reporting.charts.error_analysis import target_dist_overlay
    train_y = RNG.normal(0.0, 1.0, size=3000)
    test_y = RNG.normal(0.6, 1.2, size=2000)  # target shift
    oof_pred = train_y + RNG.normal(0.0, 0.3, size=3000)
    test_pred = test_y + RNG.normal(0.0, 0.3, size=2000)
    return target_dist_overlay(
        {"train": train_y, "test": test_y},
        pred_by_split={"oof": oof_pred, "test": test_pred},
        task="regression",
    )


@entry("error_analysis", "segments_bar", "Per-subgroup metric bars with a global-reference line (worst-first).")
def _b():
    import pandas as pd
    from mlframe.reporting.charts.error_analysis import segments_bar
    df = pd.DataFrame({
        "segment": [f"grp{i}" for i in range(8)],
        "rmse": np.sort(RNG.uniform(0.5, 3.0, size=8)),
        "count": RNG.integers(100, 2000, size=8),
    })
    return segments_bar(df, group_col="segment", metric_col="rmse", metric_name="RMSE", higher_is_worse=True)


# ---------------------------------------------------------------------------
# PREDICTION STABILITY
# ---------------------------------------------------------------------------


@entry("prediction_stability", "prediction_stability",
       "Ensemble member-disagreement: per-row spread histogram, spread-vs-mean scatter, uncertainty calibration (mean |error| rises with disagreement).")
def _b():
    from mlframe.reporting.charts.prediction_stability import compose_prediction_stability_figure
    n, m = 6000, 10
    y_true = RNG.normal(0.0, 1.0, size=n)
    easy = np.zeros(n, dtype=bool)
    easy[: n // 2] = True
    # Easy region: members agree, predictions near truth. Hard region: members scatter widely AND carry a real error.
    member_noise = np.where(easy[:, None], 0.05, 1.2)
    bias = np.where(easy, 0.0, RNG.normal(0.0, 1.0, size=n))
    preds = (y_true + bias)[:, None] + RNG.normal(0.0, 1.0, size=(n, m)) * member_noise
    return compose_prediction_stability_figure(preds, y_true=y_true,
                                               suptitle="Ensemble prediction stability (easy vs hard region)")


# ---------------------------------------------------------------------------
# SLICE FINDER
# ---------------------------------------------------------------------------


@entry("slice_finder", "slice_finder", "Worst-K feature-value slices ranked by error-degradation x support.")
def _b():
    import pandas as pd
    from mlframe.reporting.charts.slice_finder import find_weak_slices
    X, yt, yp, names = _error_data()
    return find_weak_slices(pd.DataFrame(X, columns=names), yt, yp, task="regression", max_arity=2).figure


@entry("fairness_calibration", "fairness_calibration",
       "Per-subgroup reliability overlay + per-group ECE bar; max-min ECE gap as a calibration-fairness disparity (one group deliberately miscalibrated).")
def _b():
    from mlframe.reporting.charts.fairness_calibration import compose_fairness_calibration_figure
    n = 12000
    g = RNG.integers(0, 3, n)
    score = RNG.random(n)
    y = (RNG.random(n) < score).astype(np.int64)
    # Group 2 is made overconfident (concave score warp) -> high ECE while groups 0/1 stay calibrated.
    score_mis = score.copy()
    score_mis[g == 2] = np.clip(score[g == 2] ** 0.35, 1e-3, 1 - 1e-3)
    labels = np.array(["region_A", "region_B", "region_C"])[g]
    return compose_fairness_calibration_figure(y, score_mis, labels)


@entry("calibration_by_feature", "calibration_by_feature",
       "Per-feature calibration: reliability + ECE conditioned on quantile bins of a continuous feature. The model is "
       "calibrated for low feature values but overconfident for high ones, so the per-bin ECE line climbs across the "
       "feature range and the max-min heterogeneity metric trips red -- a miscalibration a single pooled curve hides.")
def _b():
    from mlframe.reporting.charts.calibration_by_feature import compose_calibration_by_feature_figure
    n = 16000
    feature = RNG.normal(0.0, 1.0, size=n)
    score = RNG.random(n)
    y = (RNG.random(n) < score).astype(np.int64)
    # Overconfident only where the feature is high (concave warp on the upper half) -> calibration degrades with feature.
    high = feature > np.median(feature)
    score[high] = np.clip(score[high] ** 0.3, 1e-3, 1 - 1e-3)
    return compose_calibration_by_feature_figure(y, score, feature, feature_name="risk_score", n_feature_bins=4)


@entry("calibration_heatmap_2d", "calibration_heatmap_2d",
       "2D calibration-ECE heatmap over a quantile grid of two features. The model is overconfident ONLY in the high-f0 "
       "AND high-f1 corner (a localized pocket either 1D view averages away); that corner cell lights up red on the "
       "RdYlGn_r grid while the rest stays green, and the worst-cell ECE + location is the headline.")
def _b2():
    from mlframe.reporting.charts.calibration_heatmap_2d import compose_calibration_heatmap_2d_figure
    n = 60000
    fx = RNG.normal(0.0, 1.0, size=n)
    fy = RNG.normal(0.0, 1.0, size=n)
    base = 1.0 / (1.0 + np.exp(-(0.8 * fx + 0.8 * fy)))
    y = (RNG.random(n) < base).astype(np.int64)
    score = base.copy()
    corner = (fx > np.median(fx)) & (fy > np.median(fy))
    score[corner] = np.clip(score[corner] + 0.35, 1e-3, 1 - 1e-3)  # overconfident pocket only at the joint high corner
    return compose_calibration_heatmap_2d_figure(y, score, fx, fy, feat_x_name="f0", feat_y_name="f1", n_bins=5)


# ---------------------------------------------------------------------------
# PDP / ICE
# ---------------------------------------------------------------------------


@entry("pdp_ice", "pdp_ice", "1-D PDP+ICE for the top features and a 2-D PDP interaction heatmap (small sklearn model).")
def _b():
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from mlframe.reporting.charts.pdp_ice import compose_pdp_figure
    n, K = 3000, 5
    names = [f"f{i}" for i in range(K)]
    X = RNG.normal(0.0, 1.0, size=(n, K))
    y = 3.0 * X[:, 0] + 2.0 * np.sin(X[:, 1]) + X[:, 0] * X[:, 2] + RNG.normal(0.0, 0.3, size=n)
    Xdf = pd.DataFrame(X, columns=names)
    model = RandomForestRegressor(n_estimators=40, max_depth=6, random_state=0, n_jobs=1).fit(Xdf, y)
    return compose_pdp_figure(
        model, Xdf, ["f0", "f1", "f2"], interaction_pair=("f0", "f2"),
        suptitle="Partial dependence / ICE", sample=600, grid=15,
    )


# ---------------------------------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------------------------------


@entry("model_comparison", "model_comparison", "ROC overlay + leaderboard bars + between-model prediction-correlation heatmap (3 synthetic models).")
def _b():
    from mlframe.reporting.charts.model_comparison import compose_model_comparison_figure
    y, _ = _binary_separable(n=4000)
    # Three models of varying skill / correlation.
    logit = RNG.normal(1.4 * y - 0.7, 1.0)
    def sig(z):
        return 1.0 / (1.0 + np.exp(-z))
    m1 = sig(logit)
    m2 = sig(logit * 0.7 + RNG.normal(0.0, 0.5, size=len(y)))  # similar
    m3 = sig(RNG.normal(0.8 * y - 0.4, 1.3))  # weaker, less correlated
    from sklearn.metrics import roc_auc_score
    per_model = {
        "model_strong": {"y_true": y, "y_score": m1, "metrics": {"roc_auc": float(roc_auc_score(y, m1))}},
        "model_similar": {"y_true": y, "y_score": m2, "metrics": {"roc_auc": float(roc_auc_score(y, m2))}},
        "model_weak": {"y_true": y, "y_score": m3, "metrics": {"roc_auc": float(roc_auc_score(y, m3))}},
    }
    return compose_model_comparison_figure(per_model, task_type="binary", metric="roc_auc")


# ---------------------------------------------------------------------------
# SPLIT COMPARISON (cross-split overfitting)
# ---------------------------------------------------------------------------


@entry("split_comparison", "split_comparison",
       "Cross-split overfitting view for ONE model: grouped headline-metric bars per train/val/test + delta table with "
       "a RED traffic-light verdict. Synthetic memorizes train (AUC ~0.99) but barely beats chance on test (AUC ~0.70).")
def _b():
    from mlframe.reporting.charts.split_comparison import compose_split_comparison_figure
    n = 4000
    # Train: score nearly equals the label (memorized) -> AUC ~0.99. Val/test: progressively weaker signal.
    y_tr = (RNG.random(n) < 0.5).astype(np.int8)
    s_tr = np.clip(y_tr + RNG.normal(0.0, 0.08, n), 0.0, 1.0)
    def _weak(sep, seed):
        r = np.random.default_rng(seed)
        y = (r.random(n) < 0.5).astype(np.int8)
        s = 1.0 / (1.0 + np.exp(-(sep * (y - 0.5) + r.normal(0.0, 1.0, n))))
        return {"y_true": y, "y_score": s}
    per_split = {
        "train": {"y_true": y_tr, "y_score": s_tr},
        "val": _weak(1.1, 11),
        "test": _weak(0.9, 22),
    }
    return compose_split_comparison_figure(per_split, task="classification", model_name="lgbm_overfit")


# ---------------------------------------------------------------------------
# TRAINING CURVE
# ---------------------------------------------------------------------------


@entry("training_curve", "training_curve", "Train/val metric vs iteration with the early-stopping marker + post-ES shading.")
def _b():
    from mlframe.reporting.charts.training_curve import compose_training_curve_figure
    n_iter = 200
    it = np.arange(n_iter)
    train_ll = 0.7 * np.exp(-it / 60.0) + 0.05
    val_ll = 0.7 * np.exp(-it / 50.0) + 0.12 + np.maximum(0, (it - 90)) * 0.0015  # overfits after ~90
    history = {"logloss": {"train": train_ll, "val": val_ll}}
    return compose_training_curve_figure(history, es_iteration=92, suptitle="Training curve (synthetic history)")


# ---------------------------------------------------------------------------
# LEARNING CURVE
# ---------------------------------------------------------------------------


@entry("learning_curve", "learning_curve", "Holdout score vs increasing train size (cheap sklearn estimator on synthetic).")
def _b():
    from sklearn.linear_model import Ridge
    from sklearn.metrics import get_scorer
    from mlframe.training.diagnostics.learning_curve import compute_learning_curve, learning_curve_panel
    n, K = 2500, 8
    X = RNG.normal(0.0, 1.0, size=(n, K))
    y = X @ RNG.normal(0.0, 1.0, size=K) + RNG.normal(0.0, 0.5, size=n)
    res = compute_learning_curve(
        lambda: Ridge(alpha=1.0), X, y,
        scorer=get_scorer("r2"), holdout=0.25, n_jobs=1, scorer_name="r2",
    )
    return learning_curve_panel(res, title="Learning curve")


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------


def _render_shap() -> Tuple[str, List[str]]:
    """Render the SHAP panels (or a placeholder note PNG). Returns (status, [relative_paths])."""
    out_dir = GALLERY / "shap_panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import shap  # noqa: F401
    except Exception as e:
        return _placeholder_png(
            "shap_panels", "shap_unavailable",
            f"SHAP not installed ({type(e).__name__}); beeswarm + dependence skipped.",
        )
    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from mlframe.reporting.charts.shap_panels import shap_summary_and_dependence
        n, K = 2000, 6
        names = [f"f{i}" for i in range(K)]
        X = RNG.normal(0.0, 1.0, size=(n, K))
        logit = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.8 * X[:, 2]
        y = (1.0 / (1.0 + np.exp(-logit)) > RNG.random(n)).astype(int)
        Xdf = pd.DataFrame(X, columns=names)
        model = RandomForestClassifier(n_estimators=60, max_depth=6, random_state=0, n_jobs=1).fit(Xdf, y)
        base = out_dir / "shap"
        res = shap_summary_and_dependence(
            model, Xdf, plot_file=str(base) + ".png", plot_outputs="matplotlib[png]", top_k=4,
        )
        rels = [str(Path(p).relative_to(GALLERY)).replace(os.sep, "/") for p in res.paths]
        if not rels:
            return _placeholder_png("shap_panels", "shap_skipped", f"SHAP produced no figures: {res.skipped}")
        return "ok", rels
    except Exception:
        ERRORS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ERRORS_LOG.open("a", encoding="ascii", errors="ignore") as f:
            f.write("=== shap_panels/shap ===\n")
            f.write(traceback.format_exc() + "\n")
        return _placeholder_png("shap_panels", "shap_error", "SHAP render raised; see _errors.log.")


def _render_shap_interactions() -> Tuple[str, List[str]]:
    """Render the SHAP feature-pair interaction summary (top-pairs bar + heatmap) on a planted f0*f1 interaction."""
    out_dir = GALLERY / "shap_interactions"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import shap  # noqa: F401
    except Exception as e:
        return _placeholder_png("shap_interactions", "shap_unavailable", f"SHAP not installed ({type(e).__name__}); interaction summary skipped.")
    try:
        import pandas as pd
        from sklearn.ensemble import GradientBoostingClassifier
        from mlframe.reporting.charts.shap_interactions import shap_interaction_summary
        n, K = 2000, 6
        names = [f"f{i}" for i in range(K)]
        X = RNG.normal(0.0, 1.0, size=(n, K))
        logit = 2.5 * (X[:, 0] * X[:, 1]) + 0.2 * X[:, 2]
        y = (1.0 / (1.0 + np.exp(-logit)) > RNG.random(n)).astype(int)
        Xdf = pd.DataFrame(X, columns=names)
        model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0).fit(Xdf, y)
        base = out_dir / "shap_interactions"
        res = shap_interaction_summary(model, Xdf, plot_file=str(base) + ".png", plot_outputs="matplotlib[png]", top_pairs=10)
        rels = [str(Path(p).relative_to(GALLERY)).replace(os.sep, "/") for p in res.paths]
        if not rels:
            return _placeholder_png("shap_interactions", "shap_int_skipped", f"SHAP interactions produced no figures: {res.skipped}")
        return "ok", rels
    except Exception:
        ERRORS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ERRORS_LOG.open("a", encoding="ascii", errors="ignore") as f:
            f.write("=== shap_interactions/shap_interactions ===\n")
            f.write(traceback.format_exc() + "\n")
        return _placeholder_png("shap_interactions", "shap_int_error", "SHAP interaction render raised; see _errors.log.")


def _placeholder_png(category: str, name: str, message: str) -> Tuple[str, List[str]]:
    """Write a simple text-note PNG so the gallery slot is never empty."""
    import matplotlib.pyplot as plt
    out_dir = GALLERY / category
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=110)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=11)
    rel = f"{category}/{name}.png"
    fig.savefig(str(GALLERY / rel), bbox_inches="tight")
    plt.close(fig)
    return "placeholder", [rel]


# ---------------------------------------------------------------------------
# TEMPORAL
# ---------------------------------------------------------------------------


@entry("temporal", "target_acf_pacf", "Target ACF + PACF by lag with Bartlett white-noise bounds (autocorrelated synthetic).")
def _b():
    from mlframe.reporting.charts.temporal import compose_target_acf_figure
    n = 3000
    # AR(1) target so ACF decays and PACF cuts off at lag 1.
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * y[t - 1] + RNG.normal(0.0, 1.0)
    return compose_target_acf_figure(y, suptitle="Target serial structure (AR(1))")


@entry("temporal", "target_temporal_audit", "Target-rate-over-time audit: kept bins, sparse bins, segment means, change-points.")
def _b():
    from types import SimpleNamespace
    from mlframe.reporting.charts.temporal import build_temporal_audit_spec
    import pandas as pd
    # Build a duck-typed TemporalAuditResult: a binary target rate with a regime shift mid-series.
    n_bins = 40
    starts = pd.date_range("2025-01-01", periods=n_bins, freq="D")
    rates = np.concatenate([RNG.normal(0.3, 0.02, 20), RNG.normal(0.55, 0.02, 20)])
    bins = []
    for i in range(n_bins):
        kept = not (5 <= i <= 6)  # mark a couple sparse/dropped bins
        bins.append(SimpleNamespace(bin_start=starts[i], target_rate=float(rates[i]), kept=kept))
    kept_idx = [i for i in range(n_bins) if bins[i].kept]
    segments = [
        {"start_idx": 0, "end_idx": len(kept_idx) // 2, "mean_rate": 0.3},
        {"start_idx": len(kept_idx) // 2, "end_idx": len(kept_idx), "mean_rate": 0.55},
    ]
    audit = SimpleNamespace(
        bins=bins, segments=segments, change_point_indices=[len(kept_idx) // 2],
        target_name="conversion", granularity="D", target_type="binary", timestamp_col="Time",
    )
    return build_temporal_audit_spec(audit)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    GALLERY.mkdir(parents=True, exist_ok=True)
    if ERRORS_LOG.exists():
        ERRORS_LOG.unlink()

    rendered: List[Tuple[str, str, str, str]] = []  # (category, name, description, rel_path)
    failed: List[Tuple[str, str, str]] = []         # (category, name, reason)

    total = len(ENTRIES) + 1  # + shap special
    for i, (category, name, description, builder) in enumerate(ENTRIES, start=1):
        heartbeat(f"render {i}/{total}: {category}/{name}")
        try:
            spec = builder()
            rel = _save_spec(spec, category, name)
            rendered.append((category, name, description, rel))
            print(f"[OK]   {category}/{name}")
        except Exception:
            tb = traceback.format_exc()
            ERRORS_LOG.parent.mkdir(parents=True, exist_ok=True)
            with ERRORS_LOG.open("a", encoding="ascii", errors="ignore") as f:
                f.write(f"=== {category}/{name} ===\n{tb}\n")
            failed.append((category, name, tb.strip().splitlines()[-1] if tb.strip() else "unknown"))
            print(f"[FAIL] {category}/{name}")

    # SHAP (special: shap writes its own figures / placeholder).
    heartbeat(f"render {total}/{total}: shap_panels")
    shap_status, shap_rels = _render_shap()
    shap_desc = "SHAP beeswarm + dependence plots for a small tree model."
    for rel in shap_rels:
        nm = Path(rel).stem
        rendered.append(("shap_panels", nm, shap_desc, rel))
    print(f"[{shap_status.upper()}] shap_panels ({len(shap_rels)} file(s))")

    heartbeat(f"render {total}/{total}: shap_interactions")
    si_status, si_rels = _render_shap_interactions()
    si_desc = "SHAP feature-pair interaction summary: top-pairs bar + interaction-strength heatmap."
    for rel in si_rels:
        nm = Path(rel).stem
        rendered.append(("shap_interactions", nm, si_desc, rel))
    print(f"[{si_status.upper()}] shap_interactions ({len(si_rels)} file(s))")

    _write_index(rendered)
    heartbeat(f"done: rendered={len(rendered)} failed={len(failed)}")

    print("\n==== GALLERY SUMMARY ====")
    print(f"rendered: {len(rendered)}")
    print(f"failed:   {len(failed)}")
    for category, name, reason in failed:
        print(f"  FAIL {category}/{name}: {reason}")
    print(f"gallery dir: {GALLERY}")
    return 0


def _write_index(rendered: List[Tuple[str, str, str, str]]) -> None:
    """Write docs/gallery/index.md grouping the PNGs by category."""
    from collections import OrderedDict
    by_cat: "OrderedDict[str, List[Tuple[str, str, str]]]" = OrderedDict()
    for category, name, description, rel in rendered:
        by_cat.setdefault(category, []).append((name, description, rel))

    lines: List[str] = []
    lines.append("# mlframe reporting chart gallery")
    lines.append("")
    lines.append("Every chart / diagnostic in the mlframe reporting subsystem, rendered to PNG on synthetic")
    lines.append("data chosen to make each chart meaningful. Regenerate with `python scripts/render_gallery.py`.")
    lines.append("")
    lines.append(f"Total images: {len(rendered)} across {len(by_cat)} categories.")
    lines.append("")
    lines.append("## Contents")
    lines.append("")
    for category in by_cat:
        anchor = category.replace("_", "-")
        lines.append(f"- [{category}](#{anchor})")
    lines.append("")
    for category, items in by_cat.items():
        lines.append(f"## {category}")
        lines.append("")
        for name, description, rel in items:
            lines.append(f"### {name}")
            lines.append("")
            lines.append(description)
            lines.append("")
            lines.append(f"![{name}]({rel})")
            lines.append("")
    (GALLERY / "index.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
