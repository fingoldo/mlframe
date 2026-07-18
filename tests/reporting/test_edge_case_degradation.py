"""Graceful-degradation harness for every reporting composer / builder.

Each composer is fed a battery of degenerate inputs (single-class y, zero-variance scores, tiny n, all-NaN/inf,
mixed-NaN, empty arrays, huge K, single constant bin, absent class) and must, for each: NOT raise, return a
``FigureSpec`` (or the documented ``None``/result contract), AND render through BOTH MatplotlibRenderer and
PlotlyRenderer without error. A degenerate panel must be an honest ``AnnotationPanelSpec`` placeholder, never a
misleading/empty chart and never a divide-by-zero / index-OOB crash.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlframe.reporting.charts import (
    adversarial_validation,
    build_calibration_drift_spec,
    build_calibration_spec,
    build_decision_curve_spec,
    calibration_drift,
    compose_binary_figure,
    compose_ltr_figure,
    compose_model_comparison_figure,
    compose_multiclass_figure,
    compose_multilabel_figure,
    compose_pdp_figure,
    compose_quantile_figure,
    compose_regression_figure,
    compose_training_curve_figure,
    error_bias_per_feature,
    find_weak_slices,
    metric_over_time,
    psi_heatmap,
    residual_vs_time,
    target_dist_overlay,
    weak_segment_heatmap,
)
from mlframe.reporting.charts.calibration_drift import CalibrationDriftResult
from mlframe.reporting.charts.slice_finder import SliceFinderResult
from mlframe.reporting.charts.error_analysis import ErrorBiasResult
from mlframe.reporting.renderers.base import get_renderer
from mlframe.reporting.spec import FigureSpec

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- render helper


_MPL = get_renderer("matplotlib")
_PLOTLY = get_renderer("plotly")


def _render_both(fig: FigureSpec) -> None:
    """Render the spec on both backends; close mpl handles so the Agg buffer does not leak across cases."""
    assert isinstance(fig, FigureSpec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # degenerate inputs legitimately emit numpy/sklearn RuntimeWarnings
        mpl_fig = _MPL.render(fig)
        try:
            plotly_fig = _PLOTLY.render(fig)
            assert plotly_fig is not None
        finally:
            plt.close(mpl_fig)


def _to_result_fig(obj):
    """Pull the FigureSpec out of a builder result wrapper (SliceFinderResult / ErrorBiasResult / ...)."""
    if isinstance(obj, FigureSpec):
        return obj
    for attr in ("figure", "fig", "spec"):
        f = getattr(obj, attr, None)
        if isinstance(f, FigureSpec):
            return f
    raise AssertionError(f"no FigureSpec on result {type(obj).__name__}")


# --------------------------------------------------------------------------- model stubs


class _LinearModel:
    """Groups tests for: LinearModel."""
    def __init__(self, w):
        """Helper: Init  ."""
        self.w = np.asarray(w, dtype=np.float64)

    def predict(self, X):
        """Predict."""
        return np.asarray(X, dtype=np.float64) @ self.w


# --------------------------------------------------------------------------- degenerate label generators


def _single_class(n=200, cls=1):
    """Helper: Single class."""
    y = np.full(n, cls, dtype=int)
    s = np.linspace(0.1, 0.9, n)
    return y, s


def _const_score(n=200):
    """Helper: Const score."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n)
    s = np.full(n, 0.5)
    return y, s


def _all_nan(n=200):
    """Helper: All nan."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n)
    s = np.full(n, np.nan)
    return y, s


def _all_inf(n=200):
    """Helper: All inf."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n)
    s = np.full(n, np.inf)
    return y, s


def _mixed_nan(n=200):
    """Helper: Mixed nan."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n)
    s = rng.random(n)
    s[: n // 2] = np.nan
    s[n // 2 : n // 2 + 5] = np.inf
    return y, s


def _empty():
    """Helper: Empty."""
    return np.array([], dtype=int), np.array([], dtype=float)


def _tiny(n):
    """Helper: Tiny."""
    rng = np.random.default_rng(n)
    y = rng.integers(0, 2, n)
    if n >= 2:  # ensure at least one of each where possible
        y[0] = 0
        y[-1] = 1
    s = rng.random(n)
    return y, s


# --------------------------------------------------------------------------- BINARY


_BINARY_CASES = {
    "single_class_1": _single_class(cls=1),
    "single_class_0": _single_class(cls=0),
    "const_score": _const_score(),
    "all_nan": _all_nan(),
    "all_inf": _all_inf(),
    "mixed_nan": _mixed_nan(),
    "empty": _empty(),
    "n1": _tiny(1),
    "n5": _tiny(5),
    "n15": _tiny(15),
}

_BINARY_TEMPLATE = "ROC PR SCORE_DIST KS THRESHOLD GAIN PIT"


@pytest.mark.parametrize("case", sorted(_BINARY_CASES))
def test_binary_degenerate(case):
    """Binary degenerate."""
    y, s = _BINARY_CASES[case]
    fig = compose_binary_figure(y, s, panels_template=_BINARY_TEMPLATE)
    _render_both(fig)


# --------------------------------------------------------------------------- MULTICLASS


def _mc_case(name, K=3, n=300):
    """Helper: Mc case."""
    rng = np.random.default_rng(1)
    if name == "single_class":
        y = np.zeros(n, dtype=int)
    elif name == "absent_class":  # class K-1 present in proba columns but never in y_true
        y = rng.integers(0, K - 1, n)
    elif name == "empty":
        return np.array([], dtype=int), np.zeros((0, K)), list(range(K))
    elif name == "n1":
        y = np.array([0])
        n = 1
    elif name == "n5":
        y = rng.integers(0, K, 5)
        n = 5
    else:
        y = rng.integers(0, K, n)
    if name == "const_proba":
        P = np.full((n, K), 1.0 / K)
    elif name == "all_nan":
        P = np.full((n, K), np.nan)
    elif name == "huge_K":
        K = 50
        y = rng.integers(0, K, n)
        P = rng.random((n, K))
        P /= P.sum(1, keepdims=True)
        return y, P, list(range(K))
    else:
        P = rng.random((n, K))
        P /= np.maximum(P.sum(1, keepdims=True), 1e-12)
    return y, P, list(range(K))


_MC_CASES = ["single_class", "absent_class", "const_proba", "all_nan", "empty", "n1", "n5", "huge_K"]
_MC_TEMPLATE = "CONFUSION CONFUSED_PAIRS PR_F1 ROC PR_CURVES CALIB_GRID PROB_DIST TOP_K_ACC"


@pytest.mark.parametrize("case", _MC_CASES)
def test_multiclass_degenerate(case):
    """Multiclass degenerate."""
    y, P, classes = _mc_case(case)
    fig = compose_multiclass_figure(y, P, classes, panels_template=_MC_TEMPLATE)
    _render_both(fig)


# --------------------------------------------------------------------------- MULTILABEL


def _ml_case(name, K=4, n=300):
    """Helper: Ml case."""
    rng = np.random.default_rng(2)
    if name == "empty":
        return np.zeros((0, K), dtype=int), np.zeros((0, K))
    if name == "n1":
        n = 1
    elif name == "n5":
        n = 5
    if name == "huge_K":
        K = 50
    Y = rng.integers(0, 2, (n, K))
    P = rng.random((n, K))
    if name == "single_class":  # every label all-zero (no positives anywhere)
        Y = np.zeros((n, K), dtype=int)
    elif name == "all_one":
        Y = np.ones((n, K), dtype=int)
    elif name == "const_proba":
        P = np.full((n, K), 0.5)
    elif name == "all_nan":
        P = np.full((n, K), np.nan)
    return Y, P


_ML_CASES = ["single_class", "all_one", "const_proba", "all_nan", "empty", "n1", "n5", "huge_K"]
_ML_TEMPLATE = "PR_F1 ROC CALIB_GRID COOCCURRENCE CARDINALITY JACCARD_DIST HAMMING_DIST THRESHOLD_SWEEP"


@pytest.mark.parametrize("case", _ML_CASES)
def test_multilabel_degenerate(case):
    """Multilabel degenerate."""
    Y, P = _ml_case(case)
    K = Y.shape[1]
    fig = compose_multilabel_figure(Y, P, list(range(K)), panels_template=_ML_TEMPLATE)
    _render_both(fig)


# --------------------------------------------------------------------------- LTR


def _ltr_case(name):
    """Helper: Ltr case."""
    rng = np.random.default_rng(3)
    if name == "empty":
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=int)
    if name == "single_group_single_doc":
        return np.array([1]), np.array([0.5]), np.array([0])
    if name == "single_grade":  # all relevance equal -> NDCG undefined
        n = 100
        return np.ones(n, dtype=int), rng.random(n), rng.integers(0, 10, n)
    if name == "const_score":
        n = 100
        return rng.integers(0, 3, n), np.full(n, 0.5), rng.integers(0, 10, n)
    if name == "all_nan_score":
        n = 100
        return rng.integers(0, 3, n), np.full(n, np.nan), rng.integers(0, 10, n)
    if name == "n5":
        return rng.integers(0, 3, 5), rng.random(5), np.zeros(5, dtype=int)
    # normal-ish small
    n = 50
    return rng.integers(0, 4, n), rng.random(n), rng.integers(0, 5, n)


_LTR_CASES = ["empty", "single_group_single_doc", "single_grade", "const_score", "all_nan_score", "n5"]


_LTR_TEMPLATE = "NDCG_K NDCG_DIST NDCG_BY_QSIZE LIFT MRR_DIST SCORE_BY_REL TOP1_BY_QSIZE"


@pytest.mark.parametrize("case", _LTR_CASES)
def test_ltr_degenerate(case):
    """Ltr degenerate."""
    yt, ys, g = _ltr_case(case)
    fig = compose_ltr_figure(yt, ys, g, panels_template=_LTR_TEMPLATE)
    _render_both(fig)


# --------------------------------------------------------------------------- QUANTILE


def _quant_case(name):
    """Helper: Quant case."""
    rng = np.random.default_rng(4)
    alphas = (0.1, 0.5, 0.9)
    K = len(alphas)
    if name == "empty":
        return np.array([], dtype=float), np.zeros((0, K)), alphas
    if name == "n1":
        return np.array([1.0]), rng.random((1, K)), alphas
    if name == "n5":
        return rng.random(5), np.sort(rng.random((5, K)), axis=1), alphas
    n = 200
    y = rng.standard_normal(n)
    P = np.sort(rng.standard_normal((n, K)), axis=1)
    if name == "const_pred":
        P = np.zeros((n, K))
    elif name == "const_y":
        y = np.full(n, 3.0)
    elif name == "all_nan":
        P = np.full((n, K), np.nan)
    elif name == "mixed_nan":
        y[: n // 2] = np.nan
    return y, P, alphas


_QUANT_CASES = ["empty", "n1", "n5", "const_pred", "const_y", "all_nan", "mixed_nan"]


_QUANT_TEMPLATE = "RELIABILITY COVERAGE PINBALL_BY_ALPHA INTERVAL_BAND WIDTH_DIST PIT_HIST QUANTILE_RELIABILITY PINBALL_DECOMP QUANTILE_CROSSING FAN_CHART"


@pytest.mark.parametrize("case", _QUANT_CASES)
def test_quantile_degenerate(case):
    """Quantile degenerate."""
    y, P, alphas = _quant_case(case)
    fig = compose_quantile_figure(y, P, alphas, panels_template=_QUANT_TEMPLATE)
    _render_both(fig)


# --------------------------------------------------------------------------- REGRESSION


def _reg_case(name):
    """Helper: Reg case."""
    rng = np.random.default_rng(5)
    if name == "empty":
        return np.array([], dtype=float), np.array([], dtype=float)
    if name == "n1":
        return np.array([1.0]), np.array([1.2])
    if name == "n5":
        return rng.standard_normal(5), rng.standard_normal(5)
    n = 300
    y = rng.standard_normal(n)
    p = y + 0.1 * rng.standard_normal(n)
    if name == "const_pred":
        p = np.zeros(n)
    elif name == "const_y":
        y = np.full(n, 2.0)
    elif name == "all_nan":
        p = np.full(n, np.nan)
    elif name == "all_inf":
        p = np.full(n, np.inf)
    elif name == "mixed_nan":
        p[: n // 2] = np.nan
    return y, p


_REG_CASES = ["empty", "n1", "n5", "const_pred", "const_y", "all_nan", "all_inf", "mixed_nan"]


_REG_TEMPLATE = "SCATTER RESID_HIST RESID_VS_PRED ERR_BY_DECILE WORM RESID_ACF"


@pytest.mark.parametrize("case", _REG_CASES)
def test_regression_degenerate(case):
    """Regression degenerate."""
    y, p = _reg_case(case)
    fig = compose_regression_figure(y, p, panels_template=_REG_TEMPLATE)
    _render_both(fig)


# --------------------------------------------------------------------------- DECISION CURVE


@pytest.mark.parametrize("case", sorted(_BINARY_CASES))
def test_decision_curve_degenerate(case):
    """Decision curve degenerate."""
    y, s = _BINARY_CASES[case]
    res = build_decision_curve_spec(y, s)
    _render_both(_to_result_fig(res))


# --------------------------------------------------------------------------- CALIBRATION (build_calibration_spec)


def _calib_case(name):
    """Helper: Calib case."""
    if name == "empty":
        return np.array([]), np.array([]), np.array([])
    if name == "single_bin":
        return np.array([0.5]), np.array([0.5]), np.array([100])
    if name == "const_bins":
        return np.full(5, 0.5), np.full(5, 0.5), np.full(5, 0)
    if name == "all_nan":
        return np.full(5, np.nan), np.full(5, np.nan), np.full(5, 10)
    fp = np.linspace(0.05, 0.95, 10)
    return fp, fp, np.full(10, 50)


_CALIB_CASES = ["empty", "single_bin", "const_bins", "all_nan", "normal"]


@pytest.mark.parametrize("case", _CALIB_CASES)
def test_build_calibration_spec_degenerate(case):
    """Build calibration spec degenerate."""
    fp, ft, hits = _calib_case(case)
    fig = build_calibration_spec(fp, ft, hits)
    _render_both(fig)


# --------------------------------------------------------------------------- CALIBRATION DRIFT


def _calib_drift_case(name):
    """Helper: Calib drift case."""
    rng = np.random.default_rng(6)
    if name == "empty":
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype="datetime64[D]")
    n = 200
    ts = np.arange(n).astype("datetime64[D]")
    if name == "single_class":
        y = np.ones(n, dtype=int)
    else:
        y = rng.integers(0, 2, n)
    s = rng.random(n)
    if name == "const_score":
        s = np.full(n, 0.5)
    elif name == "all_nan":
        s = np.full(n, np.nan)
    elif name == "n5":
        n = 5
        ts = np.arange(n).astype("datetime64[D]")
        y = rng.integers(0, 2, n)
        s = rng.random(n)
    return y, s, ts


_CALIB_DRIFT_CASES = ["empty", "single_class", "const_score", "all_nan", "n5", "normal"]


@pytest.mark.parametrize("case", _CALIB_DRIFT_CASES)
def test_calibration_drift_degenerate(case):
    """Calibration drift degenerate."""
    y, s, ts = _calib_drift_case(case)
    res = calibration_drift(y, s, ts)
    assert isinstance(res, CalibrationDriftResult)
    fig = build_calibration_drift_spec(res)
    _render_both(fig)


# --------------------------------------------------------------------------- TRAINING CURVE


def _tc_case(name):
    """Helper: Tc case."""
    if name == "empty":
        return {}
    if name == "single_point":
        return {"loss": {"train": [0.5], "val": [0.6]}}
    if name == "all_nan":
        return {"loss": {"train": [np.nan, np.nan], "val": [np.nan, np.nan]}}
    if name == "mismatched_len":
        return {"loss": {"train": [0.5, 0.4, 0.3], "val": [0.6]}}
    return {"loss": {"train": [0.5, 0.4, 0.3], "val": [0.6, 0.55, 0.5]}}


_TC_CASES = ["empty", "single_point", "all_nan", "mismatched_len", "normal"]


@pytest.mark.parametrize("case", _TC_CASES)
def test_training_curve_degenerate(case):
    """Training curve degenerate."""
    hist = _tc_case(case)
    fig = compose_training_curve_figure(hist)
    _render_both(fig)


# --------------------------------------------------------------------------- MODEL COMPARISON


def _mc_compare_case(name):
    """Helper: Mc compare case."""
    rng = np.random.default_rng(7)

    def entry(y, s, **m):
        """Entry."""
        return {"y_true": np.asarray(y), "y_score": np.asarray(s), "metrics": dict(m)}

    if name == "empty":
        return {}, "binary"
    if name == "single_model_single_class":
        y = np.ones(100, dtype=int)
        return {"A": entry(y, rng.random(100), roc_auc=float("nan"))}, "binary"
    if name == "const_score":
        y = rng.integers(0, 2, 100)
        return {"A": entry(y, np.full(100, 0.5), roc_auc=0.5), "B": entry(y, np.full(100, 0.5), roc_auc=0.5)}, "binary"
    if name == "no_metrics":
        y = rng.integers(0, 2, 100)
        return {"A": entry(y, rng.random(100)), "B": entry(y, rng.random(100))}, "binary"
    if name == "all_nan":
        y = rng.integers(0, 2, 100)
        return {"A": entry(y, np.full(100, np.nan), roc_auc=float("nan"))}, "binary"
    if name == "regression":
        y = rng.standard_normal(100)
        return {"A": entry(y, np.zeros(100), r2=0.0)}, "regression"
    y = rng.integers(0, 2, 100)
    return {"A": entry(y, rng.random(100), roc_auc=0.7), "B": entry(y, rng.random(100), roc_auc=0.6)}, "binary"


_MC_COMPARE_CASES = ["empty", "single_model_single_class", "const_score", "no_metrics", "all_nan", "regression", "normal"]


@pytest.mark.parametrize("case", _MC_COMPARE_CASES)
def test_model_comparison_degenerate(case):
    """Model comparison degenerate."""
    per_model, task = _mc_compare_case(case)
    fig = compose_model_comparison_figure(per_model, task)
    _render_both(fig)


# --------------------------------------------------------------------------- PDP / ICE


def _pdp_case(name):
    """Helper: Pdp case."""
    rng = np.random.default_rng(8)
    model = _LinearModel([1.0, 0.0])
    if name == "no_features":
        return model, rng.normal(size=(50, 2)), []
    if name == "n1":
        return model, rng.normal(size=(1, 2)), [0]
    if name == "n5":
        return model, rng.normal(size=(5, 2)), [0, 1]
    if name == "const_feature":
        X = rng.normal(size=(100, 2))
        X[:, 0] = 3.0
        return model, X, [0]
    if name == "nan_feature":
        X = rng.normal(size=(100, 2))
        X[:, 0] = np.nan
        return model, X, [0]
    return model, rng.normal(size=(200, 2)), [0, 1]


_PDP_CASES = ["no_features", "n1", "n5", "const_feature", "nan_feature", "normal"]


@pytest.mark.parametrize("case", _PDP_CASES)
def test_pdp_degenerate(case):
    """Pdp degenerate."""
    model, X, feats = _pdp_case(case)
    fig = compose_pdp_figure(model, X, feats)
    _render_both(fig)


# --------------------------------------------------------------------------- SLICE FINDER


def _slice_case(name):
    """Helper: Slice case."""
    rng = np.random.default_rng(9)
    if name == "empty":
        return np.zeros((0, 3)), np.array([]), np.array([])
    if name == "n1":
        return rng.normal(size=(1, 3)), rng.standard_normal(1), rng.standard_normal(1)
    if name == "n5":
        return rng.normal(size=(5, 3)), rng.standard_normal(5), rng.standard_normal(5)
    if name == "no_features":
        return np.zeros((100, 0)), rng.standard_normal(100), rng.standard_normal(100)
    if name == "const_features":
        X = np.full((100, 3), 1.0)
        return X, rng.standard_normal(100), rng.standard_normal(100)
    if name == "all_nan_err":
        X = rng.normal(size=(100, 3))
        return X, np.full(100, np.nan), np.full(100, np.nan)
    X = rng.normal(size=(200, 3))
    y = X[:, 0] + rng.standard_normal(200)
    return X, y, X[:, 0]


_SLICE_CASES = ["empty", "n1", "n5", "no_features", "const_features", "all_nan_err", "normal"]


@pytest.mark.parametrize("case", _SLICE_CASES)
def test_slice_finder_degenerate(case):
    """Slice finder degenerate."""
    X, yt, yp = _slice_case(case)
    res = find_weak_slices(X, yt, yp, task="regression")
    assert isinstance(res, SliceFinderResult)
    _render_both(_to_result_fig(res))


# --------------------------------------------------------------------------- WEAK SEGMENT HEATMAP


@pytest.mark.parametrize("case", _SLICE_CASES)
def test_weak_segment_heatmap_degenerate(case):
    """Weak segment heatmap degenerate."""
    X, yt, yp = _slice_case(case)
    res = weak_segment_heatmap(X, yt, yp, task="regression")
    _render_both(_to_result_fig(res))


# --------------------------------------------------------------------------- ERROR BIAS PER FEATURE


@pytest.mark.parametrize("case", _SLICE_CASES)
def test_error_bias_per_feature_degenerate(case):
    """Error bias per feature degenerate."""
    X, yt, yp = _slice_case(case)
    res = error_bias_per_feature(X, yt, yp)
    assert isinstance(res, ErrorBiasResult)
    _render_both(_to_result_fig(res))


# --------------------------------------------------------------------------- TARGET DIST OVERLAY


def _tdo_case(name):
    """Helper: Tdo case."""
    rng = np.random.default_rng(10)
    if name == "empty_split":
        return {"train": np.array([]), "test": np.array([])}, "regression"
    if name == "single_split":
        return {"train": rng.standard_normal(100)}, "regression"
    if name == "const":
        return {"train": np.full(100, 1.0), "test": np.full(100, 1.0)}, "regression"
    if name == "all_nan":
        return {"train": np.full(100, np.nan), "test": np.full(100, np.nan)}, "regression"
    if name == "classification_single_class":
        return {"train": np.ones(100, dtype=int), "test": np.ones(100, dtype=int)}, "classification"
    return {"train": rng.standard_normal(100), "test": rng.standard_normal(100) + 1}, "regression"


_TDO_CASES = ["empty_split", "single_split", "const", "all_nan", "classification_single_class", "normal"]


@pytest.mark.parametrize("case", _TDO_CASES)
def test_target_dist_overlay_degenerate(case):
    """Target dist overlay degenerate."""
    by_split, task = _tdo_case(case)
    fig = target_dist_overlay(by_split, task=task)
    _render_both(fig)


# --------------------------------------------------------------------------- DRIFT: psi_heatmap


def _frame_case(name, p=3):
    """Helper: Frame case."""
    import pandas as pd

    rng = np.random.default_rng(11)
    if name == "empty":
        return pd.DataFrame(np.zeros((0, p))), np.array([], dtype="datetime64[D]")
    if name == "n5":
        n = 5
    elif name == "n1":
        n = 1
    else:
        n = 300
    ts = np.arange(n).astype("datetime64[D]")
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    if name == "const_features":
        X.iloc[:, :] = 1.0
    elif name == "all_nan":
        X.iloc[:, :] = np.nan
    return X, ts


_FRAME_CASES = ["empty", "n1", "n5", "const_features", "all_nan", "normal"]


@pytest.mark.parametrize("case", _FRAME_CASES)
def test_psi_heatmap_degenerate(case):
    """Psi heatmap degenerate."""
    X, ts = _frame_case(case)
    fig = psi_heatmap(X, ts, n_time_buckets=5)
    _render_both(fig)


# --------------------------------------------------------------------------- DRIFT: residual_vs_time


def _rvt_case(name):
    """Helper: Rvt case."""
    rng = np.random.default_rng(12)
    if name == "empty":
        return np.array([]), np.array([]), np.array([], dtype="datetime64[D]")
    if name == "n1":
        return np.array([1.0]), np.array([1.0]), np.arange(1).astype("datetime64[D]")
    if name == "n5":
        return rng.standard_normal(5), rng.standard_normal(5), np.arange(5).astype("datetime64[D]")
    n = 300
    ts = np.arange(n).astype("datetime64[D]")
    y = rng.standard_normal(n)
    p = y.copy()
    if name == "all_nan":
        p = np.full(n, np.nan)
    elif name == "const":
        y = np.full(n, 1.0)
        p = np.full(n, 1.0)
    return y, p, ts


_RVT_CASES = ["empty", "n1", "n5", "all_nan", "const", "normal"]


@pytest.mark.parametrize("case", _RVT_CASES)
def test_residual_vs_time_degenerate(case):
    """Residual vs time degenerate."""
    y, p, ts = _rvt_case(case)
    fig = residual_vs_time(y, p, ts)
    _render_both(fig)


# --------------------------------------------------------------------------- DRIFT: metric_over_time


def _mot_case(name):
    """Helper: Not case."""
    rng = np.random.default_rng(13)
    if name == "empty":
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype="datetime64[D]")
    if name == "n5":
        n = 5
        return rng.integers(0, 2, n), rng.random(n), np.arange(n).astype("datetime64[s]")
    n = 500
    ts = (np.arange(n) * 3600).astype("datetime64[s]")
    y = rng.integers(0, 2, n)
    p = rng.random(n)
    if name == "single_class":
        y = np.ones(n, dtype=int)
    elif name == "const_score":
        p = np.full(n, 0.5)
    elif name == "all_nan":
        p = np.full(n, np.nan)
    elif name == "no_buckets":  # min_samples too high -> zero qualifying buckets
        return y, p, ts
    return y, p, ts


_MOT_CASES = ["empty", "n5", "single_class", "const_score", "all_nan", "no_buckets", "normal"]


@pytest.mark.parametrize("case", _MOT_CASES)
def test_metric_over_time_degenerate(case):
    """Metric over time degenerate."""
    y, p, ts = _mot_case(case)
    min_samples = 10_000_000 if case == "no_buckets" else 10
    fig = metric_over_time(y, p, ts, metric="roc_auc", freq="D", min_samples=min_samples)
    _render_both(fig)


# --------------------------------------------------------------------------- DRIFT: adversarial_validation


def _adv_case(name):
    """Helper: Adv case."""
    import pandas as pd

    rng = np.random.default_rng(14)
    cols = ["a", "b", "c"]
    if name == "empty":
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)
    if name == "n5":
        return (pd.DataFrame(rng.normal(size=(5, 3)), columns=cols), pd.DataFrame(rng.normal(size=(5, 3)), columns=cols))
    if name == "identical":  # train and test indistinguishable -> AUC ~0.5
        base = pd.DataFrame(rng.normal(size=(200, 3)), columns=cols)
        return base.copy(), base.copy()
    if name == "const_features":
        a = pd.DataFrame(np.ones((200, 3)), columns=cols)
        return a, a.copy()
    if name == "all_nan":
        a = pd.DataFrame(np.full((200, 3), np.nan), columns=cols)
        return a, a.copy()
    return (pd.DataFrame(rng.normal(size=(200, 3)), columns=cols), pd.DataFrame(rng.normal(size=(200, 3)) + 2, columns=cols))


_ADV_CASES = ["empty", "n5", "identical", "const_features", "all_nan", "normal"]


@pytest.mark.parametrize("case", _ADV_CASES)
def test_adversarial_validation_degenerate(case):
    """Adversarial validation degenerate."""
    pytest.importorskip("lightgbm")
    Xa, Xb = _adv_case(case)
    fig = adversarial_validation(Xa, Xb, n_splits=2, top_features=3)
    _render_both(fig)
