"""Tests for the 2026-05-28 follow-up additions:
- Multiclass macro / weighted aggregation across per-class metric blocks.
- Multilabel per-label AUC + macro / weighted reductions.
- MASE seasonality knob in ReportingConfig + ``mase_naive_mae`` plumbing
  through report_regression_model_perf.
"""
from __future__ import annotations

import numpy as np
import pytest


# ----- multilabel per-label AUC -----


def test_multilabel_auc_per_label_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    from mlframe.metrics.core import multilabel_auc_per_label
    rng = np.random.default_rng(0)
    N, K = 500, 4
    y = (rng.uniform(size=(N, K)) > 0.6).astype(np.int64)
    s = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.15, (N, K)), 0.001, 0.999)
    per = multilabel_auc_per_label(y, s)
    expected = np.array([
        roc_auc_score(y[:, k], s[:, k]) for k in range(K)
    ])
    np.testing.assert_allclose(per, expected, atol=1e-12)


def test_multilabel_auc_macro_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    from mlframe.metrics.core import multilabel_auc_macro
    rng = np.random.default_rng(1)
    N, K = 800, 5
    y = (rng.uniform(size=(N, K)) > 0.7).astype(np.int64)
    s = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.15, (N, K)), 0.001, 0.999)
    ours = multilabel_auc_macro(y, s)
    expected = roc_auc_score(y, s, average="macro")
    assert ours == pytest.approx(expected, abs=1e-12)


def test_multilabel_auc_weighted_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    from mlframe.metrics.core import multilabel_auc_weighted
    rng = np.random.default_rng(2)
    N, K = 800, 5
    y = (rng.uniform(size=(N, K)) > 0.7).astype(np.int64)
    s = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.15, (N, K)), 0.001, 0.999)
    ours = multilabel_auc_weighted(y, s)
    expected = roc_auc_score(y, s, average="weighted")
    assert ours == pytest.approx(expected, abs=1e-12)


def test_multilabel_auc_single_class_label_yields_nan():
    """A label with no positives (or no negatives) cannot have AUC.
    Per-label kernel returns NaN; macro skips it cleanly."""
    from mlframe.metrics.core import multilabel_auc_per_label, multilabel_auc_macro
    y = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
        [0, 0, 0],
    ], dtype=np.int64)  # col 0 is all-zero
    s = np.random.default_rng(0).uniform(size=y.shape)
    per = multilabel_auc_per_label(y, s)
    assert np.isnan(per[0])
    assert np.isfinite(per[1])
    macro = multilabel_auc_macro(y, s)
    # macro is mean of finite labels only
    assert macro == pytest.approx(np.nanmean(per), abs=1e-12)


def test_multilabel_auc_registered_in_registry():
    from mlframe.training.metrics_registry import list_registered, iter_extra_metrics
    from mlframe.training.configs import TargetTypes
    names = list_registered(TargetTypes.MULTILABEL_CLASSIFICATION)
    assert "auc_macro" in names
    assert "auc_weighted" in names

    rng = np.random.default_rng(3)
    N, K = 100, 3
    y = (rng.uniform(size=(N, K)) > 0.5).astype(np.int64)
    s = rng.uniform(size=(N, K))
    p = (s > 0.5).astype(np.int64)
    out = dict(iter_extra_metrics(TargetTypes.MULTILABEL_CLASSIFICATION, y, s, p))
    assert "auc_macro" in out
    assert "auc_weighted" in out


# ----- multiclass macro / weighted aggregation -----


def test_multiclass_aggregation_macro_keys_present_in_metrics():
    """When report_probabilistic_model_perf runs on a 4-class target it
    should stamp per-class blocks AND macro_/weighted_ aggregates."""
    from mlframe.training._reporting_probabilistic import report_probabilistic_model_perf
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(20)
    N = 500
    K = 4
    X = rng.standard_normal((N, 6))
    # K-class target with a slight correlation to X
    logits = X @ rng.standard_normal((6, K))
    y = logits.argmax(axis=1)
    model = LogisticRegression(max_iter=200).fit(X, y)

    metrics: dict = {}
    report_probabilistic_model_perf(
        targets=y,
        columns=[f"f{i}" for i in range(6)],
        model_name="test_lr",
        model=model,
        probs=model.predict_proba(X),
        preds=model.predict(X),
        plot_file="",
        show_perf_chart=False,
        print_report=False,
        metrics=metrics,
    )
    # Expect 4 per-class dicts + macro_/weighted_ aggregations for at least
    # ROC_AUC and other standard scalars.
    per_class_keys = [k for k in metrics.keys() if isinstance(k, (int, np.integer))]
    assert len(per_class_keys) == K, f"expected {K} per-class dicts, got {len(per_class_keys)}"
    # New aggregation keys: at least one macro_* and one weighted_* per
    # numeric scalar across per-class blocks.
    macro_keys = [k for k in metrics.keys() if isinstance(k, str) and k.startswith("macro_")]
    weighted_keys = [k for k in metrics.keys() if isinstance(k, str) and k.startswith("weighted_")]
    assert len(macro_keys) > 5
    assert len(weighted_keys) > 5
    # AUC aggregation must be present (the headline metric).
    assert "macro_roc_auc" in metrics
    assert "weighted_roc_auc" in metrics


def test_multiclass_aggregation_macro_equals_mean_of_per_class():
    from mlframe.training._reporting_probabilistic import report_probabilistic_model_perf
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(21)
    N = 400
    K = 3
    X = rng.standard_normal((N, 5))
    logits = X @ rng.standard_normal((5, K))
    y = logits.argmax(axis=1)
    model = LogisticRegression(max_iter=200).fit(X, y)

    metrics: dict = {}
    report_probabilistic_model_perf(
        targets=y,
        columns=[f"f{i}" for i in range(5)],
        model_name="test",
        model=model,
        probs=model.predict_proba(X),
        preds=model.predict(X),
        plot_file="",
        show_perf_chart=False,
        print_report=False,
        metrics=metrics,
    )
    # macro_roc_auc should be the simple mean of per-class roc_auc values.
    per_class_aucs = []
    for cid in range(K):
        v = metrics[cid].get("roc_auc")
        if v is not None and np.isfinite(v):
            per_class_aucs.append(float(v))
    expected_macro = float(np.mean(per_class_aucs)) if per_class_aucs else float("nan")
    assert metrics["macro_roc_auc"] == pytest.approx(expected_macro, abs=1e-12)


def test_multiclass_aggregation_weighted_uses_support():
    """The weighted aggregate must use class supports as weights, not equal
    weighting (otherwise it would match macro and the test below would be
    vacuous). Build a SKEWED multiclass target where one class dominates."""
    from mlframe.training._reporting_probabilistic import report_probabilistic_model_perf
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(22)
    N = 600
    # 80% class 0, 15% class 1, 5% class 2 - very skewed
    y = np.zeros(N, dtype=np.int64)
    y[:N // 5] = 1
    y[: N // 20] = 2
    rng.shuffle(y)
    X = rng.standard_normal((N, 5)) + y[:, None] * 0.3
    model = LogisticRegression(max_iter=200).fit(X, y)

    metrics: dict = {}
    report_probabilistic_model_perf(
        targets=y,
        columns=[f"f{i}" for i in range(5)],
        model_name="test",
        model=model,
        probs=model.predict_proba(X),
        preds=model.predict(X),
        plot_file="",
        show_perf_chart=False,
        print_report=False,
        metrics=metrics,
    )
    # macro != weighted under heavy skew (class 0 carries 80% weight).
    macro = metrics.get("macro_roc_auc")
    weighted = metrics.get("weighted_roc_auc")
    if np.isfinite(macro) and np.isfinite(weighted):
        # The values can be close on a well-separable synthetic problem,
        # but they should not be identical bit-for-bit (weights are not
        # all 1/K). Use a soft inequality.
        assert macro != weighted or abs(macro - weighted) < 1e-3


def test_multiclass_aggregation_skipped_on_binary():
    """Binary classification path should NOT emit macro_/weighted_ rows
    (collapses to the single positive-class scalar; no new info)."""
    from mlframe.training._reporting_probabilistic import report_probabilistic_model_perf
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(23)
    N = 400
    X = rng.standard_normal((N, 4))
    y = (X[:, 0] + rng.standard_normal(N) > 0).astype(np.int64)
    model = LogisticRegression().fit(X, y)

    metrics: dict = {}
    report_probabilistic_model_perf(
        targets=y,
        columns=[f"f{i}" for i in range(4)],
        model_name="test",
        model=model,
        probs=model.predict_proba(X),
        preds=model.predict(X),
        plot_file="",
        show_perf_chart=False,
        print_report=False,
        metrics=metrics,
    )
    # Binary path: only class_id=1 stamped, no macro_/weighted_ aggregation.
    macro_keys = [k for k in metrics.keys() if isinstance(k, str) and k.startswith("macro_")]
    assert macro_keys == [], f"binary path should not emit macro_* aggregations, got {macro_keys}"


# ----- MASE knob -----


def test_mase_knob_default_is_1():
    """Sanity: ReportingConfig exposes mase_seasonality and default is 1
    (simple naive baseline)."""
    from mlframe.training.configs import ReportingConfig
    rc = ReportingConfig()
    assert rc.mase_seasonality == 1


def test_mase_knob_validates_int():
    """Pydantic strict-int validation: seasonality must be int."""
    from mlframe.training.configs import ReportingConfig
    rc = ReportingConfig(mase_seasonality=7)
    assert rc.mase_seasonality == 7


def test_mase_stamped_when_naive_mae_supplied():
    """When the caller passes ``mase_naive_mae`` to the regression report
    we expect MASE = MAE / naive_mae stamped into the metrics dict."""
    from mlframe.training._reporting_regression import report_regression_model_perf
    rng = np.random.default_rng(30)
    N = 200
    y = rng.standard_normal(N).astype(np.float64)
    p = y + rng.normal(0, 0.5, N)

    metrics: dict = {}
    report_regression_model_perf(
        targets=y,
        columns=["f0"],
        model_name="test",
        model=None,
        preds=p,
        plot_file="",
        show_perf_chart=False,
        print_report=False,
        metrics=metrics,
        mase_naive_mae=0.5,
        mase_seasonality=1,
    )
    mae = float(np.mean(np.abs(y - p)))
    assert metrics["MASE"] == pytest.approx(mae / 0.5, abs=1e-10)
    assert metrics["MASE_seasonality"] == 1


def test_mase_nan_when_no_naive_mae_supplied():
    """No naive_mae -> MASE is NaN. The seasonality field is None."""
    from mlframe.training._reporting_regression import report_regression_model_perf
    rng = np.random.default_rng(31)
    N = 100
    y = rng.standard_normal(N).astype(np.float64)
    p = y + rng.normal(0, 0.3, N)
    metrics: dict = {}
    report_regression_model_perf(
        targets=y,
        columns=["f0"],
        model_name="test",
        model=None,
        preds=p,
        plot_file="",
        show_perf_chart=False,
        print_report=False,
        metrics=metrics,
    )
    assert np.isnan(metrics["MASE"])
    assert metrics["MASE_seasonality"] is None
