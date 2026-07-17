"""Tests for commit-1 salvage: behavioural, no source-inspection."""

from __future__ import annotations


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

from mlframe.evaluation.reports import (
    predictions_beautify_linear,
    plot_beautified_lift,
    plot_pr_curve,
    plot_roc_curve,
)
from mlframe.training.evaluation import (
    compute_ml_perf_by_time,
    visualize_ml_metric_by_time,
)
from mlframe.preprocessing.outliers import (
    compute_outlier_detector_score,
    count_num_outofranges,
    compute_naive_outlier_score,
)
from mlframe.metrics.core import brier_and_precision_score, make_brier_precision_scorer
from mlframe.training.callbacks import stop_file


# ---------------------------------------------------------------------------
# predictions_beautify_linear
# ---------------------------------------------------------------------------


def _precision_at_top_decile(y, p):
    """Helper that precision at top decile."""
    k = max(1, len(p) // 10)
    idx = np.argsort(-p)[:k]
    return float(np.mean(y[idx]))


def test_beautify_alpha_zero_identity():
    """Beautify alpha zero identity."""
    rng = np.random.default_rng(0)
    preds = rng.random(100)
    y = rng.integers(0, 2, size=100)
    out = predictions_beautify_linear(preds, y, alpha=0.0)
    np.testing.assert_allclose(out, preds)


def test_beautify_alpha_one_returns_y():
    """Beautify alpha one returns y."""
    rng = np.random.default_rng(1)
    preds = rng.random(50)
    y = rng.integers(0, 2, size=50)
    out = predictions_beautify_linear(preds, y, alpha=1.0)
    np.testing.assert_allclose(out, y.astype(float))


def test_beautify_increasing_alpha_lifts_precision():
    """Beautify increasing alpha lifts precision."""
    rng = np.random.default_rng(42)
    n = 500
    y = rng.integers(0, 2, size=n)
    preds = rng.random(n)
    p0 = _precision_at_top_decile(y, predictions_beautify_linear(preds, y, alpha=0.0))
    p1 = _precision_at_top_decile(y, predictions_beautify_linear(preds, y, alpha=0.1))
    p2 = _precision_at_top_decile(y, predictions_beautify_linear(preds, y, alpha=0.3))
    assert p1 >= p0
    assert p2 >= p1


# ---------------------------------------------------------------------------
# plot_beautified_lift
# ---------------------------------------------------------------------------


def test_plot_beautified_lift_returns_figure():
    """Plot beautified lift returns figure."""
    rng = np.random.default_rng(3)
    preds = rng.random(200)
    y = rng.integers(0, 2, size=200)
    alphas = (0.0, 0.01, 0.05, 0.1, 0.2)
    fig = plot_beautified_lift(preds, y, alphas=alphas, metric="precision_at_top_decile")
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    line = ax.lines[0]
    assert len(line.get_xdata()) == len(alphas)
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_pr_curve / plot_roc_curve
# ---------------------------------------------------------------------------


def test_plot_pr_and_roc_smoke():
    """Plot pr and roc smoke."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression

    data = load_breast_cancer()
    X, y = data.data, data.target
    clf = LogisticRegression(max_iter=2000).fit(X, y)
    preds = clf.predict_proba(X)[:, 1]
    fig_pr = plot_pr_curve(y, preds, show_calibration=True)
    fig_roc = plot_roc_curve(y, preds, show_calibration=True)
    assert isinstance(fig_pr, Figure)
    assert isinstance(fig_roc, Figure)
    plt.close(fig_pr)
    plt.close(fig_roc)


# ---------------------------------------------------------------------------
# compute_ml_perf_by_time / visualize_ml_metric_by_time
# ---------------------------------------------------------------------------


def test_compute_ml_perf_by_time_basic():
    """Compute ml perf by time basic."""
    rng = np.random.default_rng(7)
    n = 1000
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h")
    # 10 daily bins (~24h*10 = 240; we have 1000 hours -> ~42 days)
    # Use freq="D", 1000 rows across many days
    y_true = rng.integers(0, 2, size=n)
    y_pred = rng.random(n)
    perf = compute_ml_perf_by_time(y_true, y_pred, timestamps, freq="D", metric="roc_auc", min_samples=1)
    assert "roc_auc" in perf.columns
    assert "n_samples" in perf.columns
    assert int(perf["n_samples"].sum()) == n


def test_visualize_ml_metric_by_time_returns_figure():
    """Visualize ml metric by time returns figure."""
    rng = np.random.default_rng(11)
    n = 240
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h")
    y_true = rng.integers(0, 2, size=n)
    y_pred = rng.random(n)
    perf = compute_ml_perf_by_time(y_true, y_pred, timestamps, freq="D", metric="roc_auc", min_samples=1)
    fig = visualize_ml_metric_by_time(perf, good_metric_threshold=0.5)
    assert isinstance(fig, Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# stop_file
# ---------------------------------------------------------------------------


def test_stop_file(tmp_path):
    """Stop file."""
    f = tmp_path / "stop.flag"
    check = stop_file(str(f))
    assert check() is False
    f.write_text("x")
    assert check() is True


# ---------------------------------------------------------------------------
# Library-specific callbacks — smoke only; rely on importorskip
# ---------------------------------------------------------------------------


def test_catboost_stopfile_smoke(tmp_path):
    """Catboost stopfile smoke."""
    pytest.importorskip("catboost")
    from mlframe.training.callbacks import CatBoostStopFileCallback

    cb = CatBoostStopFileCallback(str(tmp_path / "stop.flag"))
    assert cb is not None


def test_lightgbm_stopfile_smoke(tmp_path):
    """Lightgbm stopfile smoke."""
    pytest.importorskip("lightgbm")
    from mlframe.training.callbacks import LightGBMStopFileCallback

    cb = LightGBMStopFileCallback(str(tmp_path / "stop.flag"))
    assert cb is not None


def test_xgboost_stopfile_smoke(tmp_path):
    """Xgboost stopfile smoke."""
    pytest.importorskip("xgboost")
    from mlframe.training.callbacks import XGBoostStopFileCallback

    cb = XGBoostStopFileCallback(str(tmp_path / "stop.flag"))
    assert cb is not None


def test_lightning_stopfile_smoke(tmp_path):
    """Lightning stopfile smoke."""
    pytest.importorskip("pytorch_lightning")
    from mlframe.training.callbacks import LightningStopFileCallback

    cb = LightningStopFileCallback(str(tmp_path / "stop.flag"))
    assert cb is not None


# ---------------------------------------------------------------------------
# Outlier scores
# ---------------------------------------------------------------------------


def test_count_num_outofranges_and_naive_score():
    """Count num outofranges and naive score."""
    rng = np.random.default_rng(13)
    X_train = rng.normal(size=(200, 5))
    X_test = rng.normal(size=(50, 5))
    # Corrupt a few rows to force out-of-range
    X_test[0] = 1e6
    X_test[1] = -1e6
    mins = X_train.min(axis=0)
    maxs = X_train.max(axis=0)
    counts = count_num_outofranges(X_test, mins, maxs)
    assert counts.shape == (50,)
    assert counts[0] == 5 and counts[1] == 5

    score = compute_naive_outlier_score(X_train, X_test)
    assert score.shape == (50,)
    assert float(score.min()) >= 0.0
    assert float(score.max()) <= 1.0
    assert score[0] == 1.0
    assert score[1] == 1.0


def test_count_num_outofranges_is_parallel_and_matches_numpy():
    """count_num_outofranges must stay parallel (prange) and bit-identical to the vectorised numpy reference.

    The per-row count is an order-invariant integer reduction, so parallelisation cannot change the result; this pins both the parallel signature
    (a serial revert drops the n=10M win) and the bit-identity against numpy.
    """
    assert getattr(count_num_outofranges, "targetoptions", {}).get("parallel") is True, (
        "count_num_outofranges must be njit(parallel=True) for the n=10M row-parallel win"
    )

    rng = np.random.default_rng(7)
    X = rng.normal(size=(40_000, 6))
    mins = X[:1000].min(axis=0)
    maxs = X[:1000].max(axis=0)
    got = count_num_outofranges(X, mins, maxs)
    ref = ((X < mins) | (X > maxs)).sum(axis=1).astype(np.int64)
    assert np.array_equal(got, ref)


def test_nanminmax_cols_is_parallel_and_matches_numpy():
    """_nanminmax_cols must stay njit(parallel=True) and be bit-identical to np.nanmin/np.nanmax (incl. all-NaN columns).

    The fused single-pass min/max replaces two full np.nanmin + np.nanmax sweeps (~19x e2e on compute_naive_outlier_score at n=10M, 2026-06-15);
    the per-column min/max reduction is order-invariant so parallelisation cannot change the result. A serial revert or a switch back to the two
    np.nan* sweeps drops the win, and an all-NaN column must collapse to NaN exactly like numpy's empty-slice result.
    """
    from mlframe.preprocessing.outliers import _nanminmax_cols

    assert getattr(_nanminmax_cols, "targetoptions", {}).get("parallel") is True, "_nanminmax_cols must be njit(parallel=True) for the n=10M fused-pass win"

    rng = np.random.default_rng(11)
    X = rng.normal(size=(40_000, 7))
    X[rng.integers(0, X.shape[0], 50), 0] = np.nan
    X[:, 3] = np.nan  # all-NaN column -> numpy returns NaN for both bounds
    got_min, got_max = _nanminmax_cols(X)
    with np.errstate(invalid="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref_min = np.nanmin(X, axis=0)
            ref_max = np.nanmax(X, axis=0)
    assert np.array_equal(got_min, ref_min, equal_nan=True)
    assert np.array_equal(got_max, ref_max, equal_nan=True)


def test_compute_outlier_detector_score():
    """Compute outlier detector score."""
    from sklearn.ensemble import IsolationForest

    rng = np.random.default_rng(17)
    X = rng.normal(size=(100, 3))
    det = IsolationForest(random_state=0, n_estimators=20).fit(X)
    s = compute_outlier_detector_score(det, X)
    assert s.shape == (100,)
    assert set(np.unique(s)).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# brier_and_precision_score
# ---------------------------------------------------------------------------


def test_brier_and_precision_score_passing_and_failing():
    # Nearly-perfect predictions -> should pass both thresholds -> positive value
    """Brier and precision score passing and failing."""
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    good = np.array([0.02, 0.03, 0.01, 0.98, 0.97, 0.95, 0.04, 0.99])
    val_good = brier_and_precision_score(y, good, precision_threshold=0.5, brier_threshold=0.25)
    assert val_good > 0

    # Flat noisy preds -> fails thresholds -> 0
    bad = np.full_like(y, 0.5, dtype=float)
    val_bad = brier_and_precision_score(y, bad, precision_threshold=0.9, brier_threshold=0.05)
    assert val_bad == 0.0


def test_make_brier_precision_scorer():
    """Make brier precision scorer."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X, y = data.data, data.target
    clf = LogisticRegression(max_iter=2000).fit(X, y)
    scorer = make_brier_precision_scorer()
    val = scorer(clf, X, y)
    assert isinstance(val, float)
    assert val >= 0.0


# ---------------------------------------------------------------------------
# Keras compat
# ---------------------------------------------------------------------------


def _import_keras_or_skip():
    """Helper that import keras or skip."""
    pytest.importorskip("tensorflow")
    pytest.importorskip("keras")
    from mlframe.training.neural.keras_compat import build_keras_mlp

    # Try an actual build; surface broken installs as ImportError via importorskip semantics
    # rather than silently skipping (memory feedback_no_mask_via_canon_or_guards).
    try:
        build_keras_mlp(num_layers=1, num_neurons=2, input_dim=2)
    except ImportError as exc:
        pytest.skip(f"keras optional dependency missing: {exc}")


def test_build_keras_mlp_smoke():
    """Build keras mlp smoke."""
    _import_keras_or_skip()
    from mlframe.training.neural.keras_compat import build_keras_mlp

    m = build_keras_mlp(num_layers=1, num_neurons=8, input_dim=4)
    assert m is not None


def test_keras_compatible_mlp_smoke():
    """Keras compatible mlp smoke."""
    _import_keras_or_skip()
    from mlframe.training.neural.keras_compat import KerasCompatibleMLP

    rng = np.random.default_rng(0)
    X = rng.random((16, 3)).astype(np.float32)
    y = rng.random(16).astype(np.float32)
    reg = KerasCompatibleMLP(num_layers=1, num_neurons=4, epochs=1, batch_size=4, loss="mse")
    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds.shape == (16,)
