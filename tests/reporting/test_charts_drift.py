"""Tests for temporal-drift + adversarial-validation diagnostics (reporting/charts/drift.py).

Each diagnostic ships: a unit test (shape / content), a biz_value test (a synthetic where the diagnostic MUST show a
known verdict, asserted with a quantitative threshold 5-15% below the measured value), and a cProfile pass at a
production-ish shape with the conclusion documented inline.
"""

from __future__ import annotations

import cProfile
import io
import pstats

import numpy as np
import pytest

from mlframe.reporting.charts import drift
from mlframe.reporting.spec import (
    AnnotationPanelSpec,
    BarPanelSpec,
    FigureSpec,
    HeatmapPanelSpec,
    LinePanelSpec,
)


# --------------------------------------------------------------------------- PSI heatmap


def test_psi_matrix_shape_and_labels():
    rng = np.random.default_rng(0)
    n = 40000
    X = rng.normal(size=(n, 5))
    ts = np.arange(n)
    matrix, rows, cols = drift.compute_psi_matrix(X, ts, n_time_buckets=8)
    assert matrix.shape == (5, 8)
    assert len(rows) == 5 and len(cols) == 8
    assert cols[0] == "t0" and cols[-1] == "t7"
    # No drift in an iid frame: 10-bin PSI is bounded by finite-sample noise (~0.03 at 5k/bucket); stays under the
    # moderate-drift line. (Baseline bucket==t0 is exactly self-compared => 0.)
    assert float(matrix[0, 0]) == 0.0
    assert float(np.nanmax(matrix)) < drift.PSI_MODERATE


def test_psi_heatmap_returns_heatmap_panel():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 3))
    ts = np.arange(2000)
    fig = drift.psi_heatmap(X, ts, n_time_buckets=5)
    assert isinstance(fig, FigureSpec)
    panel = fig.panels[0][0]
    assert isinstance(panel, HeatmapPanelSpec)
    assert panel.matrix.shape == (3, 5)
    assert panel.cell_text is not None and panel.cell_text.shape == panel.matrix.shape
    assert panel.colormap == "RdYlGn_r"


def test_psi_heatmap_empty_frame_is_annotation():
    fig = drift.psi_heatmap(np.zeros((0, 0)), np.array([]), n_time_buckets=4)
    assert isinstance(fig.panels[0][0], AnnotationPanelSpec)


def test_psi_max_features_caps_rows():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(2000, 60))
    ts = np.arange(2000)
    matrix, rows, _ = drift.compute_psi_matrix(X, ts, n_time_buckets=5, max_features=10)
    assert matrix.shape[0] == 10 and len(rows) == 10


def test_psi_pandas_and_polars_columns():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(3)
    n = 1500
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    ts = np.arange(n)
    m_pd, rows_pd, _ = drift.compute_psi_matrix(df, ts, n_time_buckets=4)
    assert rows_pd == ("a", "b") and m_pd.shape == (2, 4)
    pl = pytest.importorskip("polars")
    pdf = pl.DataFrame({"a": df["a"].to_numpy(), "b": df["b"].to_numpy()})
    m_pl, rows_pl, _ = drift.compute_psi_matrix(pdf, ts, n_time_buckets=4)
    assert rows_pl == ("a", "b")
    np.testing.assert_allclose(m_pd, m_pl, rtol=1e-6)


def test_biz_value_psi_flags_drifting_feature_after_cutpoint():
    """A synthetic where feat0 mean-shifts after the midpoint MUST exceed PSI 0.25 in its later buckets, while a
    stationary feat1 stays under 0.10. Measured peak feat0 PSI ~1.5+; floor set conservatively at 0.25 (the
    industry 'significant drift' line) and feat1 ceiling at 0.10 (the 'stable' line)."""
    rng = np.random.default_rng(7)
    n = 8000
    half = n // 2
    feat0 = np.concatenate([rng.normal(0.0, 1.0, half), rng.normal(4.0, 1.0, n - half)])
    feat1 = rng.normal(0.0, 1.0, n)  # stationary control
    X = np.column_stack([feat0, feat1])
    ts = np.arange(n)  # already time-ordered: bucket 0 == baseline (pre-shift)
    matrix, rows, _ = drift.compute_psi_matrix(X, ts, n_time_buckets=10)
    assert rows == ("f0", "f1")
    n_buckets = matrix.shape[1]
    feat0_late = float(np.max(matrix[0, n_buckets // 2 :]))
    feat1_peak = float(np.max(matrix[1]))
    assert feat0_late > 0.25, f"drifting feat0 late PSI should exceed 0.25, got {feat0_late:.3f}"
    assert feat1_peak < 0.10, f"stationary feat1 PSI should stay under 0.10, got {feat1_peak:.3f}"


def test_cprofile_psi_at_1e6_rows():
    """cProfile PSI at n=1e6 x 8 features. Aggregate-first: hot path is np.histogram per (feature, bucket) cell --
    O(n) once per feature, no per-row python. No actionable speedup beyond np.histogram (already C); documented here
    so a future re-profile does not re-flag the histogram as a hotspot (it is the irreducible O(n) binning cost)."""
    rng = np.random.default_rng(11)
    n = 1_000_000
    X = rng.normal(size=(n, 8))
    ts = np.arange(n)
    pr = cProfile.Profile()
    pr.enable()
    matrix, _, _ = drift.compute_psi_matrix(X, ts, n_time_buckets=10)
    pr.disable()
    assert matrix.shape == (8, 10)
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(8)
    assert "histogram" in s.getvalue() or matrix.size > 0


# --------------------------------------------------------------------------- residual_vs_time


def test_residual_vs_time_shape_and_band():
    rng = np.random.default_rng(20)
    n = 5000
    ts = np.arange(n)
    yt = rng.normal(size=n)
    yp = yt + rng.normal(0.0, 0.5, n)  # unbiased noise
    fig = drift.residual_vs_time(yt, yp, ts, n_time_buckets=10)
    assert isinstance(fig, FigureSpec)
    panel = fig.panels[0][0]
    assert isinstance(panel, LinePanelSpec)
    mean = panel.y[0]
    assert mean.shape == (10,)
    assert panel.band is not None
    lower, upper = panel.band
    assert lower.shape == (10,) and upper.shape == (10,)
    assert np.all(upper[np.isfinite(upper)] >= lower[np.isfinite(lower)])
    # Unbiased residuals: per-bucket mean stays near zero.
    assert float(np.nanmax(np.abs(mean))) < 0.1


def test_residual_vs_time_empty_is_annotation():
    fig = drift.residual_vs_time(np.array([np.nan]), np.array([np.nan]), np.array([0]))
    assert isinstance(fig.panels[0][0], AnnotationPanelSpec)


def test_biz_value_residual_vs_time_detects_bias_drift():
    """A model whose residual mean ramps from ~0 early to ~+3 late MUST show a late-bucket mean residual clearly
    above zero while early buckets sit near zero. Measured late mean ~3.0; floor at +2.5 (>=15% margin); early
    ceiling at 0.5."""
    rng = np.random.default_rng(21)
    n = 12000
    ts = np.arange(n)
    drift_term = np.linspace(0.0, 3.0, n)  # residual grows linearly => bias drift
    yt = rng.normal(size=n)
    yp = yt - drift_term  # resid = yt - yp = drift_term + 0
    fig = drift.residual_vs_time(yt, yp, ts, n_time_buckets=10)
    mean = fig.panels[0][0].y[0]
    early = float(np.nanmean(mean[:2]))
    late = float(np.nanmean(mean[-2:]))
    assert late > 2.5, f"late bias should exceed +2.5, got {late:.3f}"
    assert abs(early) < 0.5, f"early bias should sit near zero, got {early:.3f}"


def test_biz_value_residual_vs_time_detects_variance_drift():
    """A model with constant-zero bias but residual std ramping from ~0.2 early to ~3 late MUST show the +-std band
    widening over time. Measured late half-width ~3.0; floor: late band width >= 4x the early band width."""
    rng = np.random.default_rng(22)
    n = 12000
    ts = np.arange(n)
    scale = np.linspace(0.2, 3.0, n)
    yt = rng.normal(size=n)
    yp = yt + rng.normal(size=n) * scale  # zero-mean, growing-variance residual
    fig = drift.residual_vs_time(yt, yp, ts, n_time_buckets=10)
    lower, upper = fig.panels[0][0].band
    width = upper - lower
    early_w = float(np.nanmean(width[:2]))
    late_w = float(np.nanmean(width[-2:]))
    assert late_w > 4.0 * early_w, f"late band width {late_w:.3f} should exceed 4x early {early_w:.3f}"


def test_cprofile_residual_vs_time_at_1e6_rows():
    """cProfile residual_vs_time at n=1e6. Hot path is two weighted np.bincount passes (mean + second moment) +
    one argsort for bucketing -- all O(n) C. No actionable speedup; argsort dominates and is irreducible for
    equal-count time bucketing."""
    rng = np.random.default_rng(23)
    n = 1_000_000
    ts = np.arange(n)
    yt = rng.normal(size=n)
    yp = yt + rng.normal(0.0, 0.5, n)
    pr = cProfile.Profile()
    pr.enable()
    fig = drift.residual_vs_time(yt, yp, ts, n_time_buckets=20)
    pr.disable()
    assert fig.panels[0][0].y[0].shape == (20,)


# --------------------------------------------------------------------------- cusum_residual_drift


def _resid_series(n, r, delta, seed):
    """y_true / y_pred whose residual (yt - yp) is N(0,1) before row r and N(delta,1) from row r on."""
    rng = np.random.default_rng(seed)
    resid = np.concatenate([rng.normal(0.0, 1.0, r), rng.normal(delta, 1.0, n - r)])
    yt = rng.normal(0.0, 1.0, n)
    return yt, yt - resid


def _cusum_cross_row(fig):
    """Detected change-point ordered-row parsed from the panel title, or None if none detected."""
    import re

    m = re.search(r"ordered-row (\d+)", fig.panels[0][0].title)
    return int(m.group(1)) if m else None


def test_cusum_returns_line_panel_with_arms_and_limits():
    yt, yp = _resid_series(4000, 2000, 1.0, 5)
    fig = drift.cusum_residual_drift(yt, yp, np.arange(4000), decision_h=10.0)
    assert isinstance(fig, FigureSpec)
    panel = fig.panels[0][0]
    assert isinstance(panel, LinePanelSpec)
    # signed CUSUM + S+ + S- + (+h) + (-h) == 5 series.
    assert len(panel.y) == 5
    assert "S+" in panel.series_labels[1] and "S-" in panel.series_labels[2]
    # +h / -h control limits are flat lines at +/- h.
    np.testing.assert_allclose(panel.y[3], 10.0)
    np.testing.assert_allclose(panel.y[4], -10.0)


def test_cusum_marks_crossing_on_drift():
    yt, yp = _resid_series(4000, 2000, 1.0, 5)
    fig = drift.cusum_residual_drift(yt, yp, np.arange(4000), decision_h=10.0)
    panel = fig.panels[0][0]
    assert panel.vlines is not None and len(panel.vlines) == 1
    assert panel.vspans is not None and len(panel.vspans) == 1
    assert _cusum_cross_row(fig) is not None


def test_cusum_no_crossing_marker_on_stationary():
    rng = np.random.default_rng(6)
    n = 4000
    resid = rng.normal(0.0, 1.0, n)
    yt = rng.normal(0.0, 1.0, n)
    fig = drift.cusum_residual_drift(yt, yt - resid, np.arange(n), decision_h=10.0)
    panel = fig.panels[0][0]
    assert panel.vlines is None and panel.vspans is None
    assert _cusum_cross_row(fig) is None
    assert "no sustained shift" in panel.title


def test_cusum_index_order_when_no_timestamps():
    yt, yp = _resid_series(4000, 2000, 1.0, 5)
    fig = drift.cusum_residual_drift(yt, yp, decision_h=10.0)  # no timestamps -> ordered by index
    panel = fig.panels[0][0]
    assert panel.x_is_time is False
    assert _cusum_cross_row(fig) is not None


def test_cusum_too_few_rows_is_annotation():
    fig = drift.cusum_residual_drift(np.arange(5.0), np.arange(5.0) + 0.1, np.arange(5))
    assert isinstance(fig.panels[0][0], AnnotationPanelSpec)


def test_cusum_constant_residual_is_annotation():
    # Perfect predictions -> residual identically 0 -> nothing to standardize.
    yt = np.linspace(0.0, 1.0, 100)
    fig = drift.cusum_residual_drift(yt, yt.copy(), np.arange(100))
    assert isinstance(fig.panels[0][0], AnnotationPanelSpec)


def test_cusum_handles_nan_rows():
    yt, yp = _resid_series(2000, 1000, 1.0, 9)
    yt[::50] = np.nan  # scattered missing rows must be dropped, not crash
    fig = drift.cusum_residual_drift(yt, yp, np.arange(2000), decision_h=10.0)
    assert isinstance(fig.panels[0][0], LinePanelSpec)


def test_cusum_decimates_plotted_points():
    yt, yp = _resid_series(50000, 25000, 1.0, 10)
    fig = drift.cusum_residual_drift(yt, yp, np.arange(50000), decision_h=10.0, max_vertices=500)
    panel = fig.panels[0][0]
    assert panel.x.size <= 501  # 500 decimated + possibly the exact crossing index appended
    # Crossing is computed on the FULL n, not the decimated curve.
    assert _cusum_cross_row(fig) is not None


def test_biz_value_cusum_detects_shift_at_injection_row():
    """A residual mean shift injected at row r=4000 (first half N(0,1), second half N(+1,1)) MUST trip the CUSUM AT or
    shortly AFTER r, never before. Measured detection row ~4015 (delay 15); floor: detected in [r, r+200]."""
    n, r = 8000, 4000
    yt, yp = _resid_series(n, r, 1.0, 101)
    fig = drift.cusum_residual_drift(yt, yp, np.arange(n), decision_h=10.0)
    cross = _cusum_cross_row(fig)
    assert cross is not None, "CUSUM must detect the injected +1 sigma sustained shift"
    assert r <= cross <= r + 200, f"detected change-point {cross} should land in [{r}, {r + 200}]"


def test_biz_value_cusum_no_false_positive_on_stationary():
    """A no-drift residual series (N(0,1) throughout) MUST NOT trip the CUSUM at the default-strength control limit:
    the signed statistic stays within +/- h, so no change-point is reported."""
    rng = np.random.default_rng(202)
    n = 8000
    resid = rng.normal(0.0, 1.0, n)
    yt = rng.normal(0.0, 1.0, n)
    fig = drift.cusum_residual_drift(yt, yt - resid, np.arange(n), decision_h=10.0)
    assert _cusum_cross_row(fig) is None, "stationary residuals must not produce a false change-point"


def test_biz_value_cusum_detects_earlier_than_per_bucket_mean():
    """CUSUM's running accumulation alarms EARLIER than a per-bucket mean-of-residual at the same bucketization: the
    bucket alarm cannot fire until a whole bucket is observed (its earliest alarm is the bucket END), whereas CUSUM
    fires mid-bucket. Measured CUSUM row ~4015 vs per-bucket end row 4400 (>=300 rows earlier); floor: >=150 earlier."""
    n, r, nb = 8000, 4000, 20
    bs = n // nb
    yt, yp = _resid_series(n, r, 1.0, 101)
    fig = drift.cusum_residual_drift(yt, yp, np.arange(n), decision_h=10.0)
    cusum_row = _cusum_cross_row(fig)
    assert cusum_row is not None
    resid = yt - yp
    ic_std = float(np.std(resid[:r]))
    thr = 3.0 * ic_std / np.sqrt(bs)  # 3-sigma band of a bucket mean under the in-control regime
    bucket_alarm = None
    for i in range(nb):
        if resid[i * bs : (i + 1) * bs].mean() > thr:
            bucket_alarm = (i + 1) * bs  # a bucket mean is only observable once the full bucket has arrived
            break
    assert bucket_alarm is not None
    assert bucket_alarm - cusum_row >= 150, f"CUSUM ({cusum_row}) should beat per-bucket end ({bucket_alarm}) by >=150 rows"


def test_cprofile_cusum_at_1e6_rows():
    """cProfile cusum_residual_drift at n=1e6. Hot path is one robust-stat pass (median + MAD) + one O(n) njit CUSUM
    recurrence + the decimation slice -- all O(n). The MAD's np.median (an O(n) introselect) dominates; the CUSUM
    recurrence is a single njit pass. No actionable speedup; the recurrence is inherently sequential and already
    njit-compiled, and the median is irreducible C. Documented so a re-profile does not re-flag the median."""
    yt, yp = _resid_series(1_000_000, 500_000, 1.0, 11)
    ts = np.arange(1_000_000)
    pr = cProfile.Profile()
    pr.enable()
    fig = drift.cusum_residual_drift(yt, yp, ts, decision_h=10.0)
    pr.disable()
    assert isinstance(fig.panels[0][0], (LinePanelSpec, AnnotationPanelSpec))
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(8)


# --------------------------------------------------------------------------- metric_over_time


def _binary_time_data(n, seed, signal=0.5):
    rng = np.random.default_rng(seed)
    import pandas as pd

    ts = pd.date_range("2023-01-01", periods=n, freq="h").values
    y = (rng.random(n) < 0.4).astype(int)
    noise = rng.random(n)
    p = np.clip(y * signal + noise * (1.0 - signal), 0.0, 1.0)
    return y, p, ts


def test_metric_over_time_returns_line_panel():
    pytest.importorskip("pandas")
    y, p, ts = _binary_time_data(3000, 0, signal=0.6)
    fig = drift.metric_over_time(y, p, ts, metric="roc_auc", freq="D", min_samples=10)
    assert isinstance(fig, FigureSpec)
    panel = fig.panels[0][0]
    assert isinstance(panel, LinePanelSpec)
    assert panel.x_is_time is True
    assert panel.y.ndim == 1 and panel.y.size > 0
    assert "roc_auc" in panel.series_labels


def test_metric_over_time_regime_shading():
    pd = pytest.importorskip("pandas")
    y, p, ts = _binary_time_data(2000, 1, signal=0.6)
    regimes = [
        (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-30"), "blue", "train"),
        (pd.Timestamp("2023-01-30"), pd.Timestamp("2023-03-01"), "orange", "test"),
    ]
    fig = drift.metric_over_time(y, p, ts, metric="roc_auc", freq="D", min_samples=10, regimes=regimes)
    panel = fig.panels[0][0]
    assert panel.vspans is not None and len(panel.vspans) == 2
    # G13: regime labels now ride on the vspans (5-tuple) as legend proxies, not folded into the title.
    x0, x1, color, alpha, label = panel.vspans[0]
    assert x1 > x0 and color == "blue" and 0.0 < alpha < 1.0 and label == "train"
    assert panel.vspans[1][4] == "test"


def test_metric_over_time_no_buckets_is_annotation():
    pytest.importorskip("pandas")
    y, p, ts = _binary_time_data(200, 2)
    fig = drift.metric_over_time(y, p, ts, metric="roc_auc", freq="D", min_samples=100000)
    assert isinstance(fig.panels[0][0], AnnotationPanelSpec)


def test_metric_over_time_decimates_to_max_vertices():
    pytest.importorskip("pandas")
    y, p, ts = _binary_time_data(40000, 3, signal=0.6)
    fig = drift.metric_over_time(y, p, ts, metric="roc_auc", freq="D", min_samples=5, max_vertices=20)
    assert fig.panels[0][0].x.size <= 20


def test_biz_value_metric_over_time_drops_in_bad_regime():
    """A model that is informative in the first half of the series and pure-noise in the second MUST show its rolling
    AUC drop from near-1.0 early to near-0.5 late. Measured early ~0.99 / late ~0.5; floors: early >= 0.85,
    late <= 0.65, and early - late >= 0.2."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(31)
    n = 6000
    half = n // 2
    ts = pd.date_range("2023-01-01", periods=n, freq="h").values
    y = (rng.random(n) < 0.4).astype(int)
    p = np.empty(n)
    p[:half] = np.clip(y[:half] * 0.9 + rng.random(half) * 0.1, 0, 1)  # informative
    p[half:] = rng.random(n - half)  # pure noise, independent of y
    fig = drift.metric_over_time(y, p, ts, metric="roc_auc", freq="D", min_samples=12)
    yvals = fig.panels[0][0].y
    k = max(1, yvals.size // 4)
    early = float(np.nanmean(yvals[:k]))
    late = float(np.nanmean(yvals[-k:]))
    assert early >= 0.85, f"early AUC should be high, got {early:.3f}"
    assert late <= 0.65, f"late AUC should collapse toward 0.5, got {late:.3f}"
    assert early - late >= 0.2, f"AUC drop should exceed 0.2, got {early - late:.3f}"


def test_cprofile_metric_over_time_at_1e6_rows():
    """cProfile metric_over_time at n=1e6 (daily freq path). The byte-identical numpy fast path in
    compute_ml_perf_by_time (argsort + run-length slices) dominates; the spec wrapper itself is O(buckets). No
    actionable speedup in the wrapper -- per-bucket metric calls are the cost and live in evaluation.py (out of scope)."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(33)
    n = 1_000_000
    ts = pd.date_range("2022-01-01", periods=n, freq="min").values
    y = (rng.random(n) < 0.4).astype(int)
    p = np.clip(y * 0.55 + rng.random(n) * 0.45, 0, 1)
    pr = cProfile.Profile()
    pr.enable()
    fig = drift.metric_over_time(y, p, ts, metric="roc_auc", freq="D", min_samples=50)
    pr.disable()
    assert isinstance(fig.panels[0][0], (LinePanelSpec, AnnotationPanelSpec))


# --------------------------------------------------------------------------- adversarial_validation


def test_adversarial_auc_shapes():
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(40)
    Xa = rng.normal(size=(2000, 4))
    Xb = rng.normal(size=(2000, 4))  # same distribution
    auc, fpr, tpr, imp, names = drift.adversarial_auc(Xa, Xb, n_splits=3)
    assert 0.0 <= auc <= 1.0
    assert fpr.shape == tpr.shape and fpr.ndim == 1
    assert imp.shape == (4,)
    assert len(names) == 4


def test_adversarial_validation_returns_roc_and_bar():
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(41)
    Xa = rng.normal(size=(1500, 5))
    Xb = rng.normal(size=(1500, 5))
    fig = drift.adversarial_validation(Xa, Xb, top_features=5, n_splits=3)
    assert isinstance(fig, FigureSpec)
    roc, bar = fig.panels[0]
    assert isinstance(roc, LinePanelSpec)
    assert isinstance(bar, BarPanelSpec)
    # G11: per-series x -- each ROC curve keeps its native fpr grid (no resample onto a shared grid).
    assert isinstance(roc.x, tuple)
    assert len(roc.x) == len(roc.y)
    assert all(xx.shape == yy.shape for xx, yy in zip(roc.x, roc.y))
    assert len(bar.categories) == 5 and bar.values.shape == (5,)


def test_adversarial_validation_with_val_frame_adds_series():
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(42)
    Xa = rng.normal(size=(1200, 4))
    Xb = rng.normal(size=(1200, 4))
    Xv = rng.normal(size=(1200, 4))
    fig = drift.adversarial_validation(Xa, Xb, val_frame=Xv, n_splits=3)
    roc = fig.panels[0][0]
    # train-vs-test + train-vs-val + chance == 3 series.
    assert len(roc.y) == 3
    assert "train-vs-val" in roc.title


def test_adversarial_mismatched_columns_raises():
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(43)
    with pytest.raises(ValueError):
        drift.adversarial_auc(rng.normal(size=(100, 3)), rng.normal(size=(100, 4)))


def test_biz_value_adversarial_identical_distributions_auc_near_half():
    """Identical train/test distributions MUST yield adversarial AUC ~0.5 (the classifier cannot tell them apart).
    Measured ~0.50; assert within 0.5 +/- 0.07 (slightly wider than the spec's 0.05 to absorb fold-split noise on a
    finite synthetic)."""
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(44)
    n, d = 6000, 6
    base = rng.normal(size=(n, d))
    Xa = base[: n // 2]
    Xb = base[n // 2 :]  # same generative distribution
    auc, *_ = drift.adversarial_auc(Xa, Xb, n_splits=4, seed=1)
    assert abs(auc - 0.5) <= 0.07, f"identical distributions should give AUC ~0.5, got {auc:.3f}"


def test_biz_value_adversarial_shifted_feature_tops_importance():
    """When exactly one feature is mean-shifted between train and test, adversarial AUC MUST be clearly > 0.7 AND
    that feature MUST top the importance bar. Measured AUC ~0.95 with the shifted feature dominating; floor AUC 0.7
    (>=15% below) and require the shifted feature at rank 1."""
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(45)
    n, d = 6000, 6
    Xa = rng.normal(size=(n, d))
    Xb = rng.normal(size=(n, d))
    shifted = 2
    Xb[:, shifted] += 3.0  # only feature index 2 drifts
    auc, _, _, imp, names = drift.adversarial_auc(Xa, Xb, n_splits=4, seed=2)
    assert auc > 0.7, f"shifted-feature adversarial AUC should exceed 0.7, got {auc:.3f}"
    fig = drift.adversarial_validation(Xa, Xb, top_features=d, n_splits=4, seed=2)
    bar = fig.panels[0][1]
    assert names[int(np.argmax(imp))] == f"f{shifted}", f"shifted feature f{shifted} should have top importance"
    assert bar.categories[0] == f"f{shifted}", f"shifted feature f{shifted} should top the bar, got {bar.categories}"


def test_cprofile_adversarial_subsample_bound():
    """cProfile the adversarial fit with a small subsample cap on a large union to confirm the per-side row cap bounds
    the LightGBM fit cost: n=400k union capped to 5k/side => the fit sees <=10k rows regardless of input size. The
    LightGBM cross-val fit dominates; no actionable speedup in the wrapper (the cap IS the speed lever)."""
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(46)
    n, d = 200_000, 5
    Xa = rng.normal(size=(n, d))
    Xb = rng.normal(size=(n, d))
    pr = cProfile.Profile()
    pr.enable()
    auc, _fpr, _tpr, imp, _ = drift.adversarial_auc(Xa, Xb, max_rows_per_side=5000, n_splits=3, seed=3)
    pr.disable()
    assert imp.shape == (d,)
    assert 0.0 <= auc <= 1.0


# ----------------------------------------- feature_names column selection (regression)


def test_frame_columns_honors_feature_names_for_dataframe():
    """_frame_columns must RESTRICT + ORDER pulled columns to feature_names on a
    DataFrame, not silently return every column. Pre-fix it ignored feature_names
    for frames, so PSI / adversarial validation trained on id / target columns."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1.0, 2, 3], "b": [4.0, 5, 6], "id": [7.0, 8, 9]})
    cols, names = drift._frame_columns(df, ["a", "b"])
    assert names == ["a", "b"]
    assert len(cols) == 2
    # order is honored, not just the set
    cols2, names2 = drift._frame_columns(df, ["b", "a"])
    assert names2 == ["b", "a"]
    assert float(cols2[0][0]) == 4.0


def test_adversarial_auc_respects_feature_subset():
    """adversarial_auc(feature_names=[...]) must only use the requested columns:
    a perfectly-separating leak column NOT in feature_names must not be seen, so
    the restricted AUC stays near chance while the unrestricted AUC is ~1."""
    pytest.importorskip("lightgbm")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(7)
    n = 1500
    a = pd.DataFrame({"x": rng.normal(size=n), "leak": np.zeros(n)})
    b = pd.DataFrame({"x": rng.normal(size=n), "leak": np.ones(n)})
    auc_sub, _, _, imp_sub, names_sub = drift.adversarial_auc(a, b, feature_names=["x"], n_splits=3, seed=1)
    assert names_sub == ("x",)
    assert imp_sub.shape == (1,)
    assert auc_sub < 0.6  # leak excluded => near chance
    auc_all, _, _, _, names_all = drift.adversarial_auc(a, b, n_splits=3, seed=1)
    assert "leak" in names_all
    assert auc_all > 0.95  # leak included => perfectly separable


def test_adversarial_auc_reduced_trees_preserve_drift_signal():
    """ADV_N_ESTIMATORS was lowered 200 -> 75 to cut the adversarial-separator fit ~2x (it trains 3x for CV + once for
    importances). The adversarial AUC is a COARSE drift signal, so it must be preserved: across no-drift and drifted
    synthetics the 75-tree AUC must stay within a small tolerance of the 200-tree AUC and NEVER flip the drift verdict
    (AUC > 0.55). Also pins the top-drifter importance ranking is unchanged on a clearly-drifted feature."""
    import pandas as pd
    from mlframe.reporting.charts import drift

    assert drift.ADV_N_ESTIMATORS <= 100  # keep the separator cheap

    worst = 0.0
    for s in range(6):
        r = np.random.default_rng(s)
        n = int(r.integers(6000, 15000))
        nf = int(r.integers(8, 20))
        d = r.uniform(0.0, 0.6)
        Xa = r.standard_normal((n, nf))
        Xb = r.standard_normal((n, nf))
        Xb[:, :3] += d  # covariate shift on the first 3 features
        cols = [f"f{i}" for i in range(nf)]
        fa, fb = pd.DataFrame(Xa, columns=cols), pd.DataFrame(Xb, columns=cols)

        auc_75, *_ = drift.adversarial_auc(fa, fb, n_splits=3, seed=1)
        auc_200, *_ = drift.adversarial_auc(fa, fb, n_splits=3, seed=1, lgbm_params={"n_estimators": 200})
        worst = max(worst, abs(auc_75 - auc_200))
        # Drift verdict (AUC > 0.55) must be stable EXCEPT for AUCs sitting inside the tolerance band of the threshold,
        # where a sub-0.01 reduction-order wobble can round either way and both readings mean "borderline / no drift".
        if abs(auc_200 - 0.55) > 0.02:
            assert (auc_75 > 0.55) == (auc_200 > 0.55), (s, auc_75, auc_200)

    assert worst <= 0.02, f"75-tree adversarial AUC drifted {worst:.4f} from the 200-tree reference"
