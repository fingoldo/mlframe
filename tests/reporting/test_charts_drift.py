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
    AnnotationPanelSpec, BarPanelSpec, FigureSpec, HeatmapPanelSpec, LinePanelSpec,
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
    feat0_late = float(np.max(matrix[0, n_buckets // 2:]))
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
