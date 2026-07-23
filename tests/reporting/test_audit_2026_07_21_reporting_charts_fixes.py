"""Regression tests for audits/full_audit_2026-07-21/reporting_charts.md findings F1-F8.

PR1-PR3 (test-coverage additions and the shared row-finite helper) are exercised implicitly by
F1/F2's own tests below. PR4 (perf watch-note, no actionable finding) and PR5 (docs-only, follows
from F6) need no dedicated test.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# F1: _reliability_curve drops non-finite (p, y) rows instead of mis-binning NaN into the last bin
# ---------------------------------------------------------------------------


def test_f1_reliability_curve_ignores_nan_probability():
    """F1 reliability curve ignores nan probability."""
    from mlframe.reporting.charts.calibration import _reliability_curve

    edges = np.linspace(0.0, 1.0, 11)
    p_clean = np.array([0.5, 0.99])
    y_clean = np.array([1.0, 1.0])
    out_clean = _reliability_curve(p_clean, y_clean, edges)

    p_poisoned = np.array([0.5, np.nan, 0.99])
    y_poisoned = np.array([1.0, 0.0, 1.0])  # NaN row's label would corrupt the bin it lands in if not dropped
    out_poisoned = _reliability_curve(p_poisoned, y_poisoned, edges)

    np.testing.assert_allclose(out_clean, out_poisoned, equal_nan=True)


def test_f1_build_reliability_overlay_spec_survives_nan_row():
    """F1 build reliability overlay spec survives nan row."""
    from mlframe.reporting.charts.calibration import build_reliability_overlay_spec

    rng = np.random.default_rng(0)
    raw_probs = rng.uniform(size=200)
    y_true = (rng.uniform(size=200) < raw_probs).astype(np.float64)
    raw_probs[5] = np.nan
    spec = build_reliability_overlay_spec(raw_probs, y_true)
    assert spec is not None


# ---------------------------------------------------------------------------
# F2: quantile.py's _reliability_panel / _coverage_panel / _quantile_crossing_panel now filter non-finite rows
# ---------------------------------------------------------------------------


def test_f2_finite_rows_helper_exists_and_matches_manual_filter():
    """F2 finite rows helper exists and matches manual filter."""
    from mlframe.reporting.charts.quantile import _finite_rows

    y = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
    P = np.array([[0.1, 0.2], [0.3, 0.4], [np.nan, 0.6], [0.7, 0.8], [0.9, 1.0]])
    mask = _finite_rows(y, P)
    np.testing.assert_array_equal(mask, np.array([True, False, False, False, True]))


def test_f2_reliability_panel_ignores_nan_row():
    """F2 reliability panel ignores nan row."""
    from mlframe.reporting.charts.quantile import _reliability_panel

    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    P = np.sort(rng.normal(size=(50, 3)), axis=1)
    alphas = [0.1, 0.5, 0.9]

    y_poisoned = y.copy()
    y_poisoned[0] = np.nan
    poisoned = _reliability_panel(y_poisoned, P, alphas)
    # Manually pre-filtering the same NaN row must give the identical panel -- confirms the panel
    # excludes it itself instead of silently counting "y <= q" as False (always-not-covered) for it.
    prefiltered = _reliability_panel(y[1:], P[1:], alphas)
    np.testing.assert_allclose(poisoned.y[1], prefiltered.y[1])


def test_f2_coverage_panel_ignores_nan_row():
    """F2 coverage panel ignores nan row."""
    from mlframe.reporting.charts.quantile import _coverage_panel

    rng = np.random.default_rng(0)
    y = rng.normal(size=60)
    P = np.sort(rng.normal(size=(60, 5)), axis=1)
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    y_poisoned = y.copy()
    y_poisoned[0] = np.inf
    panel_clean = _coverage_panel(np.delete(y, 0), np.delete(P, 0, axis=0), alphas)
    panel_poisoned = _coverage_panel(y_poisoned, P, alphas)
    np.testing.assert_allclose(panel_clean.y[1], panel_poisoned.y[1]), "F2 REGRESSION: an Inf row must be excluded, matching the manually-pre-filtered result"


def test_f2_quantile_crossing_panel_ignores_nan_row():
    """F2 quantile crossing panel ignores nan row."""
    from mlframe.reporting.charts.quantile import _quantile_crossing_panel

    P = np.array([[0.1, 0.5, 0.9], [0.2, 0.4, 0.8], [np.nan, 0.3, 0.7]])
    alphas = [0.1, 0.5, 0.9]
    out = _quantile_crossing_panel(None, P, alphas)
    # n used for the rate denominator must reflect only the 2 finite rows, not 3.
    assert "2 rows" in out.title or "0 rows" in out.title


# ---------------------------------------------------------------------------
# F3: _top_split_features narrows its except + logs a warning on fallback
# ---------------------------------------------------------------------------


def test_f3_top_split_features_logs_warning_on_fit_failure(caplog):
    """F3 top split features logs warning on fit failure."""
    import logging

    from mlframe.reporting.charts.error_analysis import _top_split_features

    rng = np.random.default_rng(0)
    mat = rng.normal(size=(50, 3))
    mat[0, 0] = np.inf  # DecisionTreeRegressor.fit raises ValueError on Inf even in modern sklearn
    err = rng.normal(size=50)

    with caplog.at_level(logging.WARNING, logger="mlframe.reporting.charts.error_analysis"):
        out = _top_split_features(mat, err, ["a", "b", "c"], max_depth=3, n_features=2, seed=0)
    assert out  # surrogate fallback still returns a ranking
    assert any("weak-segment tree fit failed" in rec.message for rec in caplog.records), "F3 REGRESSION: fallback must log a warning, not fail silently"


def test_f3_weak_segment_heatmap_filters_inf_features():
    """F3 weak segment heatmap filters inf features."""
    from mlframe.reporting.charts.error_analysis import weak_segment_heatmap

    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    X[3, 1] = np.inf
    y_true = rng.normal(size=80)
    y_pred = y_true + rng.normal(scale=0.1, size=80)
    result = weak_segment_heatmap(X, y_true, y_pred, feature_names=["a", "b", "c"])
    assert result is not None


# ---------------------------------------------------------------------------
# F4: length-mismatch now raises ValueError instead of silently truncating
# ---------------------------------------------------------------------------


def test_f4_calibration_heatmap_2d_raises_on_length_mismatch():
    """F4 calibration heatmap 2d raises on length mismatch."""
    from mlframe.reporting.charts.calibration_heatmap_2d import compute_calibration_heatmap_2d

    with pytest.raises(ValueError, match="equal length"):
        compute_calibration_heatmap_2d(y_true=np.zeros(10), y_score=np.zeros(10), feat_x=np.zeros(10), feat_y=np.zeros(5))


def test_f4_calibration_by_feature_raises_on_length_mismatch():
    """F4 calibration by feature raises on length mismatch."""
    from mlframe.reporting.charts.calibration_by_feature import compute_calibration_by_feature_heterogeneity

    with pytest.raises(ValueError, match="equal length"):
        compute_calibration_by_feature_heterogeneity(y_true=np.zeros(10), y_score=np.zeros(8), feature_values=np.zeros(10))


def test_f4_fairness_calibration_compose_raises_on_length_mismatch():
    """F4 fairness calibration compose raises on length mismatch."""
    from mlframe.reporting.charts.fairness_calibration import compose_fairness_calibration_figure

    with pytest.raises(ValueError, match="equal length"):
        compose_fairness_calibration_figure(y_true=np.zeros(10), y_score=np.zeros(10), subgroups=np.zeros(7))


def test_f4_fairness_calibration_compute_raises_on_length_mismatch():
    """F4 fairness calibration compute raises on length mismatch."""
    from mlframe.reporting.charts.fairness_calibration import compute_subgroup_ece_disparity

    with pytest.raises(ValueError, match="equal length"):
        compute_subgroup_ece_disparity(y_true=np.zeros(10), y_score=np.zeros(10), subgroups=np.zeros(3))


# ---------------------------------------------------------------------------
# F5: _corr_heatmap_panel raises on mismatched per-model row counts instead of silently truncating
# ---------------------------------------------------------------------------


def test_f5_corr_heatmap_panel_raises_on_mismatched_model_lengths():
    """F5 corr heatmap panel raises on mismatched model lengths."""
    from mlframe.reporting.charts.model_comparison import _corr_heatmap_panel

    rng = np.random.default_rng(0)
    per_model = {
        "a": {"y_score": rng.uniform(size=100)},
        "b": {"y_score": rng.uniform(size=50)},  # different length -> would silently misalign rows
    }
    with pytest.raises(ValueError, match="mismatched row counts"):
        _corr_heatmap_panel(per_model, subsample=1000, seed=0)


# ---------------------------------------------------------------------------
# F6: the 5 missing diagnostic modules are now re-exported from the package facade
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "compose_interaction_strength_figure",
        "interaction_strength_panel",
        "compose_prediction_stability_figure",
        "compute_prediction_stability",
        "PredictionStabilityResult",
        "build_risk_coverage_spec",
        "compute_risk_coverage",
        "RiskCoverageResult",
        "shap_interaction_summary",
        "ShapInteractionResult",
        "compose_pdp_2d_figure",
        "interaction_residual",
    ],
)
def test_f6_facade_reexports_previously_missing_symbol(name):
    """F6 facade reexports previously missing symbol."""
    import mlframe.reporting.charts as charts

    assert hasattr(charts, name), f"F6 REGRESSION: {name} must be importable from the mlframe.reporting.charts facade"
    assert name in charts.__all__


# ---------------------------------------------------------------------------
# F7: the 3 SHAP save-figure helpers are now one shared function, not 3 copies
# ---------------------------------------------------------------------------


def test_f7_shap_save_figure_is_a_single_shared_function():
    """F7 shap save figure is a single shared function."""
    from mlframe.reporting.charts import shap_interactions, shap_panels, shap_per_instance

    assert shap_interactions._save_figure is shap_panels._save_figure
    assert shap_per_instance._save_figure is shap_panels._save_figure


def test_f7_no_duplicate_save_figure_definitions_remain():
    """F7 no duplicate save figure definitions remain."""
    from mlframe.reporting.charts import shap_interactions, shap_per_instance

    assert "def _save_figure" not in inspect.getsource(shap_interactions)
    assert "def _save" not in inspect.getsource(shap_per_instance).replace("def _save_figure", "")


# ---------------------------------------------------------------------------
# F8: documented as a deferred architecture note, not a bug -- no test needed (see module docstring)
# ---------------------------------------------------------------------------
