"""Cluster 20 regressions: a NaN label keeps the calibration noise floor, the Spearman bench uses the live threshold name, and a failed SHAP pair choice is visible
(audit 2026-11-20, findings 20-F10, 20-F25 and 20-F27).

One NaN label made the prevalence NaN, so the noise floor was NaN and the traffic light silently lost the floor it is meant to apply. The bench read and
wrote an attribute that no longer exists on prediction_stability. The 2-D PDP swallowed a SHAP failure at DEBUG and quietly changed which pair it drew.
"""

import logging
from pathlib import Path

import numpy as np

from mlframe.reporting.charts import _rank_stats
from mlframe.reporting.charts.calibration_by_feature import compute_calibration_by_feature_heterogeneity
from mlframe.reporting.charts.pdp_2d import _default_pair


def test_one_nan_label_keeps_a_finite_noise_floor():
    """With a NaN label among the targets the noise floor is still finite, and equals the floor without that row's label effect (20-F10)."""
    rng = np.random.default_rng(0)
    n = 4000
    score = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < score).astype(float)
    feature = rng.normal(size=n)
    clean = compute_calibration_by_feature_heterogeneity(y, score, feature)
    y_nan = y.copy()
    y_nan[0] = np.nan
    with_nan = compute_calibration_by_feature_heterogeneity(y_nan, score, feature)
    assert np.isfinite(clean["noise_floor"])
    assert np.isfinite(with_nan["noise_floor"])
    assert abs(with_nan["noise_floor"] - clean["noise_floor"]) < 0.01


def test_spearman_bench_uses_the_live_threshold_name():
    """The bench writes _rank_stats.SPEARMAN_NJIT_MIN_N, which exists, and no longer touches the removed prediction_stability attribute (20-F25)."""
    assert hasattr(_rank_stats, "SPEARMAN_NJIT_MIN_N")
    bench = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "reporting" / "_benchmarks" / "bench_prediction_stability_spearman_iter83.py"
    text = bench.read_text(encoding="utf-8")
    assert "ps)._SPEARMAN_NJIT_MIN_N" not in text
    assert "rs.SPEARMAN_NJIT_MIN_N" in text


class _ImportanceModel:
    """A model whose only attribute is a feature-importance vector."""

    feature_importances_ = np.array([0.1, 0.7, 0.2])


def test_failed_shap_pair_choice_is_logged_as_a_warning(monkeypatch, caplog):
    """When the SHAP pair choice raises, the importance-based pair is used and a WARNING says so (20-F27)."""
    import mlframe.reporting.charts.shap_interactions as si

    def boom(*args, **kwargs):
        """Fail like a broken SHAP install."""
        raise RuntimeError("shap unavailable")

    monkeypatch.setattr(si, "shap_interaction_summary", boom)
    X = np.zeros((10, 3))
    with caplog.at_level(logging.WARNING, logger="mlframe.reporting.charts.pdp_2d"):
        pair = _default_pair(_ImportanceModel(), X, ["a", "b", "c"], 3)
    assert pair is not None
    assert any("SHAP pair selection failed" in r.getMessage() for r in caplog.records)
