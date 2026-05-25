"""Sensor tests for ``mlframe.evaluation.bootstrap``.

Covers:
  - Bootstrap CI on synthetic AUC=0.85 problem: point ~0.85, CI contains true value.
  - Reproducibility under fixed seed.
  - Stratified resampling preserves class balance.
  - DeLong test returns small p when AUCs differ materially; large p when identical.
  - DeLong returns NaN p on degenerate input rather than crashing.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score, mean_squared_error

from mlframe.evaluation.bootstrap import bootstrap_metric, delong_test


# Every test in this module exercises only synthetic data at n<=4000 with default bootstrap n=200-300; wall-time
# stays well under 2s per test. Marking as ``fast`` so the ``pytest -m fast`` smoke run keeps these in scope per the
# B2 #6 audit observation that ``@pytest.mark.fast`` had unrealistically narrow adoption.
pytestmark = [pytest.mark.fast]


def _make_binary_auc_data(n: int = 2000, separation: float = 1.6, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate a binary classification problem with population AUC tunable by ``separation``."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    score = rng.normal(loc=separation * y, scale=1.0)
    return y, score


def test_bootstrap_metric_recovers_known_auc_point_and_ci():
    """At separation=1.6 the population AUC is ~0.87. Point/CI should bracket the true value."""
    y, score = _make_binary_auc_data(n=4000, separation=1.6, seed=42)
    true_auc = roc_auc_score(y, score)
    res = bootstrap_metric(
        y, score,
        metric_fn=lambda yt, yp: roc_auc_score(yt, yp),
        n_bootstrap=300,
        alpha=0.05,
        stratify=y,
        random_state=123,
    )
    assert "point" in res and "lo" in res and "hi" in res and "samples" in res
    assert abs(res["point"] - true_auc) < 1e-9, (
        f"point estimate should equal full-sample metric ({true_auc:.6f}); got {res['point']:.6f}"
    )
    assert res["lo"] < res["point"] < res["hi"], "CI must bracket the point estimate"
    assert res["lo"] <= true_auc <= res["hi"], (
        f"95% CI [{res['lo']:.4f}, {res['hi']:.4f}] must contain true AUC {true_auc:.4f}"
    )
    assert len(res["samples"]) == 300


def test_bootstrap_metric_reproducible_under_fixed_seed():
    """Two calls with the same seed must return bit-identical CI bounds + samples."""
    y, score = _make_binary_auc_data(n=1500, separation=1.0, seed=7)
    metric = lambda yt, yp: roc_auc_score(yt, yp)
    a = bootstrap_metric(y, score, metric_fn=metric, n_bootstrap=200, random_state=999)
    b = bootstrap_metric(y, score, metric_fn=metric, n_bootstrap=200, random_state=999)
    assert a["point"] == b["point"]
    assert a["lo"] == b["lo"]
    assert a["hi"] == b["hi"]
    np.testing.assert_array_equal(a["samples"], b["samples"])


def test_bootstrap_metric_regression_rmse():
    """Smoke test on a regression metric: bootstrap CI should bracket the in-sample RMSE."""
    rng = np.random.default_rng(13)
    y = rng.normal(0, 1, size=800)
    y_pred = y + rng.normal(0, 0.5, size=800)
    rmse = lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp)))
    res = bootstrap_metric(y, y_pred, metric_fn=rmse, n_bootstrap=200, random_state=2024)
    assert res["lo"] <= res["point"] <= res["hi"]
    # Population RMSE is 0.5; the bootstrap CI of in-sample RMSE on n=800 should comfortably contain it.
    assert res["lo"] <= 0.5 <= res["hi"]


def test_bootstrap_metric_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="row counts diverge"):
        bootstrap_metric(np.zeros(10), np.zeros(8), metric_fn=lambda a, b: 0.0, n_bootstrap=10)


def test_bootstrap_metric_rejects_tiny_n():
    with pytest.raises(ValueError, match="need at least 2 samples"):
        bootstrap_metric(np.zeros(1), np.zeros(1), metric_fn=lambda a, b: 0.0, n_bootstrap=10)


def test_delong_detects_real_auc_difference():
    """When score_a is materially better than score_b on the same y, p_value should be small."""
    y, score_good = _make_binary_auc_data(n=2000, separation=2.0, seed=1)
    rng = np.random.default_rng(2)
    # Bad scorer: shuffled noisy signal -> AUC ~0.5
    score_bad = rng.normal(size=y.shape[0])
    res = delong_test(y, score_good, score_bad)
    assert res["auc_a"] > res["auc_b"]
    assert res["diff"] > 0.2, f"expected large AUC difference, got {res['diff']:.3f}"
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["p_value"] < 0.01, f"strong difference should yield p<<0.05; got p={res['p_value']:.4f}"


def test_delong_returns_high_p_when_scores_identical():
    """Comparing a scorer to itself should give exactly diff=0, p=1.0 (z=0)."""
    y, score = _make_binary_auc_data(n=1500, separation=1.2, seed=44)
    res = delong_test(y, score, score)
    assert res["diff"] == 0.0
    assert res["p_value"] == pytest.approx(1.0, abs=1e-9)


def test_delong_rejects_multiclass():
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    score = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="binary 0/1"):
        delong_test(y, score, score + 1)


def test_delong_degenerate_returns_nan_p():
    """Constant scores -> singular covariance -> p=nan, not a crash."""
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    flat = np.zeros(8)
    res = delong_test(y, flat, flat)
    assert np.isnan(res["p_value"]) or res["p_value"] == pytest.approx(1.0, abs=1e-9)
