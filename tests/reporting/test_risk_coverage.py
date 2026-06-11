"""Unit tests for the risk-coverage (selective prediction) chart builder."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts.risk_coverage import (
    MAX_PLOT_POINTS,
    build_risk_coverage_spec,
    compute_risk_coverage,
)
from mlframe.reporting.spec import FigureSpec, LinePanelSpec


def _well_ranked_binary(n=4000, seed=0):
    """Confidence (|p-0.5|) ranks correctness: confident rows are right, near-0.5 rows are coin flips."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    conf = rng.uniform(0.0, 1.0, n)  # latent confidence
    # Probability the prediction is correct grows with conf.
    correct = rng.uniform(0, 1, n) < (0.45 + 0.54 * conf)
    pred = np.where(correct, y, 1 - y)
    # Score: predicted class with margin = conf/2 from 0.5.
    score = np.where(pred == 1, 0.5 + 0.5 * conf, 0.5 - 0.5 * conf)
    return y, score


def _random_confidence_binary(n=4000, seed=1):
    """Confidence independent of correctness: same accuracy but margin carries no information."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    correct = rng.uniform(0, 1, n) < 0.75
    pred = np.where(correct, y, 1 - y)
    conf = rng.uniform(0.0, 1.0, n)  # independent of `correct`
    score = np.where(pred == 1, 0.5 + 0.5 * conf, 0.5 - 0.5 * conf)
    return y, score


def test_compute_returns_curve_and_aurc():
    y, score = _well_ranked_binary()
    cov, acc, risk, aurc, full_risk, sig = compute_risk_coverage(y, score, task="binary")
    assert cov.shape == acc.shape == risk.shape
    assert cov[-1] == pytest.approx(1.0)
    assert np.all(np.diff(cov) > 0)
    assert np.allclose(risk, 1.0 - acc)
    assert 0.0 <= aurc <= 1.0
    assert sig is True
    # random-rejection reference = flat full risk integrated over [0,1].
    assert full_risk == pytest.approx(risk[-1])


def test_accuracy_rises_as_coverage_drops_when_well_ranked():
    y, score = _well_ranked_binary()
    cov, acc, risk, aurc, full_risk, sig = compute_risk_coverage(y, score, task="binary")
    # Accuracy on the most-confident 20% exceeds accuracy at full coverage.
    a20 = float(np.interp(0.2, cov, acc))
    assert a20 > acc[-1] + 0.05
    # AURC beats the flat random-rejection AURC (= full_risk).
    assert aurc < full_risk


def test_random_confidence_curve_is_flat():
    y, score = _random_confidence_binary()
    cov, acc, risk, aurc, full_risk, sig = compute_risk_coverage(y, score, task="binary")
    a20 = float(np.interp(0.2, cov, acc))
    # No selective gain: accuracy at 20% coverage ~= full accuracy (within noise).
    assert abs(a20 - acc[-1]) < 0.03
    assert abs(aurc - full_risk) < 0.02


def test_multiclass_top_prob_confidence():
    rng = np.random.default_rng(3)
    n, k = 3000, 4
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    # Inject signal toward the true class for a confident subset.
    strong = rng.uniform(0, 1, n) < 0.6
    logits[strong, y[strong]] += 3.0
    proba = np.exp(logits)
    proba /= proba.sum(axis=1, keepdims=True)
    cov, acc, risk, aurc, full_risk, sig = compute_risk_coverage(y, proba, task="multiclass")
    assert sig is True
    assert float(np.interp(0.3, cov, acc)) > acc[-1]


def test_regression_error_drops_as_coverage_drops():
    rng = np.random.default_rng(5)
    n = 4000
    yt = rng.normal(0, 1, n)
    conf = rng.uniform(0, 1, n)
    yp = yt + rng.normal(0, 1, n) * (1.0 - conf)  # low conf -> noisy pred
    cov, acc, risk, aurc, full_risk, sig = compute_risk_coverage(yt, yp, task="regression", confidence=conf)
    assert np.all(np.isnan(acc))
    e20 = float(np.interp(0.2, cov, risk))
    assert e20 < risk[-1]
    assert aurc < full_risk


def test_constant_confidence_flat_no_signal():
    rng = np.random.default_rng(7)
    n = 1000
    y = rng.integers(0, 2, n)
    score = np.full(n, 0.7)
    cov, acc, risk, aurc, full_risk, sig = compute_risk_coverage(y, score, task="binary")
    assert sig is False
    # Constant confidence: stable-sort keeps order, risk is a running mean of an unranked sequence; endpoint = full risk.
    assert risk[-1] == pytest.approx(full_risk)
    res = build_risk_coverage_spec(y, score, task="binary")
    assert "no ranking signal" in res.figure.panels[0][0].title


def test_nan_rows_dropped():
    y = np.array([0, 1, 1, 0, 1])
    score = np.array([0.1, np.nan, 0.9, 0.2, 0.8])
    cov, acc, risk, aurc, full_risk, sig = compute_risk_coverage(y, score, task="binary")
    # 4 finite rows remain; coverage grid has 4 steps.
    assert cov.size == 4


def test_empty_after_drop():
    y = np.array([0, 1])
    score = np.array([np.nan, np.nan])
    cov, acc, risk, aurc, full_risk, sig = compute_risk_coverage(y, score, task="binary")
    assert sig is False
    assert np.isnan(risk[0])


def test_tiny_n():
    y = np.array([0, 1, 1])
    score = np.array([0.2, 0.6, 0.9])
    cov, acc, risk, aurc, full_risk, sig = compute_risk_coverage(y, score, task="binary")
    assert cov.size == 3


def test_spec_has_random_reference_line():
    y, score = _well_ranked_binary()
    res = build_risk_coverage_spec(y, score, task="binary", model_label="m")
    assert isinstance(res.figure, FigureSpec)
    panel = res.figure.panels[0][0]
    assert isinstance(panel, LinePanelSpec)
    labels = panel.series_labels
    assert any("random rejection" in lbl for lbl in labels)
    # Flat reference line is constant across coverage.
    flat = panel.y[1]
    assert np.allclose(flat, flat[0])


def test_decimation_caps_plotted_points():
    rng = np.random.default_rng(9)
    n = 50_000
    y = rng.integers(0, 2, n)
    score = rng.uniform(0, 1, n)
    res = build_risk_coverage_spec(y, score, task="binary")
    panel = res.figure.panels[0][0]
    assert panel.x.size <= MAX_PLOT_POINTS
    # AURC still computed on the full curve.
    assert res.coverage.size == n


def test_regression_requires_confidence():
    with pytest.raises(ValueError):
        compute_risk_coverage(np.zeros(5), np.zeros(5), task="regression")
