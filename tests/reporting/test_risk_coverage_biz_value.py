"""biz_value: confidence-ranked abstention buys real selective gain; random confidence does not.

Pins the headline contract: a well-ranked confidence signal materially lifts accuracy at 80% coverage over full
coverage (and beats the random-rejection AURC), while a confidence signal independent of correctness leaves the
curve flat (no selective gain). A regression silently breaking the sort / cumulative pass trips these win floors.
"""

from __future__ import annotations

import numpy as np

from mlframe.reporting.charts.risk_coverage import compute_risk_coverage


def _well_ranked_binary(n=8000, seed=0):
    """Helper: Well ranked binary."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    conf = rng.uniform(0.0, 1.0, n)
    correct = rng.uniform(0, 1, n) < (0.45 + 0.54 * conf)
    pred = np.where(correct, y, 1 - y)
    score = np.where(pred == 1, 0.5 + 0.5 * conf, 0.5 - 0.5 * conf)
    return y, score


def _random_confidence_binary(n=8000, seed=1):
    """Helper: Random confidence binary."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    correct = rng.uniform(0, 1, n) < 0.75
    pred = np.where(correct, y, 1 - y)
    conf = rng.uniform(0.0, 1.0, n)
    score = np.where(pred == 1, 0.5 + 0.5 * conf, 0.5 - 0.5 * conf)
    return y, score


def test_biz_val_risk_coverage_well_ranked_selective_gain():
    """Well-ranked confidence: accuracy@80% >= accuracy@100% + 0.05 AND AURC < random-rejection AURC."""
    y, score = _well_ranked_binary()
    cov, acc, _risk, aurc, full_risk, sig = compute_risk_coverage(y, score, task="binary")
    acc80 = float(np.interp(0.8, cov, acc))
    acc100 = float(acc[-1])
    assert sig is True
    assert acc80 - acc100 >= 0.05, f"selective gain {acc80 - acc100:.3f} below 0.05 floor"
    assert aurc < full_risk - 0.02, f"AURC {aurc:.3f} not below random {full_risk:.3f}"


def test_biz_val_risk_coverage_random_confidence_flat():
    """Random confidence: accuracy@80% ~= accuracy@100% (|gain| < 0.02), AURC ~= random AURC."""
    y, score = _random_confidence_binary()
    cov, acc, _risk, aurc, full_risk, _sig = compute_risk_coverage(y, score, task="binary")
    acc80 = float(np.interp(0.8, cov, acc))
    acc100 = float(acc[-1])
    assert abs(acc80 - acc100) < 0.02, f"random confidence leaked a gain of {acc80 - acc100:.3f}"
    assert abs(aurc - full_risk) < 0.02


def test_biz_val_risk_coverage_well_ranked_beats_random_on_aurc():
    """Same accuracy, ranked confidence -> strictly lower AURC than independent confidence."""
    y_w, s_w = _well_ranked_binary()
    y_r, s_r = _random_confidence_binary()
    _, _, _, aurc_w, _, _ = compute_risk_coverage(y_w, s_w, task="binary")
    _, _, _, aurc_r, _, _ = compute_risk_coverage(y_r, s_r, task="binary")
    assert aurc_w < aurc_r - 0.03, f"ranked AURC {aurc_w:.3f} not below random {aurc_r:.3f}"
