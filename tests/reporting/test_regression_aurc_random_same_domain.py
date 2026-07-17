"""Regression: the AURC random reference must integrate over the SAME coverage domain as AURC itself.

``aurc`` integrates running-risk via ``np.trapezoid`` over coverage in [1/n, 1]. Pre-fix ``aurc_random`` was set to
``full_risk`` (the integral of the constant full_risk over [0, 1]), so on random-quality scores ``aurc`` (~full_risk
over [1/n,1] = full_risk*(1-1/n)) was biased BELOW ``aurc_random``. Post-fix the reference uses the same domain and
the two match on random scores.
"""

import numpy as np

from mlframe.reporting.charts.risk_coverage import build_risk_coverage_spec


def test_aurc_matches_random_on_random_quality_scores():
    """Aurc matches random on random quality scores."""
    rng = np.random.default_rng(0)
    n = 500
    y_true = (rng.random(n) < 0.5).astype(int)
    # Confidence scores independent of correctness -> selective prediction gives no gain: aurc ~= aurc_random.
    y_score = rng.random((n, 2))
    y_score /= y_score.sum(axis=1, keepdims=True)

    res = build_risk_coverage_spec(y_true, y_score, task="binary")

    # On random scores the running risk is approximately flat, so the two integrals coincide over the same domain.
    assert np.isclose(res.aurc, res.aurc_random, rtol=0.05, atol=0.01), (res.aurc, res.aurc_random)

    # Pin the domain-consistency directly: aurc_random must be full_risk over [1/n, 1], not full_risk over [0, 1].
    full_risk = float(res.risk[-1])
    coverage_span = float(res.coverage[-1] - res.coverage[0])
    assert np.isclose(res.aurc_random, full_risk * coverage_span, rtol=0, atol=1e-9)
