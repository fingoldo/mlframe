"""Tests for decision-curve analysis (charts/decision_curve.py).

Covers: net-benefit formula parity vs a brute-force per-pt reference (the one-sort
vectorised sweep must match a naive threshold loop), spec shape / verdict content,
and biz_value -- a useful model's curve sits above treat-all / treat-none over a pt
range, a useless (random-score) model hugs treat-none and is flagged not-useful.
"""

from __future__ import annotations

import numpy as np

from mlframe.reporting.charts.decision_curve import (
    DecisionCurveResult,
    build_decision_curve_spec,
    compute_net_benefit,
)
from mlframe.reporting.spec import FigureSpec, LinePanelSpec


def _flat(fig: FigureSpec):
    """Helper: Flat."""
    return [p for row in fig.panels for p in row if p is not None]


def _separable(n=4000, sep=2.5, seed=0):
    """Helper: Separable."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    raw = rng.standard_normal(n) + sep * y
    score = 1.0 / (1.0 + np.exp(-raw))
    return y, score


def _brute_net_benefit(y, s, pt):
    """Naive per-pt reference: at each pt, flag score>=pt, count TP/FP directly."""
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    n = s.size
    out = np.empty_like(pt)
    for i, p in enumerate(pt):
        flag = s >= p
        tp = float((flag & (y == 1)).sum())
        fp = float((flag & (y == 0)).sum())
        out[i] = tp / n - (fp / n) * (p / (1.0 - p))
    return out


# ----------------------------------------------------------------------------
# Unit: net-benefit parity + edge cases
# ----------------------------------------------------------------------------


def test_net_benefit_matches_bruteforce():
    """Net benefit matches bruteforce."""
    y, s = _separable(n=3000, seed=1)
    pt, nb, _nb_all, _nb_none = compute_net_benefit(y, s, n_thresholds=120)
    ref = _brute_net_benefit(y, s, pt)
    assert np.allclose(nb, ref, atol=1e-12), np.max(np.abs(nb - ref))


def test_treat_all_and_none_references():
    """Treat all and none references."""
    y, s = _separable(n=2000, seed=2)
    pt, _nb, nb_all, nb_none = compute_net_benefit(y, s, n_thresholds=80)
    prevalence = float(np.mean(y))
    odds = pt / (1.0 - pt)
    assert np.allclose(nb_all, prevalence - (1.0 - prevalence) * odds, atol=1e-12)
    assert np.allclose(nb_none, 0.0)


def test_tie_at_threshold_flagged_positive():
    # All-equal scores: at pt < score every row flagged; nb must equal treat-all there.
    """Tie at threshold flagged positive."""
    y = np.array([0, 1, 0, 1, 1])
    s = np.full(5, 0.5)
    _pt, nb, nb_all, _nb_none = compute_net_benefit(y, s, pt_grid=np.array([0.3, 0.5, 0.7]))
    # pt=0.3 and 0.5: score(0.5) >= pt -> all flagged -> nb == treat-all.
    assert np.isclose(nb[0], nb_all[0], atol=1e-12)
    assert np.isclose(nb[1], nb_all[1], atol=1e-12)
    # pt=0.7: nothing flagged -> nb == 0.
    assert np.isclose(nb[2], 0.0, atol=1e-12)


def test_empty_input_returns_zero_curves():
    """Empty input returns zero curves."""
    _pt, nb, _nb_all, nb_none = compute_net_benefit([], [], n_thresholds=10)
    assert np.allclose(nb, 0.0) and np.allclose(nb_none, 0.0)


def test_spec_shape_and_verdict_fields():
    """Spec shape and verdict fields."""
    y, s = _separable()
    res = build_decision_curve_spec(y, s)
    assert isinstance(res, DecisionCurveResult)
    assert isinstance(res.figure, FigureSpec)
    panels = _flat(res.figure)
    assert len(panels) == 1 and isinstance(panels[0], LinePanelSpec)
    assert len(panels[0].y) == 3  # model + treat-all + treat-none
    assert res.pt.size == res.net_benefit.size


# ----------------------------------------------------------------------------
# biz_value: useful model beats both references; useless model does not
# ----------------------------------------------------------------------------


def test_biz_val_useful_model_beats_trivial_policies():
    """A strong classifier's net-benefit curve must sit clearly above treat-all
    AND treat-none over a pt range. Measured max gain ~0.18 on sep=3.0; floor 0.08
    (well below measured, above any FP-noise band)."""
    y, s = _separable(n=8000, sep=3.0, seed=7)
    res = build_decision_curve_spec(y, s)
    assert res.useful, "strong model must be flagged useful"
    assert res.best_pt_advantage >= 0.08, res.best_pt_advantage
    # Somewhere the model strictly tops both references.
    ref_best = np.maximum(res.treat_all, res.treat_none)
    assert np.any(res.net_benefit > ref_best + 0.05)


def test_biz_val_useless_model_hugs_treat_none():
    """A random-score model carries no information: its net benefit must hug
    treat-none (|NB| small) and never beat treat-all by a meaningful margin, so
    the verdict is not-useful. Pins the DCA's ability to reject a useless model."""
    rng = np.random.default_rng(11)
    n = 8000
    y = rng.integers(0, 2, n)
    s = rng.random(n)  # independent of y
    res = build_decision_curve_spec(y, s)
    assert not res.useful, "random-score model must not be flagged useful"
    # A useless model never clears the upper envelope of treat-all / treat-none by a meaningful margin: at low pt it
    # merely coincides with treat-all, at high pt with treat-none. Best advantage stays in the FP-noise band.
    assert res.best_pt_advantage < 0.02, res.best_pt_advantage
    ref_best = np.maximum(res.treat_all, res.treat_none)
    assert np.nanmax(res.net_benefit - ref_best) < 0.02
