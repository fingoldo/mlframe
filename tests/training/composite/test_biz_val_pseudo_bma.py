"""Unit + biz_value tests for pseudo-BMA composite-ensemble weighting (``composite/_pseudo_bma.py``)."""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite._pseudo_bma import blend, pseudo_bma_weights


def _make_components(rng, n, good_noise, bad_noise, n_bad=2):
    """One genuinely good component (small residual noise) + ``n_bad`` noisy components tracking y loosely."""
    x = rng.normal(size=n)
    y = 2.0 * x + 1.0
    cols = [y + rng.normal(scale=good_noise, size=n)]
    for _ in range(n_bad):
        cols.append(y + rng.normal(scale=bad_noise, size=n))
    return np.column_stack(cols), y


# --------------------------------------------------------------------------- unit


def test_weights_simplex():
    rng = np.random.default_rng(0)
    P, y = _make_components(rng, 500, 0.2, 2.0)
    w = pseudo_bma_weights(P, y)
    assert w.shape == (3,)
    assert np.all(w >= 0.0)
    assert w.sum() == pytest.approx(1.0)


def test_better_component_gets_largest_weight():
    rng = np.random.default_rng(1)
    P, y = _make_components(rng, 800, 0.15, 2.5)
    w = pseudo_bma_weights(P, y)
    assert int(np.argmax(w)) == 0, "the low-noise component must win the largest weight"
    assert w[0] > 0.5


def test_identical_components_split_evenly():
    rng = np.random.default_rng(2)
    x = rng.normal(size=600)
    y = x + rng.normal(scale=0.5, size=600)
    col = y + rng.normal(scale=0.3, size=600)
    P = np.column_stack([col, col, col])  # three identical columns
    w = pseudo_bma_weights(P, y)
    assert np.allclose(w, 1.0 / 3.0, atol=1e-9)


def test_single_component_and_degenerate():
    rng = np.random.default_rng(3)
    y = rng.normal(size=50)
    # Single component.
    w1 = pseudo_bma_weights(y[:, None] + 0.1, y)
    assert w1.shape == (1,) and w1[0] == pytest.approx(1.0)
    # All-equal (perfect) predictions across 3 components, n < K guard not tripped: sigma floored, weights stay finite/simplex.
    P = np.column_stack([y, y, y])
    w = pseudo_bma_weights(P, y)
    assert np.all(np.isfinite(w)) and w.sum() == pytest.approx(1.0)
    # n < K: 2 rows, 3 components.
    w_small = pseudo_bma_weights(np.ones((2, 3)), np.array([1.0, 1.0]))
    assert w_small.sum() == pytest.approx(1.0) and np.all(w_small >= 0)


def test_bad_inputs_raise():
    y = np.zeros(5)
    with pytest.raises(ValueError):
        pseudo_bma_weights(np.zeros(5), y)  # 1-D preds
    with pytest.raises(ValueError):
        pseudo_bma_weights(np.zeros((4, 2)), y)  # row mismatch
    with pytest.raises(ValueError):
        pseudo_bma_weights(np.full((5, 2), np.nan), y)  # non-finite
    with pytest.raises(ValueError):
        pseudo_bma_weights(np.zeros((5, 2)), y, quantile=1.5)


def test_quantile_pinball_scoring():
    rng = np.random.default_rng(4)
    n = 600
    y = rng.normal(size=n)
    tau = 0.9
    true_q = np.quantile(y, tau)
    good = np.full(n, true_q)  # correct quantile predictor
    bad = np.full(n, np.quantile(y, 0.2))  # wrong level
    P = np.column_stack([good, bad])
    w = pseudo_bma_weights(P, y, quantile=tau)
    assert int(np.argmax(w)) == 0, "the component at the correct quantile level must win"


def test_blend_matches_manual():
    P = np.array([[1.0, 3.0], [2.0, 4.0]])
    w = np.array([0.25, 0.75])
    assert np.allclose(blend(P, w), P @ w)
    with pytest.raises(ValueError):
        blend(P, np.array([1.0]))


def test_bb_draws_stabilise_weight_variance():
    """BB-pseudo-BMA weights vary LESS across seeds than point pseudo-BMA on a noisy small-n set (its whole purpose)."""
    n = 60  # small + noisy -> point pseudo-BMA is seed-sensitive
    point_w0, bb_w0 = [], []
    for seed in range(25):
        rng = np.random.default_rng(1000 + seed)
        P, y = _make_components(rng, n, 0.9, 1.1)  # good barely better than bad -> unstable
        point_w0.append(pseudo_bma_weights(P, y)[0])
        bb_w0.append(pseudo_bma_weights(P, y, bb_draws=400, random_state=7)[0])
    var_point = float(np.var(point_w0))
    var_bb = float(np.var(bb_w0))
    assert var_bb < var_point, f"BB variance {var_bb:.4g} should be < point variance {var_point:.4g}"


# --------------------------------------------------------------------------- biz_value


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _gen_five(n, seed):
    """2 genuinely good components (moderate noise) + 3 noisy ones; small train makes single-best-by-train overfit."""
    r = np.random.default_rng(seed)
    x = r.normal(size=n)
    y = 3.0 * x - 0.5
    good1 = y + r.normal(scale=0.6, size=n)
    good2 = y + r.normal(scale=0.6, size=n)
    bad1 = y + r.normal(scale=2.0, size=n)
    bad2 = y + r.normal(scale=2.0, size=n)
    bad3 = y + r.normal(scale=2.0, size=n)
    return np.column_stack([good1, good2, bad1, bad2, bad3]), y


def test_biz_val_pseudo_bma_blend_beats_equal_and_single_best_overfit():
    """Mean held-out RMSE of the pseudo-BMA blend beats BOTH equal-weight (dragged by bad components) and single-best-by-train-RMSE (winner-take-all overfit)."""
    K = 5
    bma_rmses, eq_rmses, single_rmses = [], [], []
    for seed in range(20):
        P_tr, y_tr = _gen_five(200, seed)  # small train -> best-by-train is a high-variance pick
        P_te, y_te = _gen_five(3000, 1000 + seed)

        w = pseudo_bma_weights(P_tr, y_tr, bb_draws=200, random_state=0)
        bma_rmses.append(_rmse(blend(P_te, w), y_te))
        eq_rmses.append(_rmse(blend(P_te, np.full(K, 1.0 / K)), y_te))
        train_rmse = np.array([_rmse(P_tr[:, k], y_tr) for k in range(K)])
        single_rmses.append(_rmse(P_te[:, int(np.argmin(train_rmse))], y_te))

    m_bma, m_eq, m_single = float(np.mean(bma_rmses)), float(np.mean(eq_rmses)), float(np.mean(single_rmses))
    assert m_bma < m_eq * 0.85, f"pseudo-BMA mean RMSE {m_bma:.4f} must beat equal-weight {m_eq:.4f} by >=15%"
    assert m_bma < m_single * 0.97, f"pseudo-BMA mean RMSE {m_bma:.4f} must beat single-best-train {m_single:.4f}"


def test_biz_val_bb_no_worse_than_point_and_more_stable():
    """BB-pseudo-BMA blend RMSE is no worse than point pseudo-BMA (within margin) while being more stable across seeds."""
    def gen(n, seed):
        # Two NEAR-TIED good components (same expected skill) + one bad: point pseudo-BMA's weight on component 0 swings
        # seed-to-seed on estimation noise (winner-take-all via exp on a tiny, noisy elpd gap), so point weights are
        # UNSTABLE -- exactly the regime where BB (averaging elpd over Dirichlet-reweighted draws) reduces the swing.
        r = np.random.default_rng(seed)
        x = r.normal(size=n)
        y = 1.5 * x + 0.2
        good1 = y + r.normal(scale=0.6, size=n)
        good2 = y + r.normal(scale=0.6, size=n)
        bad = y + r.normal(scale=1.6, size=n)
        return np.column_stack([good1, good2, bad]), y

    point_rmses, bb_rmses, point_w0, bb_w0 = [], [], [], []
    for seed in range(15):
        P_tr, y_tr = gen(120, 10 + seed)  # small train -> noisy weights
        P_te, y_te = gen(2000, 500 + seed)
        wp = pseudo_bma_weights(P_tr, y_tr)
        wb = pseudo_bma_weights(P_tr, y_tr, bb_draws=300, random_state=3)
        point_rmses.append(_rmse(blend(P_te, wp), y_te))
        bb_rmses.append(_rmse(blend(P_te, wb), y_te))
        point_w0.append(wp[0])
        bb_w0.append(wb[0])

    mean_point = float(np.mean(point_rmses))
    mean_bb = float(np.mean(bb_rmses))
    assert mean_bb <= mean_point * 1.03, f"BB mean RMSE {mean_bb:.4f} must be no worse than point {mean_point:.4f} (+3%)"
    assert np.var(bb_w0) < np.var(point_w0), "BB weights must be more stable (lower variance) across seeds"
