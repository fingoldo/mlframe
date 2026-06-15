"""biz_value: adjusted R^2 recovers the true population R^2 where plain R^2 over-reports (qual-10).

Plain R^2 on the fitting sample is upward-biased -- adding junk predictors never lowers it, so it overstates the
honest goodness-of-fit whenever the predictor count ``p`` is non-trivial vs ``n``. ``fast_adjusted_r2_score`` applies
the Wherry / Ezekiel degrees-of-freedom penalty and tracks the true R^2 far better in that regime.

Ground truth: ``y = x0 + noise`` (x0, noise ~ N(0,1)) has population R^2 exactly 0.5 for the x0-only model; the
(p-1) extra noise predictors contribute nothing, so the honest fit stays 0.5. Measured here (7 seeds x the p/n>=0.1
cell set): mean |adjusted-true|=0.080 vs |plain-true|=0.138, adjusted wins 27/35 cells. Floors set ~10-15% below.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from mlframe.metrics.core import fast_adjusted_r2_score, fast_r2_score

TRUE_R2 = 0.5  # y = x0 + noise, both N(0,1) -> Var(signal)/Var(y) = 1/2


def _fit(n: int, p: int, seed: int):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0, 1, n)
    y = x0 + rng.normal(0, 1, n)
    X = np.column_stack([x0] + [rng.normal(0, 1, n) for _ in range(p - 1)])
    yp = LinearRegression().fit(X, y).predict(X)
    return y, yp


def test_biz_val_adjusted_r2_closer_to_true_than_plain_when_p_matters():
    """In the p/n>=0.1 regime, adjusted R^2 must be closer to the true 0.5 than plain R^2 on the MAJORITY of cells.
    Measured 27/35; floor 22/35. A regression that neutered the df-penalty drops this toward ~0/35."""
    combos = [(40, 8), (50, 10), (50, 20), (100, 20), (100, 30)]
    adj_win, cells = 0, 0
    plain_err, adj_err = [], []
    for (n, p) in combos:
        for seed in range(7):
            y, yp = _fit(n, p, seed)
            pe = abs(fast_r2_score(y, yp) - TRUE_R2)
            ae = abs(fast_adjusted_r2_score(y, yp, p) - TRUE_R2)
            plain_err.append(pe)
            adj_err.append(ae)
            cells += 1
            if ae < pe:
                adj_win += 1
    assert adj_win >= 22, f"adjusted-R2 won only {adj_win}/{cells} cells (floor 22)"
    # Mean abs error to truth: measured 0.138 -> 0.080; require >=25% reduction (floor below measured ~42%).
    assert np.mean(adj_err) <= 0.75 * np.mean(plain_err), (
        f"adjusted mean err {np.mean(adj_err):.4f} not <= 0.75 * plain {np.mean(plain_err):.4f}"
    )


def test_biz_val_adjusted_r2_no_harm_at_small_p_over_n():
    """At small p/n adjusted R^2 must converge back to plain R^2 (no harm). Measured both ~0.022; require the two
    means within 0.01 so the correction can never become the default-flip 'win' it is not."""
    combos = [(500, 5), (1000, 10), (2000, 5)]
    plain_err, adj_err = [], []
    for (n, p) in combos:
        for seed in range(7):
            y, yp = _fit(n, p, seed)
            plain_err.append(abs(fast_r2_score(y, yp) - TRUE_R2))
            adj_err.append(abs(fast_adjusted_r2_score(y, yp, p) - TRUE_R2))
    assert abs(np.mean(adj_err) - np.mean(plain_err)) < 0.01
