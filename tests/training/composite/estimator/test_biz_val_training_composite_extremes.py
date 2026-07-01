"""biz_value: GPD tail extrapolation beats the empirical quantile on heavy tails.

On a genuinely heavy-tailed residual distribution the empirical quantile of a
finite sample SATURATES at the largest observed value and under-estimates the
true 0.999 quantile. The POT/GPD extrapolation in ``TailCompositeEstimator``
extends beyond the sample max and lands measurably closer to the true tail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from mlframe.training.composite.extremes import TailCompositeEstimator


class _ZeroBase:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(X.shape[0], dtype=float)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **kw):
        return self


def _frame(n):
    return pd.DataFrame({"f0": np.linspace(0, 1, n), "base": np.zeros(n)})


def test_biz_val_extremes_gpd_beats_empirical_999_pareto():
    """GPD deep-tail quantile rel-error << empirical rel-error on a Pareto tail.

    At q=0.9999 with n=5000 the empirical quantile is past the sample's reliable
    resolution: only ~0-1 points lie beyond, so it saturates near the sample max
    and badly under-shoots the true tail (measured rel-err ~0.54). The GPD
    extrapolates past the max (measured rel-err ~0.26, a ~2.07x win). Floored at
    1.5x, averaged over 12 seeds to absorb the high tail-estimation variance.
    """
    # Pareto(b) survival 1-F(x) = x**-b for x>=1; b=1.5 -> heavy tail. True
    # q-quantile = (1-q)**(-1/b).
    b = 1.5
    q = 0.9999
    true_q = (1.0 - q) ** (-1.0 / b)

    n = 5000
    emp_errs, gpd_errs = [], []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        y = stats.pareto.rvs(b=b, size=n, random_state=rng)  # residuals = y
        est = TailCompositeEstimator(
            base_estimator=_ZeroBase(),
            transform_name="diff",
            base_column="base",
            threshold_pct=0.90,
            two_sided=False,  # one-sided upper Pareto tail
        )
        est.fit(_frame(n), y)
        assert est.gpd_fitted_

        gpd_q = est.predict_tail_quantile(_frame(1), q)[0]
        emp_q = float(np.quantile(y, q))  # naive empirical estimate
        emp_errs.append(abs(emp_q - true_q) / true_q)
        gpd_errs.append(abs(gpd_q - true_q) / true_q)

    mean_emp = float(np.mean(emp_errs))
    mean_gpd = float(np.mean(gpd_errs))
    # The empirical estimate must materially under-shoot the true tail.
    assert mean_emp > 0.3, f"empirical err unexpectedly low: {mean_emp}"
    # GPD must be at least 1.5x closer to the truth than the empirical estimate.
    assert mean_gpd < mean_emp / 1.5, (
        f"GPD rel-err {mean_gpd:.3f} not < empirical {mean_emp:.3f}/1.5"
    )


def test_biz_val_extremes_gpd_exceeds_sample_max():
    """The GPD 0.999 quantile can exceed the largest observed residual.

    This is the core EVT advantage the empirical quantile structurally cannot
    deliver (it is capped at the sample max).
    """
    rng = np.random.default_rng(11)
    n = 3000
    y = stats.pareto.rvs(b=1.5, size=n, random_state=rng)
    est = TailCompositeEstimator(
        base_estimator=_ZeroBase(), transform_name="diff",
        base_column="base", threshold_pct=0.90, two_sided=False,
    )
    est.fit(_frame(n), y)
    sample_max = float(np.max(y))
    gpd_q = est.predict_tail_quantile(_frame(1), 0.99995)[0]
    assert gpd_q > sample_max, f"GPD {gpd_q} did not exceed max {sample_max}"
