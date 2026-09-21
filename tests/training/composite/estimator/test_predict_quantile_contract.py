"""``predict_quantile`` returns ordered quantiles on every row, and a real interval on rows the model cannot serve.

Fallback rows (base out of domain, deep OOD) got the train median in every quantile column, a zero-width interval where the
uncertainty is highest; an inverse that multiplies T by a base factor that turns negative (``centered_ratio`` below ``-c``)
reversed the column order with no warning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.transforms import get_transform, list_transforms

_ALPHAS = [0.1, 0.5, 0.9]


class _QuantileInner(BaseEstimator, RegressorMixin):
    """Predicts T quantiles as mean + z_alpha * std of the training T, the same on every row."""

    def fit(self, X, y):
        """Remember the T mean and spread."""
        t = np.asarray(y, dtype=np.float64)
        self.mu_, self.sd_ = float(np.mean(t)), float(np.std(t)) or 1.0
        return self

    def predict(self, X):
        """The T mean."""
        return np.full(len(X), self.mu_)

    def predict_quantile(self, X, alpha):
        """Columns at each alpha (a scalar alpha gives one column)."""
        from scipy.stats import norm

        a = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
        out = np.tile(self.mu_ + norm.ppf(a) * self.sd_, (len(X), 1))
        return out[:, 0] if np.isscalar(alpha) else out


def _frame(n: int = 600, seed: int = 0):
    """A positive base and a y that grows with it, plus a group column."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(5.0, 20.0, n)
    X = pd.DataFrame({"base": base, "base2": rng.uniform(1.0, 5.0, n), "feat": rng.normal(size=n), "grp": rng.integers(0, 3, n)})
    return X, 2.0 * base + rng.normal(0.0, 1.0, n) + 5.0


def _fitted(name: str):
    """A wrapper over the quantile inner for transform ``name``."""
    X, y = _frame()
    t = get_transform(name)
    kw: dict = {"base_column": "base"}
    if t.requires_groups:
        kw["group_column"] = "grp"
    if t.n_bases > 1:
        kw["base_columns"] = ("base", "base2")
    return CompositeTargetEstimator(base_estimator=_QuantileInner(), transform_name=name, **kw).fit(X, y), X


_QUANTILE_OK = [n for n in list_transforms() if n not in {"ratio", "logratio", "reciprocal_residual"}]  # these raise by contract


@pytest.mark.parametrize("name", _QUANTILE_OK)
def test_quantile_columns_are_ordered_on_every_row(name: str):
    """Quantile columns are non-decreasing in alpha, including rows whose base lies far outside the train range."""
    est, X = _fitted(name)
    X_pred = X.iloc[:40].copy()
    X_pred.loc[X_pred.index[:10], "base"] = -50.0  # below any train base, and below -c for centered_ratio
    try:
        q = np.asarray(est.predict_quantile(X_pred, _ALPHAS))
    except NotImplementedError:
        pytest.skip(f"{name}: predict_quantile is not supported by contract")
    finite = np.all(np.isfinite(q), axis=1)
    assert np.all(np.diff(q[finite], axis=1) >= -1e-12), f"{name}: crossed quantiles on {int((np.diff(q[finite], axis=1) < 0).any(axis=1).sum())} rows"


def test_a_row_the_model_cannot_serve_gets_a_real_interval():
    """A NaN base falls back to the train-y quantile at each alpha: q10 < q90, not the median twice."""
    est, X = _fitted("linear_residual")
    X_pred = X.iloc[:5].copy()
    X_pred.loc[X_pred.index[0], "base"] = np.nan
    q = np.asarray(est.predict_quantile(X_pred, _ALPHAS))
    assert q[0, 0] < q[0, 2], f"fallback row interval collapsed: {q[0]}"
