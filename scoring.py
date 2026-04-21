"""Scoring utilities salvaged from the legacy ``Models`` module.

Contains loss functions, scorers, a log-uniform distribution helper for
RandomizedSearchCV, and a proxy for scoring probabilistic classifier outputs.
"""

import numpy as np
from scipy.stats import uniform
from sklearn.metrics import make_scorer


def rmse_loss(y_true, y_pred):
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


rmse_score = make_scorer(rmse_loss, greater_is_better=False)


def rmsle_loss(y_true, y_pred):
    """Root mean squared logarithmic error. Negative predictions are clipped to 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(np.clip(y_pred, 0, None)), 2)))


rmsle_score = make_scorer(rmsle_loss, greater_is_better=False)


class log_uniform:
    """Log-uniform continuous distribution over ``[base**a, base**b]``.

    Compatible with scipy.stats random-variable interface expected by
    ``RandomizedSearchCV``.
    """

    def __init__(self, a=-1, b=0, base=10):
        self.loc = a
        self.scale = b - a
        self.base = base

    def rvs(self, size=None, random_state=None):
        myuniform = uniform(loc=self.loc, scale=self.scale)
        if size is None:
            return np.power(self.base, myuniform.rvs(random_state=random_state))
        return np.power(self.base, myuniform.rvs(size=size, random_state=random_state))


def ProbaScoreProxy(y_true, y_probs, class_idx, proxied_func, **kwargs):
    """Wrap a scalar-target scorer so it can consume probability matrices.

    Passes column ``class_idx`` of ``y_probs`` to ``proxied_func(y_true, ...)``.
    """
    return proxied_func(y_true, y_probs[:, class_idx], **kwargs)
