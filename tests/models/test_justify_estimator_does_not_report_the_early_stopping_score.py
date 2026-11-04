"""justify_estimator's expected score for a CatBoost-like estimator is not inflated by the eval split it early-stopped on."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from mlframe.models.tuning_rules import justify_estimator


class _MemorisesItsEvalSet(RegressorMixin, BaseEstimator):
    """Predicts the eval targets exactly for the eval rows it saw at fit time and the training mean elsewhere, like a model that early-stopped on them."""

    def fit(self, X, y, eval_set=None, plot=False, verbose=False, early_stopping_rounds=None):
        """Remember the training mean and, when given, the eval rows."""
        self.mean_ = float(np.mean(y))
        self.eval_ = None if eval_set is None else (np.asarray(eval_set[0]), np.asarray(eval_set[1]))
        return self

    def predict(self, X):
        """Eval rows get their true target, everything else the mean."""
        X = np.asarray(X)
        if self.eval_ is not None and X.shape == self.eval_[0].shape and np.array_equal(X, self.eval_[0]):
            return self.eval_[1]
        return np.full(len(X), self.mean_)


_MemorisesItsEvalSet.__module__ = "catboost.fake"


def test_expected_score_is_not_the_optimistic_eval_split_score():
    """The eval-split score is 1.0 by construction; the out-of-sample CV mean is about 0, and the returned score must be the lower."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 3))
    y = rng.standard_normal(300)
    est, score = justify_estimator(_MemorisesItsEvalSet(), X, y, cv=3, scoring="r2", min_score=-1.0, test_size=0.2, random_state=1)
    assert est is not None
    assert score < 0.5, score
