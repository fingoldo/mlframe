"""API16: EstimatorWithEarlyStopping must honour its split params for any eval_set-supporting
estimator, not only CatBoost. Pre-fix, a non-CatBoost estimator was fit on the FULL data and the
test_size/stratify/shuffle/random_state params were silently dead."""

import numpy as np

from mlframe.estimators.base import EstimatorWithEarlyStopping


class _EvalSetRecorder:
    """Minimal sklearn-like estimator whose fit accepts eval_set and records what it received."""

    def __init__(self):
        self.seen_eval_set = None
        self.seen_train_rows = None

    def get_params(self, deep=True):
        """Returns ``{}``."""
        return {}

    def set_params(self, **params):
        """No-op set_params stub; returns self unchanged (satisfies the sklearn fit/set_params contract without doing real work)."""
        return self

    def fit(self, X, y, eval_set=None):
        """Performs 2 setup steps, then returns self unchanged."""
        self.seen_eval_set = eval_set
        self.seen_train_rows = len(X)
        return self

    def predict(self, X):
        """Returns ``np.zeros(len(X))``."""
        return np.zeros(len(X))


class _NoEvalSet:
    """Groups tests covering NoEvalSet."""
    def __init__(self):
        self.seen_train_rows = None

    def get_params(self, deep=True):
        """Returns ``{}``."""
        return {}

    def set_params(self, **params):
        """No-op set_params stub; returns self unchanged (satisfies the sklearn fit/set_params contract without doing real work)."""
        return self

    def fit(self, X, y):
        """Performs 1 setup step, then returns self unchanged."""
        self.seen_train_rows = len(X)
        return self

    def predict(self, X):
        """Returns ``np.zeros(len(X))``."""
        return np.zeros(len(X))


def test_eval_set_estimator_receives_validation_split():
    """Eval set estimator receives validation split."""
    n = 200
    X = np.random.RandomState(0).normal(size=(n, 3))
    y = np.arange(n) % 2
    base = _EvalSetRecorder()
    wrapper = EstimatorWithEarlyStopping(base_estimator=base, test_size=0.25, random_state=0)
    wrapper.fit(X, y)
    rec = wrapper.fitted_estimator_
    assert rec.seen_eval_set is not None, "eval_set was not passed (split params were dead pre-fix)"
    assert rec.seen_train_rows == 150, "train fold should be 75% of rows, not the full data"
    assert len(rec.seen_eval_set[0][0]) == 50, "validation fold should be 25% of rows"


def test_non_eval_set_estimator_fits_full_data_without_crash():
    """Non eval set estimator fits full data without crash."""
    n = 60
    X = np.random.RandomState(1).normal(size=(n, 2))
    y = np.arange(n) % 2
    wrapper = EstimatorWithEarlyStopping(base_estimator=_NoEvalSet(), test_size=0.2)
    wrapper.fit(X, y)
    assert wrapper.fitted_estimator_.seen_train_rows == n, "no-eval-set estimator fits full data"
