"""dro_reweight_fit accepts a classifier with more than two classes when loss_fn takes the probability matrix."""
import numpy as np
from sklearn.linear_model import LogisticRegression

from mlframe.data_valuation._adversarial_reweighting import dro_reweight_fit


def test_three_class_logistic_regression_runs():
    """predict_proba of shape (n, 3) was written into a 1-D buffer and raised a broadcast ValueError."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((150, 3))
    y = rng.integers(0, 3, 150)

    def fit_fn(Xt, yt, wt):
        """Weighted multinomial logistic regression."""
        return LogisticRegression(max_iter=200).fit(Xt, yt, sample_weight=wt)

    def loss_fn(y_true, proba):
        """Per-sample cross-entropy of the true class."""
        return -np.log(np.clip(proba[np.arange(len(y_true)), y_true.astype(int)], 1e-9, 1.0))

    _model, w, info = dro_reweight_fit(fit_fn, loss_fn, X, y, n_rounds=2, n_splits=3, rng=np.random.default_rng(0))
    assert w.shape == (150,) and np.isfinite(w).all() and len(info["avg_loss_history"]) == 2
