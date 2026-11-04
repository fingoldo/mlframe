"""A training fold holding a single class must give a constant probability, not an IndexError on the missing second column."""
import numpy as np

from mlframe.feature_engineering.transformer.aux_mlp import _fit_aux_mlp, _predict_aux_mlp


def test_single_class_fold_scores_as_that_class():
    """All-zero targets: the MLP has one class, so every row's positive probability is 0."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 3)).astype(np.float32)
    model, scaler = _fit_aux_mlp(X, np.zeros(60, dtype=np.float32), task="binary", seed=1, hidden_size=4, max_iter=20)
    proba, logit = _predict_aux_mlp(model, scaler, X[:5], "binary")
    assert proba.shape == (5,) and np.all(proba == 0.0) and np.isfinite(logit).all()
