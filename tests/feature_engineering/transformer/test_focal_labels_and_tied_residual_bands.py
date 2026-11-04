"""Focal objectives see 0/1 labels whatever the caller coded, and the residual bands stay non-empty when residuals tie."""
import numpy as np

from mlframe.feature_engineering.transformer.focal_lgb import _fit_focal_lgb, _predict_proba_logit
from mlframe.feature_engineering.transformer.residual_stratified_distance import compute_residual_stratified_distance_features


def test_minus_one_one_labels_give_the_same_model_as_zero_one():
    """The focal gradient assumes y in {0, 1}; {-1, 1} labels used to train a wrong gradient."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 4)).astype(np.float32)
    y01 = (X[:, 0] > 0).astype(np.float32)
    kw = dict(n_estimators=20, max_depth=3, gamma=2.0, seed=1)
    p01, _ = _predict_proba_logit(_fit_focal_lgb(X, y01, **kw), X)
    pm1, _ = _predict_proba_logit(_fit_focal_lgb(X, 2 * y01 - 1, **kw), X)
    assert np.allclose(p01, pm1, atol=1e-6)


def test_constant_target_does_not_empty_the_easy_band():
    """A constant target makes every out-of-fold residual tie at the median; the strict ``< median`` test left no easy rows and every query got the far sentinel."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3)).astype(np.float32)
    y = np.full(200, 0.5, dtype=np.float32)
    df = compute_residual_stratified_distance_features(X, y, rng.standard_normal((10, 3)).astype(np.float32), seed=1, task="regression")
    assert float(df.to_numpy().min()) > -5.0, df.to_numpy().min()
