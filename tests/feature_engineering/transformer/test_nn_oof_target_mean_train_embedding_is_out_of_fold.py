"""The neighbour index of nn_oof_target_mean is built on out-of-fold baseline predictions, like the query rows' predictions."""
import numpy as np

from mlframe.feature_engineering.transformer.nn_oof_target_mean import _fit_3baselines_predict_two


def test_train_embedding_of_the_deeper_booster_does_not_memorise_a_noise_target():
    """y is noise: an in-sample depth-5 booster correlates strongly with y, an out-of-fold one cannot."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 8)).astype(np.float32)
    y = rng.standard_normal(500).astype(np.float32)
    train_emb, query_emb = _fit_3baselines_predict_two(X, y, X[:20], "regression", 1)
    assert train_emb.shape == (500, 3) and query_emb.shape == (20, 3)
    assert abs(float(np.corrcoef(train_emb[:, 1], y)[0, 1])) < 0.3
