"""The IB baseline-code transformer scores its train rows with out-of-fold baseline predictions."""

import numpy as np

from mlframe.feature_engineering.transformer.ib_baseline_codes import _fit_3baselines_two


def test_train_predictions_of_the_baselines_do_not_memorise_a_noise_target():
    """On a target that is pure noise the train-side predictions must be out-of-fold: an in-sample fit tracks y closely, an out-of-fold one cannot."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((600, 6)).astype(np.float32)
    y = rng.standard_normal(600).astype(np.float32)
    tp, qp = _fit_3baselines_two(X, y, X[:50], task="regression", seed=1)
    assert tp.shape == (600, 3) and qp.shape == (50, 3)
    assert abs(float(np.corrcoef(tp[:, 1], y)[0, 1])) < 0.3
