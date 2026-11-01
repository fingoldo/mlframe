"""The auxiliary LightGBM models cap their leaf count at 63 instead of 2**max_depth (INP2-2).

``num_leaves=2**max_depth`` is unbounded: a caller-chosen depth of 20 asks for a million leaves per tree. The boosting-leaf transformer
already caps it at 63, and the auxiliary models now match.
"""

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from mlframe.feature_engineering.transformer import multi_aux_ensemble as mae


def test_deep_auxiliary_lightgbm_model_caps_its_leaf_count():
    """A large max_depth yields at most 63 leaves per tree, not 2**max_depth."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] > 0).astype(int)
    model = mae._fit_aux_lgb(X, y, task="binary", seed=0, n_estimators=3, max_depth=20)
    assert model.get_params()["num_leaves"] <= 63
