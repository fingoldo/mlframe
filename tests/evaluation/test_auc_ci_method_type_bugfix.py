"""Regression test for TYPE3: auc_ci result dict carries a str under "method"."""

import numpy as np

from mlframe.evaluation.bootstrap import auc_ci


def test_auc_ci_method_is_str():
    rng = np.random.default_rng(0)
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    score = rng.random(y_true.shape[0])
    res = auc_ci(y_true, score, method="delong")
    assert isinstance(res["method"], str)
    assert res["method"] == "delong"
