"""No folds measured means informativeness is unknown - not the module's red-flag verdict."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from mlframe.evaluation.cv_informativeness import cv_informativeness_check


def test_an_exhausted_split_generator_yields_unknown_not_false():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 2)), rng.normal(size=40)
    out = cv_informativeness_check(X, y, iter([]), Ridge, r2_score)
    assert out["fold_results"] == []
    assert out["informative"] is None, "False is this module's red flag; zero folds cannot justify it"


def test_measured_folds_still_give_a_boolean():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 2))
    y = X[:, 0] * 3 + rng.normal(scale=0.1, size=120)
    splits = [(np.arange(0, 80), np.arange(80, 120)), (np.arange(40, 120), np.arange(0, 40))]
    out = cv_informativeness_check(X, y, splits, Ridge, r2_score)
    assert out["informative"] is True
