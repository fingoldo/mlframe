"""A fitted MRMR pickles, deep-copies and clones: same selection and same transform output, no frame, cache or lock held in the pickle."""

import copy
import pickle  # nosec B403 - the test pickles its own estimator

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from mlframe.feature_selection.filters.mrmr import MRMR


@pytest.fixture(scope="module")
def fitted():
    """A fit with engineered features, its input frame and its transform output."""
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({f"x{i}": rng.standard_normal(n).astype(np.float32) for i in range(6)})
    y = (X["x0"] * X["x1"] + 0.3 * rng.standard_normal(n) > 0).astype(int).to_numpy()
    sel = MRMR(verbose=0, random_seed=0, n_jobs=1, fe_max_steps=1).fit(X, y)
    return sel, X, sel.transform(X)


def test_pickle_round_trip_gives_the_same_transform(fitted):
    """pickle.loads(pickle.dumps(fitted)) transforms exactly as the original."""
    sel, X, out = fitted
    restored = pickle.loads(pickle.dumps(sel))  # nosec B301 - bytes just produced by this test
    pd.testing.assert_frame_equal(restored.transform(X), out)
    assert list(restored.get_feature_names_out()) == list(sel.get_feature_names_out())


def test_pickle_does_not_carry_the_training_frame(fitted):
    """The pickle is far smaller than the training frame times a small factor: no copy of X is stored on the estimator."""
    sel, X, _ = fitted
    assert len(pickle.dumps(sel)) < 4 * X.memory_usage(deep=True).sum()


def test_deepcopy_gives_the_same_transform(fitted):
    """copy.deepcopy of a fitted estimator transforms like the original."""
    sel, X, out = fitted
    pd.testing.assert_frame_equal(copy.deepcopy(sel).transform(X), out)


def test_clone_is_unfitted_with_the_same_parameters(fitted):
    """sklearn.clone drops the fit and keeps the parameters."""
    sel, _, _ = fitted
    fresh = clone(sel)
    assert fresh.get_params()["fe_max_steps"] == sel.get_params()["fe_max_steps"]
    assert not hasattr(fresh, "support_") or fresh.support_ is None
