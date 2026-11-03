"""The meta-stacker alpha search keeps groups whole and respects time order when asked."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.ensemble import _stackers as st


def test_group_splits_never_share_a_group_between_train_and_validation():
    """Every GroupKFold fold holds each group on one side only."""
    groups = np.repeat(np.arange(20), 10)
    finite = np.ones(groups.size, dtype=bool)
    splits = st._meta_cv_splits(groups.size, finite, groups, False)
    assert splits and len(splits) == 5
    for tr, va in splits:
        assert not set(groups[tr]) & set(groups[va])


def test_time_ordered_splits_validate_only_on_later_rows():
    """Forward-chaining folds validate strictly after their training rows."""
    finite = np.ones(100, dtype=bool)
    for tr, va in st._meta_cv_splits(100, finite, None, True):
        assert tr.max() < va.min()


def test_default_stays_sklearn_kfold():
    """Without groups or time order the helper defers to sklearn."""
    assert st._meta_cv_splits(50, np.ones(50, dtype=bool), None, False) is None


@pytest.mark.parametrize("fit", ["fit_ridge_meta_stacker", "fit_lasso_meta_stacker", "fit_elasticnet_meta_stacker"])
def test_stackers_fit_with_group_cv(fit):
    """Each linear stacker accepts the group ids and still returns a usable model."""
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 3))
    y = X @ np.array([0.5, 0.3, 0.0]) + rng.normal(0, 0.1, n)
    model = getattr(st, fit)(X, y, 3, group_ids=np.repeat(np.arange(40), 10))
    assert np.asarray(model.predict(X)).shape == (n,)
