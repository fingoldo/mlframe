"""A shuffled RFECV splitter with random_state=None must give the same folds on every split within one fit."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from mlframe.feature_selection.wrappers.rfecv._cv_setup import _resolve_cv_and_val_cv


def test_repeated_splits_match_so_per_fold_prescreens_can_be_found():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(120, 3)), rng.integers(0, 2, 120)
    cv, _val_cv, _ = _resolve_cv_and_val_cv(
        cv=4, X=X, y=y, groups=None, estimator=LogisticRegression(), cv_shuffle=True, random_state=None,
        fit_params={}, early_stopping_val_nsplits=0, early_stopping_rounds=None, _polars_time_series_hint=False, verbose=0,
    )
    first = [tuple(tr) for tr, _ in cv.split(X, y)]
    second = [tuple(tr) for tr, _ in cv.split(X, y)]
    assert first == second, "random_state=None repartitioned on every split, so no fold ever matched its prescreen"
