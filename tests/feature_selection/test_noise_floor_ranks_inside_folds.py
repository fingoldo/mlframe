"""The noise-floor CV ranks each fold on its own training rows only (audit 2026-11-20, finding 12-F21)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from mlframe.feature_selection._benchmarks.fs_hybrid.round4_noise_floor_bench import cv_curve


def test_ranker_is_called_once_per_fold_on_training_rows_only():
    """A global ranking sees every row, including the rows each fold then scores. The per-fold ranker must receive only the
    training part of each fold: three calls, each with fewer rows than the full frame."""
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(rng.standard_normal((n, 4)), columns=list("abcd"))
    y = pd.Series((X["a"] + 0.5 * rng.standard_normal(n) > 0).astype(int))
    sizes = []

    def ranker(Xt, yt):
        """Record the rows it is given and return the columns in order."""
        sizes.append(len(Xt))
        return list(Xt.columns)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    cv_curve(X, y, list(X.columns), [1, 2], cv, permute=False, ranker=ranker)
    assert len(sizes) == 3
    assert all(s < n for s in sizes)
