"""Target mix on a non-stationary target that is stationary and far more predictable relative to another column, judged out of time.

The target follows a drifting level that another column ``level`` observes (a random walk with drift: its range out of time is never the range it was fitted on), plus a stationary interaction
``x2 * x3``. A tree model fitted on the raw target cannot extrapolate the level, so every later fold is predicted at the edge of what it has seen; mixing the target with the
level column removes the drift, the remainder is the stationary interaction, and the model fitted on it generalises across time. Evaluation is ``TimeSeriesSplit``: each fold fits on
the past and scores on the next block.
"""

import numpy as np
import pytest
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

from mlframe.feature_selection.filters._target_offset_regressor import TargetOffsetRegressor
from mlframe.feature_selection.filters.mrmr import MRMR

N = 4500


def _frame(seed):
    """Time-ordered frame: a random-walk level (observed with a little noise), a stationary interaction and decoys."""
    rng = np.random.default_rng(seed)
    level = np.cumsum(rng.normal(0.03, 1.0, N))
    x2, x3 = rng.standard_normal(N), rng.standard_normal(N)
    noise = rng.standard_normal((N, 3))
    y = level + 2.0 * x2 * x3 + 0.3 * rng.standard_normal(N)
    X = pd.DataFrame({"level": level + 0.05 * rng.standard_normal(N), "x2": x2, "x3": x3, "n1": noise[:, 0], "n2": noise[:, 1], "n3": noise[:, 2]})
    return X, y


def _rmse(a, b):
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _out_of_time_rmse(model_factory, X, y):
    """Mean RMSE over the TimeSeriesSplit folds, each fitted on the past only."""
    scores = []
    for tr, te in TimeSeriesSplit(n_splits=3, test_size=900).split(X):
        model = model_factory()
        model.fit(X.iloc[tr].reset_index(drop=True), y[tr])
        scores.append(_rmse(y[te], model.predict(X.iloc[te].reset_index(drop=True))))
    return float(np.mean(scores))


@pytest.mark.usefixtures("strict_mode")
def test_target_mix_beats_the_raw_target_out_of_time():
    """The mixed-target model has a far lower out-of-time error than the same regressor on the raw target."""
    seed = 0
    X, y = _frame(seed)
    raw = _out_of_time_rmse(lambda: HistGradientBoostingRegressor(random_state=0), X, y)
    mixed = _out_of_time_rmse(
        lambda: TargetOffsetRegressor(
            selector=MRMR(verbose=0, n_jobs=1, random_state=seed, fe_target_mix_enable=True, fe_target_mix_max_bases=1, fe_max_steps=1),
            regressor=HistGradientBoostingRegressor(random_state=0),
        ),
        X,
        y,
    )
    assert mixed < 0.5 * raw, f"mixed {mixed:.3f} vs raw {raw:.3f}"
