"""Business value of the RFECV futility stop: on a dense-signal frame (the full set is the pick) it skips most iterations without changing the selection."""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge

from mlframe.feature_selection.wrappers import RFECV


def _pair(seed: int):
    """Pair."""
    X, y = make_regression(3000, 14, n_informative=14, noise=20.0, random_state=seed, shuffle=False)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(14)])
    return [RFECV(estimator=Ridge(1.0), cv=5, random_state=seed, verbose=0, futility_stop=flag).fit(X, y) for flag in (False, True)]


def test_biz_val_rfecv_futility_stop_dense_signal_iterations_saved_selection_identical():
    # Measured: 6 of 14 iterations evaluated (57% saved) on every seed tried, identical 14-feature selection; threshold ~10% below.
    """Biz val rfecv futility stop dense signal iterations saved selection identical."""
    saved, same = [], []
    for seed in (0, 1):
        base, stopped = _pair(seed)
        assert len(base.eval_trace_) == 14, "the existing no-improve stop must not end the baseline search early"
        saved.append(1.0 - len(stopped.eval_trace_) / len(base.eval_trace_))
        same.append(list(stopped.get_feature_names_out()) == list(base.get_feature_names_out()))
    assert min(saved) >= 0.50
    assert all(same)
