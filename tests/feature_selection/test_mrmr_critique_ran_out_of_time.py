"""Regression (MRMR critique ST-4): ran_out_of_time_ was set only by the outer FE-loop deadline, so a fit whose
budget was consumed elsewhere (screen_predictors honours max_runtime_mins on its own) reported False. It now
OR-in a total-elapsed-vs-budget check, so any fit that exceeds its budget reports ran_out_of_time_=True.
"""
import numpy as np
import pandas as pd


def test_tiny_budget_sets_ran_out_of_time():
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.integers(0, 6, size=(1500, 25)).astype(float), columns=[f"f{i}" for i in range(25)])
    y = pd.Series((rng.random(1500) < 0.5).astype(int))
    m = MRMR(max_runtime_mins=1e-4, full_npermutations=1, cv=2, run_additional_rfecv_minutes=False)
    m.fit(X, y)
    assert m.ran_out_of_time_ is True, "a fit exceeding its max_runtime_mins budget must report ran_out_of_time_"


def test_ample_budget_does_not_flag_timeout():
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.integers(0, 4, size=(300, 6)).astype(float), columns=[f"f{i}" for i in range(6)])
    y = pd.Series((X["f0"] + X["f1"] > 3).astype(int))
    m = MRMR(max_runtime_mins=60.0, full_npermutations=1, cv=2, run_additional_rfecv_minutes=False)
    m.fit(X, y)
    assert m.ran_out_of_time_ is False
