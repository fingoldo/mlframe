"""Regression (MRMR critique S-F2): under redundancy_aggregator='jmim' candidates are SCORED by the JMIM joint-MI
criterion, but the fleuret confirmation path called evaluate_gain WITHOUT use_jmim -> it confirmed JMIM picks against
the CMIM conditional-MI null (statistic mismatch). use_jmim is now threaded through the confidence chain
(get_fleuret_criteria_confidence_parallel -> parallel_fleuret -> get_fleuret_criteria_confidence -> evaluate_gain),
mirroring use_su/use_mm, so the confirmation uses the same statistic as the scoring.
"""
import inspect

import numpy as np
import pandas as pd


def test_confidence_chain_threads_use_jmim():
    from mlframe.feature_selection.filters import fleuret
    for fn in (fleuret.parallel_fleuret, fleuret.get_fleuret_criteria_confidence):
        assert "use_jmim" in inspect.signature(fn).parameters, f"{fn.__name__} must accept use_jmim (S-F2)"


def test_jmim_fit_completes_with_confirmation():
    from mlframe.feature_selection.filters.mrmr import MRMR
    import warnings
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.integers(0, 5, size=(500, 8)).astype(float), columns=[f"f{i}" for i in range(8)])
    y = pd.Series((X["f0"].astype(int) ^ X["f1"].astype(int) > 0).astype(int))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(redundancy_aggregator="jmim", full_npermutations=2, cv=2, run_additional_rfecv_minutes=False).fit(X, y)
    assert hasattr(m, "support_") and len(m.get_feature_names_out()) >= 1
