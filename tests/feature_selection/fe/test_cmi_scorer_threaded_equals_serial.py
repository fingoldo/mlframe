"""The host CMI candidate scorer returns the same Series whether its candidates run on one thread or on a pool."""
import numpy as np
import pandas as pd

import mlframe.feature_selection.filters._mi_greedy_cmi_fe as cmi_mod


def test_pool_scores_equal_serial_scores(monkeypatch):
    """40 candidates (above the pool floor) with a two-column support: identical values and order with the pool on and off."""
    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")
    rng = np.random.default_rng(0)
    n = 3000
    cand = pd.DataFrame(rng.normal(size=(n, 40)), columns=[f"c{i}" for i in range(40)])
    support = pd.DataFrame(rng.normal(size=(n, 2)), columns=["a", "b"])
    y = ((cand["c3"] * support["a"]) + 0.5 * rng.normal(size=n) > 0).astype(int).to_numpy()
    threaded = cmi_mod.score_candidates_by_cmi(cand, y, support)
    monkeypatch.setattr(cmi_mod, "_PARALLEL_CMI_MIN_CANDS", 10**9)
    serial = cmi_mod.score_candidates_by_cmi(cand, y, support)
    pd.testing.assert_series_equal(threaded, serial)
    assert threaded.idxmax() == "c3"
