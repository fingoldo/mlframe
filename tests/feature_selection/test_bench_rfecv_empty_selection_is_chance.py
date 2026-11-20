"""An RFECV that keeps no column scores chance in the importance-aggregation bench (audit 2026-11-20, finding 11-F2)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import mlframe.feature_selection._benchmarks.bench_rfecv_importance_agg as bench


class _EmptyRFECV:
    """Stand-in selector that keeps nothing."""

    def __init__(self, **_kwargs):
        """Accept and ignore the selector arguments."""
        self._selected_cols_cache = []

    def fit(self, X, y):
        """Pretend to fit; the selection stays empty."""
        return self


def test_empty_selection_scores_chance(monkeypatch):
    """With nothing selected the bench must report AUC 0.5, not a refit on every column."""
    rng = np.random.default_rng(0)
    Xdf = pd.DataFrame(rng.standard_normal((300, 3)), columns=["a", "b", "c"])
    y = (Xdf["a"] > 0).astype(int).to_numpy()
    monkeypatch.setattr(bench, "_make_scenario", lambda name, seed: (Xdf, y, LogisticRegression(), 0.0))
    monkeypatch.setattr(bench, "RFECV", _EmptyRFECV)
    auc, n_kept, _ = bench._honest_score("any", 0, "legacy")
    assert auc == 0.5 and n_kept == 0
