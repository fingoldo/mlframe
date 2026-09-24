"""cascade_select_stable must hand each run distinct rows, so the inner CV cannot see a row on both sides."""

import numpy as np
import pandas as pd

import mlframe.feature_selection.cascade_select_stability as css


def test_each_run_sees_half_the_rows_with_no_duplicates(monkeypatch):
    seen = []

    def fake_cascade(X, y, estimator_factory, **kw):
        seen.append(X["row_id"].to_numpy().copy())
        return {"final_selected": []}

    monkeypatch.setattr(css, "cascade_select", fake_cascade)
    X = pd.DataFrame({"row_id": np.arange(100), "f": np.zeros(100)})
    css.cascade_select_stable(X, np.zeros(100), estimator_factory=None, n_bootstrap=5)
    assert len(seen) == 5
    for ids in seen:
        assert len(ids) == 50
        assert len(np.unique(ids)) == len(ids), "a duplicated row lands in both the train and test folds of the inner CV"
