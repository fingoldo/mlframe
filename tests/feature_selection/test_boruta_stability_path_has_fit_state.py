"""The stability path leaves the fitted state the public accessors read (audit 2026-11-20, finding 13-F2)."""
import numpy as np
import pandas as pd

from mlframe.feature_selection.boruta_shap import BorutaShap


def test_subset_works_after_a_stability_fit():
    """Subset() read starting_X_, which the stability path never set, so it raised AttributeError."""
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(rng.standard_normal((n, 4)), columns=["a", "b", "c", "d"])
    y = (X["a"] + 0.3 * rng.standard_normal(n) > 0).astype(int)
    model = BorutaShap(stability_subsamples=2, n_trials=3, verbose=False)
    model.fit(X, y)
    subset = model.Subset()
    assert subset.shape[0] == n
