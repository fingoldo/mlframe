"""With premerge_clusters, Subset() returns the cluster members the selection includes, not only their representatives (audit 2026-11-20, finding 13-F1)."""
import numpy as np
import pandas as pd

from mlframe.feature_selection.boruta_shap import BorutaShap


def test_subset_matches_selected_features_for_a_correlated_cluster():
    """Two near-duplicate columns collapse to one representative; an accepted representative must come back with its member, and Subset() stays inside selected_features_."""
    rng = np.random.default_rng(0)
    n = 400
    a = rng.standard_normal(n)
    X = pd.DataFrame({"a": a, "a2": a + 0.01 * rng.standard_normal(n), "n1": rng.standard_normal(n), "n2": rng.standard_normal(n)})
    y = (a + 0.3 * rng.standard_normal(n) > 0).astype(int)
    model = BorutaShap(premerge_clusters=True, n_trials=5, verbose=False)
    model.fit(X, y)
    subset = set(model.Subset().columns)
    assert subset <= set(model.selected_features_)
    if "a" in model.accepted_:
        assert {"a", "a2"} <= subset
