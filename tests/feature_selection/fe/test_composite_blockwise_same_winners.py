"""Scored block by block the composite aggregates keep the same winners and the same recipes as the full broadcast."""
import numpy as np
import pandas as pd

import mlframe.feature_selection.filters._composite_group_agg_fe as comp


def _frame(seed=0, n=3000):
    """Two group columns, four numeric columns, and a target that depends on one composite key's mean."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"g1": rng.integers(0, 6, n), "g2": rng.integers(0, 5, n)})
    for i in range(4):
        X[f"x{i}"] = rng.normal(size=n)
    key_mean = X.groupby(["g1", "g2"])["x0"].transform("mean").to_numpy()
    y = (key_mean + 0.5 * rng.normal(size=n) > 0).astype(np.int64)
    return X, y


def test_blockwise_winners_equal_full_winners(monkeypatch):
    """A small block budget forces several blocks per key set; the kept names and their recipes equal the full path's."""
    X, y = _frame()
    num_cols = [f"x{i}" for i in range(4)]
    sets = [("g1", "g2")]
    full_enc, full_recipes = comp.generate_composite_group_agg_features(X, sets, num_cols)
    base = list(num_cols)
    eng_to_source = {name: full_recipes[name]["num_col"] for name in full_enc.columns}
    from mlframe.feature_selection.filters._grouped_agg_fe import score_grouped_agg_by_cmi_uplift
    scores_full = score_grouped_agg_by_cmi_uplift(X, full_enc, y, base, n_bins=10, eng_to_source=eng_to_source)
    keep = scores_full[(scores_full["cmi"] >= 1e-4) & (scores_full["uplift"] >= 0.0)]
    want = list(keep["engineered_col"].head(5))
    monkeypatch.setenv("MLFRAME_COMPOSITE_BLOCK_MB", "0.02")
    frame, _scores, recipes = comp.composite_group_agg_top_blockwise(
        X, y, sets, num_cols, base_cols=base, top_k=5, n_bins=10, min_cmi=1e-4, min_uplift=0.0, eng_to_source={}
    )
    assert set(frame.columns) == set(want)
    assert set(recipes) == set(want)
    assert all(np.array_equal(frame[c].to_numpy(), full_enc[c].to_numpy()) for c in want)
