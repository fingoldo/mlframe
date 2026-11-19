"""The block-by-block composite aggregates apply the same redundancy test against the source column as the full path.

The blockwise path (taken when the broadcast of every candidate would not fit the block budget) was called with an empty source map, so ``uplift`` was ``cmi - 0`` and an aggregate that only
re-states its source passed there while the full path rejected it. The source of each candidate now comes from its recipe.
"""

import numpy as np
import pandas as pd

import mlframe.feature_selection.filters._composite_group_agg_fe as comp
from mlframe.feature_selection.filters._grouped_agg_fe import score_grouped_agg_by_cmi_uplift


def _frame(seed: int = 0, n: int = 6000):
    """Two group columns (12 keys) and a source ``x`` equal to the key mean plus small noise; y is a function of the key, so the key mean of ``x`` only re-states ``x``."""
    rng = np.random.default_rng(seed)
    g1, g2 = rng.integers(0, 4, n), rng.integers(0, 3, n)
    key_mean = (g1 * 3 + g2 * 5).astype(float)
    x = key_mean + rng.normal(0.0, 0.3, n)
    y = (key_mean > np.median(key_mean)).astype(np.int64)
    return pd.DataFrame({"g1": g1, "g2": g2, "x": x, "noise": rng.normal(size=n)}), y


def test_blockwise_rejects_the_aggregate_that_restates_its_source(monkeypatch):
    """A small block budget forces the blockwise path; the key mean of a source that already identifies the key is rejected there as on the full path."""
    X, y = _frame()
    sets, num_cols = [("g1", "g2")], ["x"]
    enc, recipes = comp.generate_composite_group_agg_features(X, sets, num_cols)
    scores = score_grouped_agg_by_cmi_uplift(X, enc, y, num_cols, n_bins=10, eng_to_source={c: recipes[c]["num_col"] for c in enc.columns})
    full = scores[(scores["cmi"] >= 1e-4) & (scores["uplift"] >= 0.0)]
    mean_name = next(c for c in enc.columns if c.startswith("cgrpagg_mean("))
    assert mean_name not in set(full["engineered_col"]), "premise: the full path must reject the key mean of a source that already identifies the key"
    monkeypatch.setenv("MLFRAME_COMPOSITE_BLOCK_MB", "0.02")
    frame, _scores, _recipes = comp.composite_group_agg_top_blockwise(
        X, y, sets, num_cols, base_cols=num_cols, top_k=10, n_bins=10, min_cmi=1e-4, min_uplift=0.0, eng_to_source={}
    )
    assert mean_name not in set(frame.columns)
    assert set(frame.columns) == set(full["engineered_col"].head(10))
