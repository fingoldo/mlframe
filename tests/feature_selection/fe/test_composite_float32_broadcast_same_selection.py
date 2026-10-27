"""The composite broadcast column stores the criterion dtype, and the CMI ranking of the top candidates is the same as with float64 columns."""

import numpy as np
import pandas as pd
import pytest

import mlframe.feature_selection.filters._composite_group_agg_fe as comp
from mlframe.feature_selection.filters._grouped_agg_fe import score_grouped_agg_by_cmi_uplift


@pytest.mark.parametrize("seed", range(4))
def test_float32_broadcast_keeps_the_top_candidates(seed, monkeypatch):
    """Scales spanning five orders of magnitude; the same top-10 names in the same order under the relaxed (float32) and strict (float64) criterion dtype."""
    rng = np.random.default_rng(seed)
    n = 6000
    X = pd.DataFrame({"g1": rng.integers(0, 12, n), "g2": rng.integers(0, 50, n)})
    for i in range(6):
        X[f"x{i}"] = rng.normal(size=n) * (10.0 ** rng.integers(-2, 3))
    y = (X.groupby(["g1", "g2"])["x0"].transform("mean").to_numpy() + rng.normal(size=n) > 0).astype(int)
    nums = [f"x{i}" for i in range(6)]
    tops = {}
    for relax in ("1", "0"):
        monkeypatch.setenv("MLFRAME_CRIT_DTYPE_RELAXED", relax)
        enc, recipes = comp.generate_composite_group_agg_features(X, [("g1", "g2")], nums)
        eng = {k: recipes[k]["num_col"] for k in enc.columns}
        scores = score_grouped_agg_by_cmi_uplift(X, enc, y, nums, n_bins=10, eng_to_source=eng)
        tops[relax] = list(scores.sort_values("cmi", ascending=False)["engineered_col"].head(10))
        if relax == "1":
            assert enc.dtypes.map(lambda d: d == np.float32).all()
    assert tops["1"] == tops["0"]
