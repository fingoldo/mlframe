"""biz_value test for ``feature_selection.filters.boruta_select``.

The win: genuinely relevant features should be "confirmed" (repeatedly beat the max-shadow importance more
often than chance) while pure-noise features should NOT be confirmed (win rate indistinguishable from, or
below, 50% against their own shadow) -- the all-relevant selection guarantee Boruta provides, distinct from
MRMR's minimal-redundant philosophy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from mlframe.feature_selection.filters._boruta import boruta_select


def _importance_fn(X, y):
    model = RandomForestRegressor(n_estimators=60, max_depth=5, random_state=0, n_jobs=1)
    model.fit(X, y)
    return model.feature_importances_


def test_biz_val_boruta_select_confirms_relevant_rejects_noise():
    rng = np.random.default_rng(0)
    n = 800
    x_relevant_1 = rng.normal(0, 1, n)
    x_relevant_2 = rng.normal(0, 1, n)
    x_noise_1 = rng.normal(0, 1, n)
    x_noise_2 = rng.normal(0, 1, n)
    y = 3.0 * x_relevant_1 - 2.0 * x_relevant_2 + rng.normal(0, 0.5, n)

    X = pd.DataFrame({"relevant_1": x_relevant_1, "relevant_2": x_relevant_2, "noise_1": x_noise_1, "noise_2": x_noise_2})

    result = boruta_select(X, y, _importance_fn, n_iterations=20, random_state=0)
    by_name = dict(zip(result["feature_names"], result["decision"]))

    assert by_name["relevant_1"] == "confirmed", by_name
    assert by_name["relevant_2"] == "confirmed", by_name
    assert by_name["noise_1"] != "confirmed", by_name
    assert by_name["noise_2"] != "confirmed", by_name


def test_boruta_select_returns_expected_keys_and_shapes():
    rng = np.random.default_rng(1)
    n = 300
    X = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    y = X["a"].to_numpy() * 2 + rng.normal(0, 0.2, n)

    result = boruta_select(X, y, _importance_fn, n_iterations=10, random_state=1)
    assert len(result["hit_counts"]) == 2
    assert len(result["win_rate"]) == 2
    assert len(result["decision"]) == 2
    assert result["feature_names"] == ["a", "b"]
