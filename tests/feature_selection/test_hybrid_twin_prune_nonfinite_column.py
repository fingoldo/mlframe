"""A selected column with no finite value must not switch the whole conditional-twin pass off."""

import numpy as np
import pandas as pd

from mlframe.feature_selection._hybrid_twin_prune import prune_conditional_twins


def test_all_nan_column_does_not_disable_the_twin_pass():
    """The near-duplicate of a kept column is still dropped when another selected column is entirely NaN."""
    rng = np.random.default_rng(0)
    n = 600
    a = rng.normal(size=n)
    X = pd.DataFrame({"a": a, "a_twin": a + 0.01 * rng.normal(size=n), "dead": np.full(n, np.nan)})
    y = a + 0.5 * rng.normal(size=n)
    out = prune_conditional_twins(X, y, ["a", "a_twin", "dead"], {"a": 1.0, "a_twin": 0.5, "dead": 0.0}, classification=False)
    assert "a" in out and "a_twin" not in out
