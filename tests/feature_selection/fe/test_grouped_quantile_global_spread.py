"""The global IQR and p90-p10 fallbacks read from the sorted column match np.quantile on the fit dtype (the per-group spreads already use this form)."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._grouped_quantile_fe import generate_grouped_quantile_features


def test_global_spreads_equal_np_quantile():
    """The recipes carry the same global spread values np.quantile gives, for float32-sized and tied data."""
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({"g": rng.integers(0, 4, n).astype(str), "v": np.round(rng.normal(size=n), 1)})
    _, recipes = generate_grouped_quantile_features(X, ["g"], ["v"])
    x = X["v"].to_numpy().astype(np.float32)
    iqr = [r for r in recipes.values() if r.get("op") == "iqr"]
    p90 = [r for r in recipes.values() if r.get("op") == "p90p10"]
    assert iqr and p90
    np.testing.assert_allclose(iqr[0]["global_iqr"], float(np.quantile(x, 0.75)) - float(np.quantile(x, 0.25)), rtol=1e-6)
    np.testing.assert_allclose(p90[0]["global_p90p10"], float(np.quantile(x, 0.90)) - float(np.quantile(x, 0.10)), rtol=1e-6)
