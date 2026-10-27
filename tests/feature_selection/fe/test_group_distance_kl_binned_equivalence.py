"""The group-distance KL feature from per-row global bins equals the one from an ``np.histogram`` call per group."""
import numpy as np
import pandas as pd

import mlframe.feature_selection.filters._group_distance_fe as gd


def _frame(seed, integers):
    """A group column, a numeric column with ties on the bin edges (integers) or continuous, plus NaN / inf rows and one tiny group."""
    rng = np.random.default_rng(seed)
    n = 3000
    x = rng.integers(0, 40, n).astype(float) if integers else rng.normal(size=n)
    x[::97] = np.nan
    x[5] = np.inf
    g = rng.integers(0, 9, n).astype(str)
    g[:3] = "tiny"
    return pd.DataFrame({"g": g, "x": x})


def test_kl_lookup_equals_the_histogram_per_group_reference(monkeypatch):
    """Both tie-heavy and continuous columns: every group's KL value is identical to the reference that calls ``np.histogram`` on the group's finite values."""
    for seed, integers in [(0, True), (1, False), (2, True)]:
        X = _frame(seed, integers)
        _enc, raw = gd.generate_group_distance_features(X, ["g"], ["x"])
        kl = raw[gd.engineered_name_group_kl("x", "g")]["kl_lookup"]
        x = X["x"].to_numpy()
        fin_all = x[np.isfinite(x)]
        edges = np.linspace(fin_all.min(), max(fin_all.max(), fin_all.min() + 1e-300) if fin_all.max() > fin_all.min() else fin_all.min() + 1.0, gd._DIST_NBINS + 1)
        hist = np.histogram(fin_all, bins=edges)[0].astype(np.float64)
        for key, value in kl.items():
            vals = x[(X["g"] == key).to_numpy()]
            fin = vals[np.isfinite(vals)]
            want = gd._kl_divergence(fin, edges, hist) if fin.size >= gd._MIN_GROUP_SIZE else 0.0
            assert value == want, (seed, key, value, want)
