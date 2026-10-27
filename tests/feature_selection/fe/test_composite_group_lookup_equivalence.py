"""The per-group lookups of the composite group aggregates equal the ``Series.items()`` dictionary comprehensions they replaced."""
import numpy as np
import pandas as pd

import mlframe.feature_selection.filters._composite_group_agg_fe as comp


def test_lookups_match_the_items_comprehension_reference():
    """Mean / std / count lookups on a float32 column with NaN rows, an all-NaN group and a one-row group (std NaN -> 0.0): same keys, same Python floats."""
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame({"g1": rng.integers(0, 6, n), "g2": rng.integers(0, 5, n), "x": rng.normal(size=n)})
    X.loc[::13, "x"] = np.nan
    X.loc[(X.g1 == 5) & (X.g2 == 4), "x"] = np.nan  # an all-NaN cell
    X.loc[0, ["g1", "g2"]] = [99, 99]  # a one-row cell
    gc = ("g1", "g2")
    enc, raw = comp.generate_composite_group_agg_features(X, [gc], ["x"], stats=("mean", "std", "count"))
    keys = comp.build_composite_keys(X, gc)
    x32 = X["x"].to_numpy().astype(np.float32)
    grouped = pd.DataFrame({"_g": keys, "_v": x32}).groupby("_g", observed=True, sort=False)["_v"]
    want_mean = {str(k): float(v) for k, v in grouped.mean().items()}  # the residual base keeps NaN
    want_mean_stat = {k: (v if np.isfinite(v) else 0.0) for k, v in want_mean.items()}  # the broadcast statistic zeroes it
    want_std = {str(k): (float(v) if np.isfinite(v) else 0.0) for k, v in grouped.std(ddof=1).items()}
    want_count = {str(k): (float(v) if np.isfinite(v) else 0.0) for k, v in grouped.agg("count").items()}
    r_mean = raw[comp.engineered_name_composite_agg("x", gc, "mean")]
    r_std = raw[comp.engineered_name_composite_agg("x", gc, "std")]
    r_count = raw[comp.engineered_name_composite_agg("x", gc, "count")]

    def same(got, want):
        """Equal key sets and equal values per key, NaN equal to NaN (the dictionary order follows the group codes, not first appearance)."""
        assert set(got) == set(want)
        assert all((got[k] == want[k]) or (np.isnan(got[k]) and np.isnan(want[k])) for k in want)

    same(r_mean["group_lookup_dict"], want_mean_stat)
    same(r_mean["lookup_mean"], want_mean)
    same(r_mean["lookup_std"], want_std)
    same(r_std["group_lookup_dict"], want_std)
    same(r_count["group_lookup_dict"], want_count)
    assert enc.shape[0] == n
