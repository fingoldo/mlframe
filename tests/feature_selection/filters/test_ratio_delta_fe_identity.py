"""CPX4 identity regression for _ratio_delta_fe.py optimizations.

Pins that:
  * the hoisted per-column to_numpy() pairwise ratio/log-ratio loop produces
    bit-identical features + accepted-pair lists; and
  * the vectorized _map_group_keys gather (grouped_delta_features /
    apply_grouped_delta) is bit-identical to the per-row dict.get listcomp.
"""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _ratio_delta_fe as rd


def _ref_map(keys, lookup, global_value):
    """Reference per-row dict.get listcomp (pre-optimization behaviour)."""
    return np.array([lookup.get(str(_k), global_value) for _k in keys], dtype=np.float64)


def test_map_group_keys_matches_per_row_loop():
    """Map group keys matches per row loop."""
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 50, size=10_000)
    lookup = {str(k): float(rng.normal()) for k in range(40)}  # 10 keys unseen
    glob = 3.14
    got = rd._map_group_keys(keys, lookup, glob)
    ref = _ref_map(keys, lookup, glob)
    assert np.array_equal(got, ref)


def test_pairwise_ratio_hoist_identity():
    """Pairwise ratio hoist identity."""
    rng = np.random.default_rng(1)
    cols = [f"c{i}" for i in range(8)]
    X = pd.DataFrame({c: rng.normal(5.0, 2.0, size=2000) for c in cols})

    df, acc = rd.pairwise_ratio_features(X, cols)
    # Reference: recompute each surviving pair directly from the columns.
    for a, b in acc:
        a_vals = np.asarray(X[a].to_numpy(), dtype=np.float64)
        b_vals = np.asarray(X[b].to_numpy(), dtype=np.float64)
        r = rd._safe_div(a_vals, b_vals, 1e-9)
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        name = rd.engineered_name_ratio(a, b)
        assert np.array_equal(df[name].to_numpy(), r), f"ratio {a}/{b} differs"


def test_pairwise_log_ratio_hoist_identity():
    """Pairwise log ratio hoist identity."""
    rng = np.random.default_rng(2)
    cols = [f"c{i}" for i in range(8)]
    X = pd.DataFrame({c: rng.normal(5.0, 2.0, size=2000) for c in cols})

    df, acc = rd.pairwise_log_ratio_features(X, cols)
    for a, b in acc:
        a_vals = np.asarray(X[a].to_numpy(), dtype=np.float64)
        b_vals = np.asarray(X[b].to_numpy(), dtype=np.float64)
        lr = np.log1p(np.abs(a_vals) + 1e-9) - np.log1p(np.abs(b_vals) + 1e-9)
        lr = np.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0)
        name = rd.engineered_name_log_ratio(a, b)
        assert np.array_equal(df[name].to_numpy(), lr), f"log_ratio {a}/{b} differs"


def test_grouped_delta_and_apply_identity():
    """Grouped delta and apply identity."""
    rng = np.random.default_rng(3)
    n = 20_000
    g = rng.integers(0, 100, size=n)
    num_cols = ["x0", "x1"]
    X = pd.DataFrame({"grp": g, "x0": rng.normal(size=n), "x1": rng.normal(2.0, 3.0, size=n)})
    _enc, recipes = rd.grouped_delta_features(X, "grp", num_cols)

    # Replay-apply path with unseen groups must match its own reference map.
    g2 = rng.integers(0, 130, size=n)  # some unseen groups
    X_test = pd.DataFrame({"grp": g2, "x0": rng.normal(size=n), "x1": rng.normal(size=n)})
    name_z = rd.engineered_name_grouped_delta_std("x0", "grp")
    rec = recipes[name_z]
    out = rd.apply_grouped_delta(X_test, rec)

    # Reference: per-row dict.get listcomp for mean + std, then div.
    from mlframe.feature_selection.filters._internals import group_key_strings

    g_vals = group_key_strings(X_test["grp"])
    x = np.asarray(X_test["x0"].to_numpy(), dtype=np.float64)
    pm = _ref_map(g_vals, dict(rec["lookup_mean"]), float(rec["global_mean"]))
    ps = _ref_map(g_vals, dict(rec["lookup_std"]), float(rec["global_std"]) or 1.0)
    ps = np.where(ps > 0.0, ps, 1.0)
    ref = np.nan_to_num((x - pm) / ps, nan=0.0, posinf=0.0, neginf=0.0)
    assert np.array_equal(out, ref)
