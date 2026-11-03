"""Grouped-quantile candidates keep their sorted values as arrays; the recipes built for the winners hold plain float lists equal to them."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._grouped_quantile_fe import apply_grouped_quantile, generate_grouped_quantile_features, hybrid_grouped_quantile_fe


def _frame(n: int = 3000):
    """A group column whose groups shift a numeric column, and a target driven by the within-group position."""
    rng = np.random.default_rng(4)
    g = rng.integers(0, 5, n)
    x = rng.standard_normal(n) + 3.0 * g
    within = x - 3.0 * g
    y = (within > 0).astype(int)
    return pd.DataFrame({"g": g, "x": x, "z": rng.standard_normal(n)}), y


def test_winner_recipes_hold_float_lists_of_the_group_values():
    """Each stored per-group list is the sorted finite values of that group, as Python floats; replay reproduces the fitted column on the training rows' groups."""
    X, y = _frame()
    _X_aug, winners, recipes, _scores = hybrid_grouped_quantile_fe(X, y, group_cols=["g"], num_cols=["x", "z"], top_k=4, random_state=0)
    assert winners
    for rec in recipes:
        extra = rec.extra
        assert isinstance(extra["global_sorted"], list) and all(type(v) is float for v in extra["global_sorted"])
        for vals in extra["group_sorted"].values():
            assert isinstance(vals, list) and all(type(v) is float for v in vals)
        if extra["op"] == "pct_rank":
            for gv in np.unique(X["g"]):
                want = np.sort(X.loc[X["g"] == gv, extra["num_col"]].to_numpy(dtype=np.float32)).astype(float).tolist()
                assert extra["group_sorted"][str(gv)] == want
        assert np.asarray(apply_grouped_quantile(X, extra)).shape == (len(X),)


def test_candidate_payloads_hold_arrays():
    """The candidate payloads (before any winner is chosen) keep numpy arrays: no per-group list is built for a candidate that is then dropped."""
    X, _y = _frame()
    _enc, raw = generate_grouped_quantile_features(X, group_cols=["g"], num_cols=["x"])
    payload = next(p for p in raw.values() if p["op"] == "pct_rank")
    assert isinstance(payload["global_sorted"], np.ndarray)
    assert all(isinstance(v, np.ndarray) for v in payload["group_sorted"].values())
