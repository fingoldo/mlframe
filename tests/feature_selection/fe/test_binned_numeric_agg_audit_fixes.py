"""Binned numeric aggregate: replay bins at float64, non-finite aggregate columns rank by finite variance, the host redo reseeds, float64 label coding."""

from __future__ import annotations

import numpy as np
import pandas as pd

import mlframe.feature_selection.filters._binned_numeric_agg_fe as fe


def test_apply_binned_numeric_agg_bins_group_at_float64(monkeypatch):
    """A value one ulp below an edge stays in the lower cell on replay, as it did at fit time."""
    monkeypatch.setenv("MLFRAME_CRIT_DTYPE_RELAXED", "1")
    edge = 0.1
    X = pd.DataFrame({"g": [float(np.nextafter(edge, 0.0))]})
    recipe = {"group_col": "g", "edges": np.array([edge]), "lookup": np.array([10.0, 20.0]), "global": -1.0}
    assert fe.apply_binned_numeric_agg(X, recipe)[0] == 10.0


def test_top_agg_by_variance_ranks_nan_bearing_column_by_finite_variance():
    """A NaN in an aggregate column no longer makes its variance NaN and scrambles the sort."""
    rng = np.random.default_rng(0)
    n = 200
    big = rng.normal(0, 10.0, n)
    big[3] = np.nan
    X = pd.DataFrame({"a": rng.normal(0, 1.0, n), "b": rng.normal(0, 2.0, n), "big_nan": big})
    top, var = fe._top_agg_by_variance(X, ["a", "b", "big_nan"], 3)
    assert top == ["big_nan", "b", "a"]
    assert np.isfinite(var["big_nan"])


def test_redundancy_gate_host_redo_reseeds_the_permutation_generator(monkeypatch):
    """After a device failure the host redo draws the same seeds as a clean run, not the advanced generator's."""
    rng = np.random.default_rng(1)
    n = 120
    X = pd.DataFrame({"g": rng.normal(size=n), "a": rng.normal(size=n)})
    y = (rng.random(n) > 0.5).astype(int)

    class _Dev:
        """Device source stand-in carrying only the candidate names."""

        names = ["binagg_mean(a|qbin(g))"]

    raw = {_Dev.names[0]: {"group_col": "g", "agg_col": "a", "stat": "mean"}}
    seen: list = []

    def fake_build(cands, raw_, X_, src_bins, nbins, y_cls, gain, _rng, *rest):
        """First call burns draws then fails; the second records its first draw."""
        if not seen:
            seen.append(None)
            _rng.integers(0, 2**63 - 1, size=3)
            raise RuntimeError("device fault")
        seen.append(int(_rng.integers(0, 2**63 - 1)))

    monkeypatch.setattr(fe, "_build_binned_agg_recipes", fake_build)
    fe._redundancy_gate_binagg(X, y, raw, None, _Dev(), ["g"], ["a"], ("mean",), 10, 5, 7, 0.005, None)
    assert seen[1] == int(np.random.default_rng(7).integers(0, 2**63 - 1))


def test_group_selection_y_codes_keeps_large_integer_labels_distinct():
    """Integer labels above 2**24 stay two classes instead of merging at float32 precision."""
    y = np.array([16777216, 16777217] * 20)
    codes = fe._group_selection_y_codes(y)
    assert len(np.unique(codes)) == 2
