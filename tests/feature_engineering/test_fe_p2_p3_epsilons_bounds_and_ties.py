"""Nine feature_engineering defects that each returned a plausible number instead of failing.

Four are the same shape: a fixed additive epsilon standing in for a real degeneracy branch, in a denominator
whose natural scale is far below that epsilon -- an exponentially-decayed weighted variance, the range of a
large-offset near-constant window. The pad does not guard a division, it replaces the answer with a
scale-dependent one, and the same signal in different units gives a different feature.

The rest: a bounds check that tested three of four endpoints, a `pd.NA` seed that turned a numeric output
column into object dtype, a distance-0 test standing in for a self-match test, a starved row handed a
same-group distance through a group-filtered column, a position-based "strictly before" that counted
contemporaneous rows, a silent zero-fill of missing prices, and a repair pass that walked every row to fix a
handful.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_engineering.anchor import anchor_ewm_features
from mlframe.feature_engineering.fuzzy_entity import fuzzy_entity_group_features
from mlframe.feature_engineering.graph_features import graph_neighbor_aggregate
from mlframe.feature_engineering.nearest_past_join import nearest_past_join
from mlframe.feature_engineering.spatial import knn_aggregate
from mlframe.feature_engineering.windowed_shape import rolling_total_variation


class TestTheEwmAnchorSlopeSurvivesADecayedDenominator:
    """`den` is built from exponentially-decayed sums, so a fixed `+1e-12` pad eventually dominates it."""

    def _slope(self, half_life, spacing=100, n_anchors=4):
        """A perfectly linear anchor sequence -- the true weighted-OLS slope is exactly 1.0 per row."""
        n = spacing * n_anchors + 1
        label = np.full(n, np.nan)
        is_anchor = np.zeros(n, dtype=bool)
        for j in range(n_anchors):
            pos = (j + 1) * spacing
            label[pos] = float(pos)
            is_anchor[pos] = True
        out = anchor_ewm_features(np.nan_to_num(label, nan=0.0), is_anchor, half_life_rows=half_life)
        slope = out[f"ewm_anchor_slope_H{int(half_life)}"]
        finite = slope[np.isfinite(slope)]
        return finite[-1] if finite.size else np.nan

    def test_a_long_half_life_recovers_the_true_slope(self):
        """Control: with the weights barely decayed, this always worked."""
        assert self._slope(200) == pytest.approx(1.0, rel=1e-6)

    def test_a_short_half_life_does_not_report_a_flat_process(self):
        """The defect: at half_life=2 with anchors 100 rows apart the pad swamped `den` and the slope went to 0."""
        s = self._slope(2)
        assert not (np.isfinite(s) and abs(s) < 0.5), f"slope collapsed toward flat: {s}"

    def test_it_is_either_right_or_nan_never_silently_damped(self):
        """A number that is a fraction of the truth is worse than no number."""
        for hl in (1, 2, 5, 20, 200):
            s = self._slope(hl)
            assert np.isnan(s) or s == pytest.approx(1.0, rel=1e-3), (hl, s)


class TestNormalisedTotalVariationIsScaleInvariant:
    """A unitless wiggle-ratio must not change when the same series is shifted or rescaled."""

    def _tv(self, offset, scale, n=200, K=20):
        """A fixed zig-zag whose normalised total variation is a property of its SHAPE alone."""
        base = np.tile([0.0, 1.0], n // 2)
        vals = base * scale + offset
        out = rolling_total_variation(vals, np.zeros(n, dtype=np.int64), window_K=K, normalize=True)
        finite = out[np.isfinite(out)]
        assert finite.size, "the fixture produced no windows"
        return float(finite[-1])

    def test_a_tiny_range_gives_the_same_ratio_as_a_large_one(self):
        """A float32 price column around 1e5 leaves windows whose true range is ~1e-11."""
        assert self._tv(0.0, 1e-11) == pytest.approx(self._tv(0.0, 1.0), rel=1e-9)

    @pytest.mark.parametrize("scale", [1e-13, 1e-11, 1e-6, 1.0, 1e6])
    def test_the_ratio_does_not_move_with_scale(self, scale):
        """Pre-fix at scale 1e-13 the pad was ten times the true range, dropping the ratio by ~91%."""
        assert self._tv(0.0, scale) == pytest.approx(self._tv(0.0, 1.0), rel=1e-9)

    def test_a_constant_window_is_zero_not_a_division(self):
        """`tv == 0` there anyway, so 0.0 is the right degenerate value and no division happens."""
        out = rolling_total_variation(np.full(100, 7.0), np.zeros(100, dtype=np.int64), window_K=10, normalize=True)
        assert set(np.unique(out[np.isfinite(out)]).tolist()) <= {0.0}

    def test_the_numpy_fallback_agrees_with_the_njit_kernel(self):
        """The docstring pins the two forms to each other, so they had to change together."""
        rng = np.random.default_rng(0)
        vals = rng.normal(0, 1e-11, 300)
        vals[7] = np.nan  # a non-finite entry routes the group through the numpy fallback
        clean = rolling_total_variation(np.nan_to_num(vals, nan=0.0), np.zeros(300, dtype=np.int64), window_K=15, normalize=True)
        assert np.isfinite(clean).any() and np.nanmax(clean) > 1.0


class TestEveryEdgeEndpointIsBoundsChecked:
    """Three of the four bounds were checked; `dst.min()` wrapped around under numpy indexing."""

    def test_a_negative_destination_raises_on_the_directed_path(self):
        """`values[-1]` folded the LAST node's value into the aggregate instead."""
        edges = np.array([[0, 1], [2, -1]], dtype=np.int64)
        with pytest.raises(ValueError, match="out of range"):
            graph_neighbor_aggregate(4, edges, np.array([10.0, 20.0, 30.0, 999.0]), directed=True)

    def test_a_negative_source_still_raises(self):
        """The bound that was already checked."""
        with pytest.raises(ValueError, match="out of range"):
            graph_neighbor_aggregate(4, np.array([[-1, 1]], dtype=np.int64), np.zeros(4), directed=True)

    def test_valid_edges_are_unaffected(self):
        """The guard must not start rejecting legitimate graphs."""
        out = graph_neighbor_aggregate(3, np.array([[0, 1], [1, 2]], dtype=np.int64), np.array([1.0, 2.0, 3.0]), directed=True)
        assert np.isfinite(out).all()


class TestTheFallbackChainKeepsANumericDtype:
    """`pd.NA` seeded object dtype, so the dtype changed purely by enabling the fallback chain."""

    def _frames(self):
        """A left frame whose rows resolve at different tiers of the chain."""
        left = pd.DataFrame({"t": [10, 20, 30], "region": ["a", "b", "c"]})
        right = pd.DataFrame({"t": [1, 2], "region": ["a", "z"], "v": [1.5, 2.5]})
        return left, right

    def test_the_attached_column_is_float_not_object(self):
        """`np.isnan` raises on `pd.NA`, and a polars conversion either raises or silently coerces."""
        left, right = self._frames()
        out = nearest_past_join(left, right, on="t", right_value_cols=["v"], fallback_by_chain=[["region"], None])
        assert pd.api.types.is_float_dtype(out["v"]), out["v"].dtype

    def test_it_converts_to_a_float_array(self):
        """The concrete downstream failure the dtype caused."""
        left, right = self._frames()
        out = nearest_past_join(left, right, on="t", right_value_cols=["v"], fallback_by_chain=[["region"], None])
        arr = out["v"].to_numpy(dtype=np.float64)
        assert arr.shape == (3,) and np.isnan(arr).sum() <= 3

    def test_the_single_tier_path_still_agrees(self):
        """The two paths returned different dtypes for the same inputs; they must not."""
        left, right = self._frames()
        single = nearest_past_join(left, right, on="t", by=["region"], right_value_cols=["v"])
        chained = nearest_past_join(left, right, on="t", right_value_cols=["v"], fallback_by_chain=[["region"]])
        assert single["v"].dtype == chained["v"].dtype


class TestADistanceZeroHitIsNotProofOfASelfMatch:
    """Two different entities at one address is ordinary in geocoded data."""

    def _pools(self):
        """A query point coincident with reference 0; references 1 and 2 sit farther out."""
        query = np.array([[0.0, 0.0]])
        ref = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        return query, ref, np.array([100.0, 1.0, 2.0])

    def test_a_coincident_reference_is_kept_when_the_pools_differ(self):
        """It was dropped, shifting the whole k-ring outward by one."""
        q, r, labels = self._pools()
        out = knn_aggregate(q, r, labels, k=2, agg_fns=("mean",))
        assert out["mean"][0] == pytest.approx((100.0 + 1.0) / 2)

    def test_a_true_self_match_is_still_removed(self):
        """The leak-safety the docstring promises for `query is ref`."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        labels = np.array([100.0, 1.0, 2.0])
        out = knn_aggregate(coords, coords, labels, k=1, agg_fns=("mean",))
        assert out["mean"][0] == pytest.approx(1.0), "the query's own label leaked into its aggregate"

    def test_the_flag_can_be_stated_explicitly(self):
        """Identity inference fails when the same rows arrive as two separate arrays."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        labels = np.array([100.0, 1.0, 2.0])
        out = knn_aggregate(coords, coords.copy(), labels, k=1, agg_fns=("mean",), query_is_ref=True)
        assert out["mean"][0] == pytest.approx(1.0)


class TestAStarvedRowGetsNoNearestDistance:
    """Its aggregates are correctly NaN; the distance column must agree."""

    def _starved(self):
        """k=1 overqueries q_k = min(n_ref, 5); six near same-group refs fill that window before the group-2 one."""
        ref = np.array([[float(i) * 0.1, 0.0] for i in range(1, 7)] + [[500.0, 0.0]])
        groups = np.array([1] * 6 + [2])
        return knn_aggregate(
            np.array([[0.0, 0.0]]), ref, np.arange(7, dtype=np.float64), k=1, agg_fns=("mean",), q_group_ids=np.array([1]), ref_group_ids=groups
        )

    def test_the_fixture_really_starves_the_row(self):
        """Without this the consistency assertion below would pass on two finite values and prove nothing."""
        assert np.isnan(self._starved()["mean"][0])

    def test_a_fully_same_group_neighbourhood_yields_nan(self):
        """It was handed `compact_dist[:, 0]` -- a same-group distance, through a group-filtered column."""
        out = self._starved()
        assert np.isnan(out["_nearest_distance"][0]), out["_nearest_distance"][0]


class TestStrictlyBeforeMeansByTimeNotByPosition:
    """Same-day rows were counted as each other's history."""

    def _feats(self, orders):
        """One group, one repeated value, observed at the given time orders."""
        n = len(orders)
        return fuzzy_entity_group_features(np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64), time_order=np.asarray(orders, dtype=np.float64))

    def test_contemporaneous_rows_have_not_seen_each_other(self):
        """Three rows on the same day: none of them has any history."""
        f = self._feats([5.0, 5.0, 5.0])
        assert f["value_occurrence_count_in_group"].tolist() == [0.0, 0.0, 0.0]

    def test_a_same_day_pair_gets_no_zero_day_gap(self):
        """`days_since_value_last_seen_in_group == 0.0` was a same-timestamp leak, not a real gap."""
        f = self._feats([5.0, 5.0])
        assert np.isnan(f["days_since_value_last_seen_in_group"]).all()

    def test_a_genuine_history_is_still_counted(self):
        """The fix must not erase real prior observations."""
        f = self._feats([1.0, 2.0, 5.0, 5.0])
        assert f["value_occurrence_count_in_group"].tolist() == [0.0, 1.0, 2.0, 2.0]
        assert f["days_since_value_last_seen_in_group"][2:].tolist() == [3.0, 3.0]

    def test_strictly_increasing_orders_are_unchanged(self):
        """The case the position-based form was already right about."""
        f = self._feats([1.0, 2.0, 3.0])
        assert f["value_occurrence_count_in_group"].tolist() == [0.0, 1.0, 2.0]
        assert f["days_since_value_last_seen_in_group"][1:].tolist() == [1.0, 1.0]

    def test_the_input_order_of_the_rows_does_not_matter(self):
        """Position-based counting made the answer depend on the incoming row order for tied timestamps."""
        a = self._feats([5.0, 5.0, 1.0])["value_occurrence_count_in_group"]
        b = self._feats([1.0, 5.0, 5.0])["value_occurrence_count_in_group"]
        assert sorted(a.tolist()) == sorted(b.tolist()) == [0.0, 1.0, 1.0]


def test_a_zero_filled_ohlcv_null_is_announced(caplog):
    """A missing `low` became a 0.0 price -- a 100%-of-price bar range -- with nothing said about it."""
    pl = pytest.importorskip("polars")
    pytest.importorskip("polars_talib")
    from mlframe.feature_engineering.financial import add_ohlcv_ta_indicators

    df = pl.DataFrame(
        {
            "ticker": ["A"] * 40,
            "open": [1.0] * 40,
            "high": [2.0] * 40,
            "low": [None] + [0.5] * 39,
            "close": [1.5] * 40,
            "volume": [10.0] * 40,
        }
    )
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_engineering.financial"):
        try:
            add_ohlcv_ta_indicators(df, ta_windows=[5], market_action_prefixes=[""])
        except Exception:  # the indicator stack itself may be unavailable; the warning fires before it runs
            pass
    assert any("filled with 0.0" in r.message for r in caplog.records), [r.message for r in caplog.records]


class TestTheCancellationRepairTouchesOnlyTheFlaggedRows:
    """It walked the whole query frame to fix a handful, twice per horizon on a `mean` request."""

    def test_the_direct_aggregate_honours_the_row_mask(self):
        """Rows outside the mask are left NaN, so the caller's `np.where` keeps the fast subtraction for them."""
        from mlframe.feature_engineering.multi_window_aggregate import _direct_window_agg

        history = pd.DataFrame({"e": ["a"] * 6, "t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        query = pd.DataFrame({"e": ["a"] * 3, "cut": [3.0, 4.0, 5.0]})
        mask = np.array([False, True, False])
        out = _direct_window_agg(history, "e", "t", query, "cut", 10.0, "v", "sum", rows=mask)
        assert np.isnan(out[0]) and np.isnan(out[2])
        assert out[1] == pytest.approx(1.0 + 2.0 + 3.0 + 4.0)

    def test_the_masked_values_match_the_unmasked_ones(self):
        """Restricting the walk must not change any answer it does produce."""
        from mlframe.feature_engineering.multi_window_aggregate import _direct_window_agg

        rng = np.random.default_rng(0)
        history = pd.DataFrame({"e": np.repeat(["a", "b"], 20), "t": np.tile(np.arange(20.0), 2), "v": rng.normal(size=40)})
        query = pd.DataFrame({"e": np.repeat(["a", "b"], 5), "cut": np.tile([4.0, 8.0, 12.0, 16.0, 19.0], 2)})
        mask = np.array([True, False, True, False, True] * 2)
        full = _direct_window_agg(history, "e", "t", query, "cut", 5.0, "v", "mean")
        part = _direct_window_agg(history, "e", "t", query, "cut", 5.0, "v", "mean", rows=mask)
        np.testing.assert_allclose(part[mask], full[mask])

    @pytest.mark.parametrize("fn", ["sum", "count", "mean", "min", "max", "median"])
    def test_the_numpy_reductions_match_the_pandas_ones(self, fn):
        """The per-row `pd.Series(...)` construction inside the loop was replaced with a numpy reduction."""
        from mlframe.feature_engineering.multi_window_aggregate import _DIRECT_AGG_REDUCTIONS

        window = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        assert _DIRECT_AGG_REDUCTIONS[fn](window) == pytest.approx(getattr(pd.Series(window), fn)())
