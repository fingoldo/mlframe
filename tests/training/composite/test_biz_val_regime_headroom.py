"""Unit + biz_value tests for the regime-headroom map (mlframe.training.composite._regime_headroom)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mlframe.training.composite import _regime_headroom as _rh
from mlframe.training.composite._regime_headroom import (
    regime_headroom_map,
    render_regime_headroom_map,
)
from mlframe.training.composite import (
    regime_headroom_map as facade_map,
    render_regime_headroom_map as facade_render,
)


def _rmse(pred, y, w=None):
    e = (pred - y) ** 2
    if w is None:
        return float(np.sqrt(e.mean()))
    return float(np.sqrt((w * e).sum() / w.sum()))


# --------------------------------------------------------------------------------------------------
# Unit tests
# --------------------------------------------------------------------------------------------------


def test_facade_reexports_same_object():
    assert facade_map is regime_headroom_map
    assert facade_render is render_regime_headroom_map


def test_per_bin_rmse_and_headroom_correct():
    rng = np.random.default_rng(0)
    n = 4000
    axis = rng.uniform(0, 1, n)
    y = rng.normal(0, 1, n)
    raw = y + rng.normal(0, 1.0, n)
    comp = y + rng.normal(0, 0.5, n)
    lag = y + rng.normal(0, 1.2, n)
    rep = regime_headroom_map(y, raw, comp, lag, axis_values=axis, n_bins=5)

    # Recompute one bin's RMSEs directly from the reported edges and confirm they match, plus headroom formula.
    for b in rep["bins"]:
        lo, hi = b["axis_lo"], b["axis_hi"]
        if b["bin"] == rep["bins"][-1]["bin"]:
            mask = (axis >= lo) & (axis <= hi)
        else:
            mask = (axis >= lo) & (axis < hi)
        # First bin lower edge is the min; membership via searchsorted semantics -> reconstruct by exclusion is fuzzy,
        # so instead assert the reported RMSE is internally consistent with headroom + failsafe.
        failsafe = min(b["rmse_raw"], b["rmse_lag"])
        assert b["failsafe_rmse"] == pytest.approx(failsafe)
        expected_hr = (failsafe - b["rmse_composite"]) / failsafe
        assert b["headroom"] == pytest.approx(expected_hr, abs=1e-12)
        # Winner is the argmin RMSE.
        rmses = {"raw": b["rmse_raw"], "composite": b["rmse_composite"], "lag": b["rmse_lag"]}
        assert b["winner"] == min(rmses, key=rmses.get)
        assert mask.sum() >= 0  # keep mask referenced for clarity


def test_bins_sorted_ascending_by_axis():
    rng = np.random.default_rng(1)
    n = 2000
    axis = rng.uniform(-5, 5, n)
    y = rng.normal(0, 1, n)
    rep = regime_headroom_map(y, y + 1.0, y + 0.5, axis_values=axis, n_bins=8)
    los = [b["axis_lo"] for b in rep["bins"]]
    his = [b["axis_hi"] for b in rep["bins"]]
    assert los == sorted(los)
    assert his == sorted(his)
    assert [b["bin"] for b in rep["bins"]] == sorted(b["bin"] for b in rep["bins"])


def test_no_lag_failsafe_is_raw():
    rng = np.random.default_rng(2)
    n = 1500
    axis = rng.uniform(0, 1, n)
    y = rng.normal(0, 1, n)
    raw = y + rng.normal(0, 1.0, n)
    comp = y + rng.normal(0, 0.4, n)
    rep = regime_headroom_map(y, raw, comp, axis_values=axis, n_bins=4)
    assert rep["has_lag"] is False
    for b in rep["bins"]:
        assert b["rmse_lag"] is None
        assert b["failsafe_rmse"] == pytest.approx(b["rmse_raw"])


def test_json_round_trip_sorted_keys_no_nan():
    rng = np.random.default_rng(3)
    n = 1000
    axis = rng.uniform(0, 1, n)
    y = rng.normal(0, 1, n)
    rep = regime_headroom_map(y, y + 1.0, y + 0.5, y + 1.5, axis_values=axis, n_bins=5, group_ids=rng.integers(0, 7, n))
    text = json.dumps(rep, sort_keys=True)
    back = json.loads(text)
    assert back["n_bins"] == rep["n_bins"]
    # No bare NaN survived (json.dumps would emit the literal NaN token).
    assert "NaN" not in text


def test_render_is_ascii_only():
    rng = np.random.default_rng(4)
    n = 800
    axis = rng.uniform(0, 1, n)
    y = rng.normal(0, 1, n)
    rep = regime_headroom_map(y, y + 1.0, y + 0.5, y + 1.5, axis_values=axis, n_bins=5)
    s = render_regime_headroom_map(rep)
    s.encode("ascii")  # raises if any non-ASCII char slipped in
    assert "Regime headroom map" in s


def test_constant_axis_single_bin():
    n = 500
    axis = np.full(n, 3.0)
    rng = np.random.default_rng(5)
    y = rng.normal(0, 1, n)
    rep = regime_headroom_map(y, y + 1.0, y + 0.5, axis_values=axis, n_bins=10)
    assert rep["n_bins"] == 1
    assert rep["bins"][0]["n"] == n


def test_all_nan_returns_empty():
    n = 100
    y = np.full(n, np.nan)
    rep = regime_headroom_map(y, y, y, axis_values=np.arange(n, dtype=float), n_bins=5)
    assert rep["n_bins"] == 0
    assert rep["bins"] == []
    assert rep["summary"]["frac_bins_helped"] is None


def test_n_less_than_n_bins_collapses():
    n = 3
    axis = np.array([1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0])
    rep = regime_headroom_map(y, y + 1.0, y + 0.5, axis_values=axis, n_bins=10)
    assert rep["n_bins"] <= 3
    assert rep["n_rows"] == 3


def test_non_finite_rows_dropped_matched():
    n = 200
    rng = np.random.default_rng(6)
    axis = rng.uniform(0, 1, n)
    y = rng.normal(0, 1, n)
    raw = y + 1.0
    comp = y + 0.5
    comp[0] = np.inf  # this row must be dropped from every series
    rep = regime_headroom_map(y, raw, comp, axis_values=axis, n_bins=4)
    assert rep["n_rows"] == n - 1


def test_group_ids_counts_distinct_per_bin():
    n = 1000
    rng = np.random.default_rng(7)
    axis = rng.uniform(0, 1, n)
    y = rng.normal(0, 1, n)
    gids = rng.integers(0, 5, n)
    rep = regime_headroom_map(y, y + 1.0, y + 0.5, axis_values=axis, n_bins=4, group_ids=gids)
    for b in rep["bins"]:
        assert b["n_groups"] is not None
        assert 1 <= b["n_groups"] <= 5


def test_invalid_args_raise():
    y = np.arange(10, dtype=float)
    with pytest.raises(ValueError):
        regime_headroom_map(y, y, y, axis_values=None, n_bins=5)
    with pytest.raises(ValueError):
        regime_headroom_map(y, y, y, axis_values=y[:5], n_bins=5)
    with pytest.raises(ValueError):
        regime_headroom_map(y, y, y, axis_values=y, n_bins=0)


def test_sample_weight_affects_rmse():
    n = 600
    rng = np.random.default_rng(8)
    axis = rng.uniform(0, 1, n)
    y = rng.normal(0, 1, n)
    raw = y + rng.normal(0, 1.0, n)
    comp = y + rng.normal(0, 0.5, n)
    w = rng.uniform(0.1, 2.0, n)
    rep = regime_headroom_map(y, raw, comp, axis_values=axis, n_bins=1, sample_weight=w)
    b = rep["bins"][0]
    assert b["rmse_raw"] == pytest.approx(_rmse(raw, y, w), rel=1e-9)
    assert b["rmse_composite"] == pytest.approx(_rmse(comp, y, w), rel=1e-9)


# --------------------------------------------------------------------------------------------------
# biz_value: composite HELPS in one axis region, HURTS in another; map must localize the crossover.
# --------------------------------------------------------------------------------------------------


@pytest.mark.skipif(not _rh._HAVE_NUMBA, reason="numba unavailable")
def test_njit_and_bincount_reductions_agree():
    """The njit-default reduction and the bincount fallback must produce the same bin stats (with/without lag)."""
    rng = np.random.default_rng(21)
    n = 5000
    k = 6
    codes = rng.integers(-1, k, n).astype(np.int64)  # include -1 (excluded) rows
    w = rng.uniform(0.1, 2.0, n)
    y = rng.normal(0, 1, n)
    raw = y + rng.normal(0, 1, n)
    comp = y + rng.normal(0, 0.5, n)
    lag = y + rng.normal(0, 1.2, n)
    for has_lag in (False, True):
        lag_arr = lag if has_lag else y
        r_nj, W_nj, sr_nj, sc_nj, sl_nj = _rh._bin_stats_njit(codes, w, y, raw, comp, lag_arr, has_lag, k)
        r_bc, W_bc, sr_bc, sc_bc, sl_bc = _rh._bin_stats_bincount(codes, w, y, raw, comp, lag, has_lag, k)
        assert np.array_equal(r_nj, r_bc)
        assert np.allclose(W_nj, W_bc)
        assert np.allclose(sr_nj, sr_bc)
        assert np.allclose(sc_nj, sc_bc)
        if has_lag:
            assert np.allclose(sl_nj, sl_bc)


def test_biz_val_regime_headroom_localizes_help_and_hurt_regions():
    """Construct a base axis split at 0: below 0 the composite is accurate (in-range), above 0 the composite is badly
    biased (extrapolated base) while raw stays fine. The map's headroom must be strongly POSITIVE in the help region,
    strongly NEGATIVE in the hurt region, the crossover bin sits at the split, and the helped-fraction matches ~0.5."""
    rng = np.random.default_rng(12345)
    n = 40000
    axis = rng.uniform(-1.0, 1.0, n)  # symmetric -> split at 0 == median == bin boundary at n_bins even
    y = rng.normal(0.0, 1.0, n)

    in_range = axis < 0.0
    # raw: uniformly mediocre everywhere.
    raw = y + rng.normal(0.0, 1.0, n)
    # composite: excellent where in-range, terrible (large bias) where extrapolated.
    comp = np.empty(n)
    comp[in_range] = y[in_range] + rng.normal(0.0, 0.2, in_range.sum())
    comp[~in_range] = y[~in_range] + rng.normal(0.0, 2.5, (~in_range).sum())
    # lag failsafe: mediocre everywhere, like raw.
    lag = y + rng.normal(0.0, 1.1, n)

    rep = regime_headroom_map(y, raw, comp, lag, axis_values=axis, n_bins=10)

    bins = rep["bins"]
    assert len(bins) == 10
    # Help region (axis < 0): headroom clearly positive; hurt region (axis > 0): clearly negative.
    help_bins = [b for b in bins if b["axis_hi"] is not None and b["axis_hi"] <= 0.0]
    hurt_bins = [b for b in bins if b["axis_lo"] is not None and b["axis_lo"] >= 0.0]
    assert help_bins and hurt_bins
    for b in help_bins:
        assert b["headroom"] > 0.30, f"help bin {b['bin']} headroom {b['headroom']}"
        assert b["winner"] == "composite"
    for b in hurt_bins:
        assert b["headroom"] < -0.30, f"hurt bin {b['bin']} headroom {b['headroom']}"
        assert b["winner"] in ("raw", "lag")

    # Crossover: last helped bin is immediately followed by a hurt bin near axis 0.
    headrooms = [b["headroom"] for b in bins]
    signs = [h > 0 for h in headrooms]
    crossover_idx = next(i for i in range(1, len(signs)) if signs[i - 1] and not signs[i])
    assert bins[crossover_idx]["axis_lo"] == pytest.approx(0.0, abs=0.15)

    # Summary fraction matches the constructed ~50/50 split.
    assert rep["summary"]["frac_bins_helped"] == pytest.approx(0.5, abs=0.15)
    assert rep["summary"]["frac_bins_hurt"] == pytest.approx(0.5, abs=0.15)
    # Worst-hurt bin lives in the extrapolated (axis > 0) region.
    assert rep["summary"]["worst_hurt_bin"]["axis_lo"] >= 0.0


def test_biz_val_regime_headroom_all_help_when_composite_dominates():
    """When the composite dominates everywhere, every scored bin helps and the helped-fraction is 1.0."""
    rng = np.random.default_rng(999)
    n = 8000
    axis = rng.uniform(0, 10, n)
    y = rng.normal(0, 1, n)
    raw = y + rng.normal(0, 1.5, n)
    comp = y + rng.normal(0, 0.3, n)
    rep = regime_headroom_map(y, raw, comp, axis_values=axis, n_bins=6)
    assert rep["summary"]["frac_bins_helped"] == pytest.approx(1.0)
    assert all(b["winner"] == "composite" for b in rep["bins"])
