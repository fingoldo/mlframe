"""Unit + biz_value tests for causal temporal base-column engineering.

Covers: causal lag shift correctness, causal trailing rolling window (current row
excluded), NaN head handling, no-future-leak (a future-only change cannot perturb an
earlier engineered value), original-row-order remapping after a time sort, and the
biz_value claim -- the auto lag-1 base lets discovery's leakage screen pass (NOT flagged
leaky) on an AR(1) target where a same-time (shift-0) base WOULD be flagged.
"""

from __future__ import annotations

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from mlframe.training.composite.discovery._base_engineering import (
    add_engineered_bases_to_pool,
    engineer_temporal_bases,
)
from mlframe.training.composite.discovery._leakage import detect_base_target_leakage


def _frame(y, t=None):
    """Frame."""
    n = len(y)
    t = np.arange(n) if t is None else np.asarray(t)
    return pl.DataFrame({"y": np.asarray(y, dtype=np.float64), "t": t})


def test_causal_lag_shift_correctness():
    """Causal lag shift correctness."""
    y = np.arange(10.0)
    bases = engineer_temporal_bases(_frame(y), "y", "t", lags=(1, 2), ops=("lag",))
    lag1 = bases["y_lag1"]
    lag2 = bases["y_lag2"]
    assert np.isnan(lag1[0])
    assert np.allclose(lag1[1:], y[:-1])  # row i sees y[i-1]
    assert np.isnan(lag2[0]) and np.isnan(lag2[1])
    assert np.allclose(lag2[2:], y[:-2])  # row i sees y[i-2]


def test_rolling_window_is_causal_excludes_current_row():
    """Rolling window is causal excludes current row."""
    y = np.arange(1.0, 9.0)  # 1..8
    bases = engineer_temporal_bases(_frame(y), "y", "t", rolling_windows=(3,), ops=("rolling_mean",))
    rm = bases["y_rollmean3"]
    assert np.all(np.isnan(rm[:3]))  # first window rows have no full strictly-past window
    # row 3 averages y[0:3] = 1,2,3 -> 2.0 ; current row (y[3]=4) excluded.
    assert rm[3] == pytest.approx(2.0)
    assert rm[4] == pytest.approx(3.0)  # y[1:4]=2,3,4
    # explicit no-current-row check: removing current value never changes the mean
    for i in range(3, len(y)):
        assert rm[i] == pytest.approx(y[i - 3 : i].mean())


def test_rolling_median_causal():
    """Rolling median causal."""
    y = np.array([10.0, 1.0, 2.0, 100.0, 3.0])
    bases = engineer_temporal_bases(_frame(y), "y", "t", rolling_windows=(3,), ops=("rolling_median",))
    rmed = bases["y_rollmedian3"]
    assert np.all(np.isnan(rmed[:3]))
    assert rmed[3] == pytest.approx(np.median([10.0, 1.0, 2.0]))
    assert rmed[4] == pytest.approx(np.median([1.0, 2.0, 100.0]))


def test_diff_is_lagged_not_same_time():
    """Diff is lagged not same time."""
    y = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    bases = engineer_temporal_bases(_frame(y), "y", "t", ops=("diff",))
    d = bases["y_diff1"]
    assert np.isnan(d[0]) and np.isnan(d[1])
    # row i = y[i-1]-y[i-2]; never uses y[i]
    assert d[2] == pytest.approx(y[1] - y[0])  # 1-0
    assert d[3] == pytest.approx(y[2] - y[1])  # 3-1
    assert d[4] == pytest.approx(y[3] - y[2])  # 6-3


def test_nan_head_handling():
    """Nan head handling."""
    y = np.arange(6.0)
    bases = engineer_temporal_bases(
        _frame(y),
        "y",
        "t",
        lags=(1, 2),
        rolling_windows=(3,),
        ops=("lag", "rolling_mean", "diff"),
    )
    assert int(np.isnan(bases["y_lag1"]).sum()) == 1
    assert int(np.isnan(bases["y_lag2"]).sum()) == 2
    assert int(np.isnan(bases["y_rollmean3"]).sum()) == 3
    assert int(np.isnan(bases["y_diff1"]).sum()) == 2


def test_no_future_leak_perturbation():
    """Changing a FUTURE target value must not alter any earlier engineered base value."""
    n = 20
    y = np.linspace(0.0, 5.0, n)
    base = engineer_temporal_bases(
        _frame(y),
        "y",
        "t",
        lags=(1, 2),
        rolling_windows=(3, 5),
        ops=("lag", "rolling_mean", "rolling_median", "diff"),
    )
    y2 = y.copy()
    y2[15:] += 1000.0  # perturb only rows 15..19
    pert = engineer_temporal_bases(
        _frame(y2),
        "y",
        "t",
        lags=(1, 2),
        rolling_windows=(3, 5),
        ops=("lag", "rolling_mean", "rolling_median", "diff"),
    )
    for name in base:
        a, b = base[name], pert[name]
        # rows 0..14 depend only on strictly-past (unchanged) targets -> identical
        np.testing.assert_array_equal(np.nan_to_num(a[:15]), np.nan_to_num(b[:15]))


def test_remaps_to_original_row_order():
    # Shuffle time so frame order != time order; engineered bases must align to frame rows.
    """Remaps to original row order."""
    rng = np.random.default_rng(0)
    n = 30
    t = rng.permutation(n)
    y = t.astype(np.float64) * 2.0  # y deterministic in time
    bases = engineer_temporal_bases(_frame(y, t), "y", "t", lags=(1,), ops=("lag",))
    lag1 = bases["y_lag1"]
    # For the row whose time is the smallest, lag1 must be NaN (no past).
    first_time_row = int(np.argmin(t))
    assert np.isnan(lag1[first_time_row])
    # For a row at time tau>0, lag1 should equal y of the row at time tau-1.
    for i in range(n):
        tau = t[i]
        if tau == 0:
            continue
        prev_row = int(np.where(t == tau - 1)[0][0])
        assert lag1[i] == pytest.approx(y[prev_row])


def test_invalid_args():
    """Invalid args."""
    f = _frame(np.arange(5.0))
    with pytest.raises(ValueError):
        engineer_temporal_bases(f, "y", "t", lags=(0,), ops=("lag",))
    with pytest.raises(ValueError):
        engineer_temporal_bases(f, "y", "t", rolling_windows=(0,), ops=("rolling_mean",))
    with pytest.raises(ValueError):
        engineer_temporal_bases(f, "y", "t", ops=("bogus",))


def test_add_to_pool_merges_without_clobbering():
    """Add to pool merges without clobbering."""
    y = np.arange(8.0)
    existing = {"my_base": np.ones(8)}
    merged = add_engineered_bases_to_pool(existing, _frame(y), "y", "t", lags=(1,), ops=("lag",))
    assert "my_base" in merged and "y_lag1" in merged
    np.testing.assert_array_equal(merged["my_base"], existing["my_base"])  # user base preserved


def test_biz_value_engineered_lag1_not_flagged_leaky_but_same_time_is():
    """AR(1): auto lag-1 base passes leakage screen; same-time (shift-0) base is flagged.

    This is the value proposition -- the user need not pre-supply the lag, and the
    mechanically-engineered base is causal, so detect_base_target_leakage clears it while
    rejecting a same-time near-identity base on the very same target.
    """
    rng = np.random.default_rng(42)
    n = 800
    y = np.empty(n)
    y[0] = 0.0
    for i in range(1, n):
        y[i] = 0.9 * y[i - 1] + rng.normal(0.0, 0.3)
    t = np.arange(n)
    bases = engineer_temporal_bases(_frame(y, t), "y", "t", lags=(1,), ops=("lag",))
    lag1 = bases["y_lag1"]

    # Same-time leaky base: y itself (shift 0) -- a perfect re-encoding of the target.
    same_time = y.copy()

    v_lag = detect_base_target_leakage(y, lag1, time_ordering=t)
    v_same = detect_base_target_leakage(y, same_time, time_ordering=t)

    assert not v_lag["is_leaky"], f"engineered lag-1 wrongly flagged leaky: {v_lag}"
    assert v_same["is_leaky"], f"same-time base should be leaky: {v_same}"
