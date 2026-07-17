"""Unit + biz_value tests for strictly-causal PER-GROUP base engineering (``_grouped_causal_bases``) and its wiring
into ``CompositeTargetDiscovery.fit``.

The engineered per-group lag / trailing-mean / expanding-mean bases are causal by construction (shift >= 1 WITHIN each
group), so on a group-sequential target the additive/diff inverse ``y = T_hat + y_prev`` reconstructs y from a per-row
REAL previous value and does not collapse on unseen groups. These tests pin: causal correctness, group-boundary
isolation, order-key handling, polars/pandas parity, the discovery-pool wiring (present when a group key is configured,
absent when engineering is off), and a quantitative group-disjoint reconstruction floor.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training._composite_target_discovery_config import CompositeTargetDiscoveryConfig
from mlframe.training.composite.discovery import CompositeTargetDiscovery
from mlframe.training.composite.discovery._grouped_causal_bases import (
    attach_grouped_causal_bases,
    engineer_grouped_causal_bases,
    maybe_add_grouped_causal_bases,
)


@pytest.fixture(scope="module", autouse=True)
def _warm_numba() -> None:
    """Trigger the @njit kernel compile once so per-test wall stays tiny (numba caches to disk thereafter)."""
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "g": ["a", "a", "a"], "md": [0, 1, 2]})
    engineer_grouped_causal_bases(df, "y", "g", "md", lags=(1, 2), trailing_windows=(2,))


def _mixed_frame() -> pd.DataFrame:
    # Two groups, deliberately unordered rows, explicit md order key.
    # well A: md 0->10, 1->20, 2->30 ; well B: md 0->100, 1->200
    return pd.DataFrame(
        {
            "y": [30, 100, 10, 200, 20],
            "well": ["A", "B", "A", "B", "A"],
            "md": [2, 0, 0, 1, 1],
        }
    )


# --------------------------------------------------------------------------- causal correctness


def test_grouped_lag1_equals_previous_within_group_ordered_by_key():
    df = _mixed_frame()
    out = engineer_grouped_causal_bases(df, "y", "well", "md", lags=(1,), ops=("lag",))
    # Original row order [A md2=30, B md0=100, A md0=10, B md1=200, A md1=20]; each lag1 is the previous IN-GROUP value.
    np.testing.assert_array_equal(out["y__gcausal_lag1"], [20, 100, 10, 100, 10])


def test_grouped_lag2_and_first_fill_group_first():
    df = _mixed_frame()
    out = engineer_grouped_causal_bases(df, "y", "well", "md", lags=(2,), ops=("lag",), first_fill="group_first")
    # lag2 sorted: A=[10(fill),10(fill),10] B=[100(fill),100(fill)] -> back-mapped.
    np.testing.assert_array_equal(out["y__gcausal_lag2"], [10, 100, 10, 100, 10])


def test_first_fill_nan_leaves_history_less_head_nan():
    df = _mixed_frame()
    out = engineer_grouped_causal_bases(df, "y", "well", "md", lags=(1,), ops=("lag",), first_fill="nan")
    lag = out["y__gcausal_lag1"]
    # First in-group rows (A md0 at original idx2, B md0 at original idx1) have no predecessor -> NaN.
    assert np.isnan(lag[1]) and np.isnan(lag[2])
    np.testing.assert_array_equal(lag[[0, 3, 4]], [20, 100, 10])


def test_trailing_and_expanding_mean_are_strictly_past():
    df = _mixed_frame()
    out = engineer_grouped_causal_bases(df, "y", "well", "md", trailing_windows=(2,), ops=("trailing_mean", "expanding_mean"))
    # A sorted y=[10,20,30]: tmean2=[10(fill),mean(10)=10,mean(10,20)=15]; expmean=[10(fill),10,15].
    np.testing.assert_allclose(out["y__gcausal_tmean2"], [15, 100, 10, 100, 10])
    np.testing.assert_allclose(out["y__gcausal_expmean"], [15, 100, 10, 100, 10])


def test_expanding_mean_long_group_matches_running_mean_of_past():
    # A single long group; expanding mean at row i must equal mean(y[:i]).
    y = np.arange(1, 21, dtype=float)
    df = pd.DataFrame({"y": y, "g": ["x"] * 20, "md": np.arange(20)})
    out = engineer_grouped_causal_bases(df, "y", "g", "md", ops=("expanding_mean",))
    exp = out["y__gcausal_expmean"]
    assert exp[0] == y[0]  # first-row fill
    for i in range(1, 20):
        assert exp[i] == pytest.approx(y[:i].mean())


def test_trailing_window_matches_manual_rolling_mean_of_strict_past():
    y = np.array([5.0, 7.0, 11.0, 2.0, 9.0, 4.0, 8.0])
    df = pd.DataFrame({"y": y, "g": ["x"] * len(y), "md": np.arange(len(y))})
    w = 3
    out = engineer_grouped_causal_bases(df, "y", "g", "md", trailing_windows=(w,), ops=("trailing_mean",))
    tmean = out[f"y__gcausal_tmean{w}"]
    assert tmean[0] == y[0]
    for i in range(1, len(y)):
        lo = max(0, i - w)
        assert tmean[i] == pytest.approx(y[lo:i].mean())


# --------------------------------------------------------------------------- group boundaries / ordering


def test_group_boundaries_never_leak_across_entities():
    # Interleaved groups: if the lag ignored group boundaries it would pull the other group's value.
    df = pd.DataFrame(
        {
            "y": [1.0, 100.0, 2.0, 200.0, 3.0, 300.0],
            "g": ["A", "B", "A", "B", "A", "B"],
            "md": [0, 0, 1, 1, 2, 2],
        }
    )
    out = engineer_grouped_causal_bases(df, "y", "g", "md", lags=(1,), ops=("lag",))
    # A sorted y=[1,2,3] lag1=[1,1,2]; B sorted y=[100,200,300] lag1=[100,100,200]; interleaved back.
    np.testing.assert_array_equal(out["y__gcausal_lag1"], [1, 100, 1, 100, 2, 200])


def test_unordered_input_is_sorted_by_order_key():
    # Same data, shuffled rows -> result must be identical after back-mapping (sort by md restores causal order).
    base = pd.DataFrame({"y": [10.0, 20.0, 30.0, 40.0], "g": ["a"] * 4, "md": [0, 1, 2, 3]})
    perm = [2, 0, 3, 1]
    shuffled = base.iloc[perm].reset_index(drop=True)
    out_base = engineer_grouped_causal_bases(base, "y", "g", "md", lags=(1,), ops=("lag",))["y__gcausal_lag1"]
    out_shuf = engineer_grouped_causal_bases(shuffled, "y", "g", "md", lags=(1,), ops=("lag",))["y__gcausal_lag1"]
    # out_shuf is in shuffled row order; reorder back to base order to compare.
    np.testing.assert_array_equal(out_shuf, out_base[perm])


def test_none_time_column_uses_frame_row_order():
    # No order key -> the given per-group frame row order is the causal order.
    df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 9.0, 4.0], "g": ["A", "A", "A", "C", "A"]})
    out = engineer_grouped_causal_bases(df, "y", "g", None, lags=(1,), ops=("lag",))
    # A rows in frame order y=[1,2,3,4]; C single row y=9. lag1: A=[1,1,2,3], C=[9]; back-mapped.
    np.testing.assert_array_equal(out["y__gcausal_lag1"], [1, 1, 2, 9, 3])


def test_single_row_group_fills_own_value():
    df = pd.DataFrame({"y": [5.0, 6.0, 7.0], "g": ["solo", "pair", "pair"], "md": [0, 0, 1]})
    out = engineer_grouped_causal_bases(df, "y", "g", "md", lags=(1,), trailing_windows=(2,), ops=("lag", "trailing_mean", "expanding_mean"))
    assert out["y__gcausal_lag1"][0] == 5.0
    assert out["y__gcausal_tmean2"][0] == 5.0
    assert out["y__gcausal_expmean"][0] == 5.0


# --------------------------------------------------------------------------- carrier parity


def test_polars_and_pandas_parity():
    pdf = _mixed_frame()
    pldf = pl.from_pandas(pdf)
    kw = dict(lags=(1, 2), trailing_windows=(2, 3), ops=("lag", "trailing_mean", "expanding_mean"))
    out_pd = engineer_grouped_causal_bases(pdf, "y", "well", "md", **kw)
    out_pl = engineer_grouped_causal_bases(pldf, "y", "well", "md", **kw)
    assert out_pd.keys() == out_pl.keys()
    for k in out_pd:
        np.testing.assert_allclose(out_pd[k], out_pl[k])


def test_string_and_numeric_group_keys_equivalent():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    df_str = pd.DataFrame({"y": y, "g": ["a", "b", "a", "b"], "md": [0, 0, 1, 1]})
    df_num = pd.DataFrame({"y": y, "g": [10, 20, 10, 20], "md": [0, 0, 1, 1]})
    a = engineer_grouped_causal_bases(df_str, "y", "g", "md", lags=(1,), ops=("lag",))["y__gcausal_lag1"]
    b = engineer_grouped_causal_bases(df_num, "y", "g", "md", lags=(1,), ops=("lag",))["y__gcausal_lag1"]
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(ops=("bogus",)),
        dict(lags=(0,)),
        dict(lags=(-1,)),
        dict(trailing_windows=(0,)),
        dict(first_fill="backfill"),
    ],
)
def test_invalid_params_raise(kwargs):
    df = _mixed_frame()
    with pytest.raises(ValueError):
        engineer_grouped_causal_bases(df, "y", "well", "md", **kwargs)


def test_mismatched_order_key_length_raises():
    # dict carrier with a short ``md`` trips the time-column length guard (protects against misaligned exotic carriers).
    bad = {"y": np.array([1.0, 2.0, 3.0]), "g": np.array([0, 0, 0]), "md": np.array([0.0, 1.0])}
    with pytest.raises(ValueError, match="time column"):
        engineer_grouped_causal_bases(bad, "y", "g", "md", lags=(1,), ops=("lag",))


def test_mismatched_group_length_raises():
    bad = {"y": np.array([1.0, 2.0, 3.0]), "g": np.array([0, 0]), "md": np.array([0.0, 1.0, 2.0])}
    with pytest.raises(ValueError, match="group column"):
        engineer_grouped_causal_bases(bad, "y", "g", "md", lags=(1,), ops=("lag",))


# --------------------------------------------------------------------------- attach / memory safety


def test_attach_does_not_mutate_caller_and_skips_existing():
    df = _mixed_frame()
    cols_before = list(df.columns)
    df2, names = attach_grouped_causal_bases(df, "y", "well", "md", lags=(1,), ops=("lag",))
    assert list(df.columns) == cols_before  # caller frame untouched (shallow copy)
    assert "y__gcausal_lag1" in names and "y__gcausal_lag1" in df2.columns
    # Re-attaching onto the augmented frame is a no-op for the already-present name.
    _df3, names3 = attach_grouped_causal_bases(df2, "y", "well", "md", lags=(1,), ops=("lag",))
    assert names3 == []


def test_attach_polars_zero_copy_semantics():
    df = pl.from_pandas(_mixed_frame())
    df2, _names = attach_grouped_causal_bases(df, "y", "well", "md", lags=(1,), ops=("lag",))
    assert "y__gcausal_lag1" not in df.columns  # original polars frame is immutable / untouched
    assert "y__gcausal_lag1" in df2.columns


# --------------------------------------------------------------------------- wiring into discovery


class _FakeConfig:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeDisc:
    def __init__(self, config):
        self.config = config


def test_maybe_add_wires_bases_when_group_configured():
    df = _mixed_frame()
    disc = _FakeDisc(_FakeConfig(engineer_causal_bases=True, engineer_causal_group_column="well", time_column="md"))
    new_df, feats, bases = maybe_add_grouped_causal_bases(disc, df, "y", ["md"], [], np.arange(len(df)))
    assert "y__gcausal_lag1" in bases and "y__gcausal_lag1" in feats and "y__gcausal_lag1" in new_df.columns


def test_maybe_add_noop_when_disabled_or_no_group():
    df = _mixed_frame()
    disc_off = _FakeDisc(_FakeConfig(engineer_causal_bases=False, engineer_causal_group_column="well", time_column="md"))
    d1, _f1, b1 = maybe_add_grouped_causal_bases(disc_off, df, "y", ["md"], [], np.arange(len(df)))
    assert b1 == [] and "y__gcausal_lag1" not in d1.columns
    disc_nogrp = _FakeDisc(_FakeConfig(engineer_causal_bases=True, engineer_causal_group_column=None, time_column="md"))
    d2, _f2, b2 = maybe_add_grouped_causal_bases(disc_nogrp, df, "y", ["md"], [], np.arange(len(df)))
    assert b2 == [] and "y__gcausal_lag1" not in d2.columns


# --------------------------------------------------------------------------- biz_value


def _group_sequential_frame(seed: int, n_groups: int = 40, per: int = 40):
    """Per-group random-walk target: strong within-group AR(1), widely varying per-group level. The per-group causal
    lag_1 is a near-perfect predictor; any group-agnostic model of raw y is dominated by the between-group level spread.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        y = rng.normal(0, 50.0)  # per-group level, wide spread
        for t in range(per):
            y = y + rng.normal(0, 1.0)  # within-group AR(1) step
            rows.append((f"w{g}", t, y, rng.normal(), rng.normal()))
    return pd.DataFrame(rows, columns=["well", "md", "y", "f0", "f1"])


def _disc_config(engineer: bool) -> CompositeTargetDiscoveryConfig:
    return CompositeTargetDiscoveryConfig(
        enabled=True,
        time_column="md",
        engineer_causal_group_column="well",
        engineer_causal_bases=engineer,
        base_candidates="auto",
        screening="mi",
        mi_sample_n=1000,
        random_state=0,
        transforms=["diff", "additive_residual", "linear_residual"],
        structural_fragility_gate_enabled=False,
        yscale_holdout_gate_enabled=False,
        multi_base_enabled=False,
        interaction_base_discovery_enabled=False,
        auto_chain_discovery_enabled=False,
        mi_gain_fdr_control=False,
    )


def test_biz_val_grouped_causal_lag_enters_pool_and_reconstructs_on_disjoint_holdout():
    """The per-group causal lag base is DISCOVERED (enters the pool) when engineering is ON and ABSENT when OFF; and the
    diff-composite it enables reconstructs y on a GROUP-DISJOINT holdout far better than the raw-y (group-agnostic) model.

    Measured: RMSE_lag/RMSE_raw ~= 0.018 (i.e. ~54x better). Floor 0.05 (~2.7x margin) catches a lag-base regression
    (a broken/leaky base collapses the ratio toward 1.0) without tripping on seed noise.
    """
    df = _group_sequential_frame(seed=7)
    train_wells = {f"w{g}" for g in range(30)}
    train_idx = np.array([i for i, w in enumerate(df["well"]) if w in train_wells])
    hold_idx = np.array([i for i, w in enumerate(df["well"]) if w not in train_wells])
    time_ordering = df["md"].to_numpy()

    disc_on = CompositeTargetDiscovery(_disc_config(True)).fit(
        df=df,
        target_col="y",
        feature_cols=["f0", "f1"],
        train_idx=train_idx,
        time_ordering=time_ordering,
    )
    pool_on = getattr(disc_on, "_auto_base_pool", {})
    assert "y__gcausal_lag1" in pool_on, "engineered causal lag base must enter the candidate pool when ON"
    discovered_bases = {r.get("base_column") for r in disc_on.report_}
    assert "y__gcausal_lag1" in discovered_bases, "a spec on the causal lag base must be evaluated/discovered"
    assert any(s.base_column == "y__gcausal_lag1" for s in disc_on.specs_), "a causal-lag composite spec must be KEPT"

    disc_off = CompositeTargetDiscovery(_disc_config(False)).fit(
        df=df,
        target_col="y",
        feature_cols=["f0", "f1"],
        train_idx=train_idx,
        time_ordering=time_ordering,
    )
    pool_off = getattr(disc_off, "_auto_base_pool", {})
    assert "y__gcausal_lag1" not in pool_off, "no causal lag base may exist with engineering OFF"
    assert not any("gcausal" in str(r.get("base_column")) for r in disc_off.report_)

    # Diff-composite reconstruction floor on the group-disjoint holdout: predicting the diff T=y-lag as 0 gives
    # y_hat = lag (the additive inverse), a per-row REAL previous value -> in-range by construction, no group collapse.
    lag = engineer_grouped_causal_bases(df, "y", "well", "md", lags=(1,), ops=("lag",))["y__gcausal_lag1"]
    y = df["y"].to_numpy()
    rmse_raw = float(np.sqrt(np.mean((y[hold_idx] - y[train_idx].mean()) ** 2)))
    rmse_lag = float(np.sqrt(np.mean((y[hold_idx] - lag[hold_idx]) ** 2)))
    ratio = rmse_lag / rmse_raw
    assert ratio <= 0.05, f"causal lag reconstruction ratio {ratio:.4f} should be <= 0.05 (raw={rmse_raw:.2f}, lag={rmse_lag:.2f})"


def test_biz_val_reconstruction_is_stable_across_seeds():
    ratios = []
    for seed in (1, 2, 3):
        df = _group_sequential_frame(seed=seed, n_groups=20, per=30)
        train_idx = np.array([i for i, w in enumerate(df["well"]) if int(w[1:]) < 15])
        hold_idx = np.array([i for i, w in enumerate(df["well"]) if int(w[1:]) >= 15])
        lag = engineer_grouped_causal_bases(df, "y", "well", "md", lags=(1,), ops=("lag",))["y__gcausal_lag1"]
        y = df["y"].to_numpy()
        rmse_raw = np.sqrt(np.mean((y[hold_idx] - y[train_idx].mean()) ** 2))
        rmse_lag = np.sqrt(np.mean((y[hold_idx] - lag[hold_idx]) ** 2))
        ratios.append(rmse_lag / rmse_raw)
    assert max(ratios) <= 0.08, f"reconstruction ratio unstable across seeds: {ratios}"


# --------------------------------------------------------------------------- profile smoke (cProfile verdict in module docstring)


def test_profile_smoke_representative_shape_is_fast():
    rng = np.random.default_rng(0)
    n, g = 50_000, 200
    df = pd.DataFrame({"y": rng.normal(size=n), "well": rng.integers(0, g, n), "md": rng.random(n)})
    t0 = time.perf_counter()
    out = engineer_grouped_causal_bases(df, "y", "well", "md", lags=(1, 2), trailing_windows=(3,))
    dt = time.perf_counter() - t0
    assert set(out) == {"y__gcausal_lag1", "y__gcausal_lag2", "y__gcausal_tmean3", "y__gcausal_expmean"}
    assert dt < 4.0, f"engineering 50k/200-group took {dt:.2f}s (numba warm expected << 1s)"
