"""Unit + biz_value tests for the composite-target VALUE report.

``build_composite_value_report`` answers "did the composite earn its keep, and where?" -- a per-group
RMSE breakdown (raw vs composite vs lag failsafe), the row-weighted net lift, the helped/hurt/tied
tallies, and the count of groups where the composite is worse than the lag failsafe. The biz_value tests
engineer a synthetic with a KNOWN helped subset and a KNOWN hurt subset and assert the report recovers
both, the worse-than-lag set, and the sign of the net lift with quantitative floors.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mlframe.training.composite import _value_report as vr_mod
from mlframe.training.composite._value_report import (
    build_composite_value_report,
    render_composite_value_report,
)
from mlframe.training.composite.report import composite_value_report


# --------------------------------------------------------------------------- unit: per-group RMSE


def test_per_group_rmse_correctness():
    # Group A: y=[0,0], raw=[2,2] -> rmse 2; comp=[1,1] -> rmse 1. Group B: y=[0], raw=[1], comp=[2].
    """Per group rmse correctness."""
    y = np.array([0.0, 0.0, 0.0])
    raw = np.array([2.0, 2.0, 1.0])
    comp = np.array([1.0, 1.0, 2.0])
    g = np.array([0, 0, 1])
    rep = build_composite_value_report(y, raw, comp, g)
    by = {e["group"]: e for e in rep["per_group"]}
    assert by[0]["rmse_raw"] == pytest.approx(2.0)
    assert by[0]["rmse_composite"] == pytest.approx(1.0)
    assert by[0]["lift_over_raw"] == pytest.approx(0.5)
    assert by[0]["n"] == 2
    assert by[1]["rmse_raw"] == pytest.approx(1.0)
    assert by[1]["rmse_composite"] == pytest.approx(2.0)


def test_helped_hurt_tied_classification():
    # helped: comp err < raw err; hurt: comp err > raw err; tied: identical.
    """Helped hurt tied classification."""
    y = np.zeros(6)
    raw = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    comp = np.array([1.0, 1.0, 3.0, 3.0, 2.0, 2.0])
    g = np.array([0, 0, 1, 1, 2, 2])
    rep = build_composite_value_report(y, raw, comp, g)
    by = {e["group"]: e for e in rep["per_group"]}
    assert by[0]["verdict_vs_raw"] == "helped"
    assert by[1]["verdict_vs_raw"] == "hurt"
    assert by[2]["verdict_vs_raw"] == "tied"
    vr = rep["aggregate"]["vs_raw"]
    assert (vr["helped"], vr["hurt"], vr["tied"]) == (1, 1, 1)
    assert vr["helped_frac"] == pytest.approx(1 / 3)


def test_row_weighted_net_lift():
    # Group A (3 rows) lift +0.5; group B (1 row) lift -1.0. Net = (3*0.5 + 1*-1)/4 = 0.125.
    """Row weighted net lift."""
    y = np.zeros(4)
    raw = np.array([2.0, 2.0, 2.0, 1.0])
    comp = np.array([1.0, 1.0, 1.0, 2.0])
    g = np.array([0, 0, 0, 1])
    rep = build_composite_value_report(y, raw, comp, g)
    assert rep["aggregate"]["net_weighted_lift_over_raw"] == pytest.approx(0.125)
    # Pooled is SSE-based over ALL rows: raw errs [2,2,2,1] -> sqrt(13/4); comp errs [1,1,1,2] -> sqrt(7/4).
    assert rep["aggregate"]["pooled_rmse_raw"] == pytest.approx(np.sqrt(3.25))
    assert rep["aggregate"]["pooled_rmse_composite"] == pytest.approx(np.sqrt(1.75))


def test_worse_than_lag_group_count():
    # Group A: comp rmse 2 > lag rmse 1 -> worse. Group B: comp 0.5 < lag 1 -> not worse.
    """Worse than lag group count."""
    y = np.zeros(4)
    raw = np.array([3.0, 3.0, 3.0, 3.0])
    comp = np.array([2.0, 2.0, 0.5, 0.5])
    lag = np.array([1.0, 1.0, 1.0, 1.0])
    g = np.array([0, 0, 1, 1])
    rep = build_composite_value_report(y, raw, comp, g, y_pred_lag=lag)
    agg = rep["aggregate"]
    assert agg["n_worse_than_lag"] == 1
    assert agg["worse_than_lag_groups"] == [0]
    by = {e["group"]: e for e in rep["per_group"]}
    assert by[0]["worse_than_lag"] is True
    assert by[1]["worse_than_lag"] is False
    assert by[0]["rmse_lag"] == pytest.approx(1.0)


def test_winner_and_lag_lift():
    """Winner and lag lift."""
    y = np.zeros(2)
    raw = np.array([2.0, 2.0])
    comp = np.array([0.5, 0.5])
    lag = np.array([1.0, 1.0])
    g = np.array([7, 7])
    rep = build_composite_value_report(y, raw, comp, g, y_pred_lag=lag)
    e = rep["per_group"][0]
    assert e["winner"] == "composite"
    assert e["lift_over_lag"] == pytest.approx(0.5)  # (1.0 - 0.5) / 1.0
    assert e["verdict_vs_lag"] == "helped"


# --------------------------------------------------------------------------- unit: edges / NaN / json


def test_single_group():
    """Single group."""
    y = np.zeros(5)
    raw = np.full(5, 2.0)
    comp = np.full(5, 1.0)
    g = np.zeros(5, dtype=int)
    rep = build_composite_value_report(y, raw, comp, g)
    assert rep["n_groups"] == 1
    assert rep["per_group"][0]["lift_over_raw"] == pytest.approx(0.5)


def test_empty_report():
    """Empty report."""
    empty = np.array([], dtype=float)
    rep = build_composite_value_report(empty, empty, empty, np.array([], dtype=int))
    assert rep["n_groups"] == 0
    assert rep["n_rows"] == 0
    assert rep["per_group"] == []
    assert rep["aggregate"]["net_verdict"] == "no data"
    # Renders without error.
    render_composite_value_report(rep).encode("ascii")


def test_all_nan_report():
    """All nan report."""
    y = np.array([np.nan, np.nan])
    rep = build_composite_value_report(y, np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([0, 1]))
    assert rep["n_groups"] == 0
    assert rep["aggregate"]["net_verdict"] == "no data"


def test_nan_rows_dropped_from_matched_comparison():
    # Group 0 has a NaN in composite on the 2nd row -> that row drops from ALL series (matched).
    """Nan rows dropped from matched comparison."""
    y = np.array([0.0, 0.0, 0.0, 0.0])
    raw = np.array([2.0, 100.0, 2.0, 2.0])
    comp = np.array([1.0, np.nan, 1.0, 1.0])
    g = np.array([0, 0, 1, 1])
    rep = build_composite_value_report(y, raw, comp, g)
    by = {e["group"]: e for e in rep["per_group"]}
    # Group 0 kept only 1 valid row (the NaN row dropped): raw=2 -> rmse 2, comp=1 -> rmse 1.
    assert by[0]["n"] == 1
    assert by[0]["rmse_raw"] == pytest.approx(2.0)
    assert rep["n_rows"] == 3


def test_json_roundtrip_sorted_keys():
    """Json roundtrip sorted keys."""
    rng = np.random.default_rng(0)
    n = 300
    g = rng.integers(0, 5, n)
    y = rng.normal(size=n)
    raw = y + rng.normal(0, 1.0, n)
    comp = y + rng.normal(0, 0.5, n)
    lag = y + rng.normal(0, 1.5, n)
    rep = build_composite_value_report(y, raw, comp, g, y_pred_lag=lag, expected_lift=0.3, expected_rmse=0.5)
    s = json.dumps(rep, sort_keys=True)
    assert "NaN" not in s and "Infinity" not in s  # strict-JSON clean (no bare NaN)
    assert json.loads(s) == rep


def test_markdown_is_ascii_only_even_with_unicode_group_label():
    """Markdown is ascii only even with unicode group label."""
    y = np.zeros(4)
    raw = np.full(4, 2.0)
    comp = np.full(4, 1.0)
    lag = np.full(4, 1.5)
    g = np.array(["well-éè", "well-éè", "shaft-❤", "shaft-❤"], dtype=object)
    rep = build_composite_value_report(y, raw, comp, g, y_pred_lag=lag)
    md = render_composite_value_report(rep)
    md.encode("ascii")  # raises if any non-ASCII slipped through
    assert "Composite target value report" in md
    assert "WORSE than the lag failsafe" in md


def test_surface_function_config_gate():
    """Surface function config gate."""
    y = np.zeros(4)
    raw = np.full(4, 2.0)
    comp = np.full(4, 1.0)
    g = np.array([0, 0, 1, 1])

    class _Cfg:
        """Groups tests covering cfg."""
        emit_composite_value_report = False

    assert composite_value_report(y, raw, comp, g, config=_Cfg()) is None

    class _CfgOn:
        """Groups tests covering cfg on."""
        emit_composite_value_report = True

    rep = composite_value_report(y, raw, comp, g, config=_CfgOn())
    assert rep is not None and "markdown" in rep
    rep["markdown"].encode("ascii")
    # No config -> default emit ON.
    assert composite_value_report(y, raw, comp, g) is not None


def test_bincount_fallback_matches_njit_default(monkeypatch):
    """The no-numba bincount fallback must produce the same report as the njit default (bit-close)."""
    rng = np.random.default_rng(3)
    n = 5000
    g = rng.integers(0, 40, n)
    y = rng.normal(size=n)
    raw = y + rng.normal(0, 1.0, n)
    comp = y + rng.normal(0, 0.5, n)
    lag = y + rng.normal(0, 1.3, n)
    # Sprinkle NaNs so the finite-gate path is exercised in both branches.
    comp[::37] = np.nan
    lag[::53] = np.nan
    rep_njit = build_composite_value_report(y, raw, comp, g, y_pred_lag=lag)
    monkeypatch.setattr(vr_mod, "_HAVE_NUMBA", False)
    rep_bc = build_composite_value_report(y, raw, comp, g, y_pred_lag=lag)
    assert rep_bc["n_rows"] == rep_njit["n_rows"]
    assert rep_bc["aggregate"]["vs_raw"] == rep_njit["aggregate"]["vs_raw"]
    assert rep_bc["aggregate"]["n_worse_than_lag"] == rep_njit["aggregate"]["n_worse_than_lag"]
    assert rep_bc["aggregate"]["net_weighted_lift_over_raw"] == pytest.approx(rep_njit["aggregate"]["net_weighted_lift_over_raw"], rel=1e-9)
    assert rep_bc["aggregate"]["pooled_rmse_composite"] == pytest.approx(rep_njit["aggregate"]["pooled_rmse_composite"], rel=1e-9)


def test_expected_vs_realized_calibration():
    """Expected vs realized calibration."""
    y = np.zeros(4)
    raw = np.full(4, 2.0)
    comp = np.full(4, 1.0)  # realized lift = 0.5
    g = np.array([0, 0, 1, 1])
    # Selector promised 0.9 but realized 0.5 -> optimistic (over-promised).
    rep = build_composite_value_report(y, raw, comp, g, expected_lift=0.9)
    evr = rep["expected_vs_realized"]
    assert evr["realized_lift"] == pytest.approx(0.5)
    assert evr["calibration"].startswith("optimistic")
    # Promised 0.5 -> on-target.
    rep2 = build_composite_value_report(y, raw, comp, g, expected_lift=0.5)
    assert rep2["expected_vs_realized"]["calibration"] == "on-target"


# --------------------------------------------------------------------------- biz_value


def _make_split_synthetic(seed=0):
    """8 groups the composite HELPS (comp err << raw err) + 4 it HURTS (comp err >> raw err, and >lag)."""
    rng = np.random.default_rng(seed)
    helped = list(range(8))
    hurt = list(range(8, 12))
    ys, raws, comps, lags, gids = [], [], [], [], []
    for g in helped + hurt:
        n = 1000 if g in helped else 250
        yv = rng.normal(50.0, 5.0, n)
        raw = yv + rng.normal(0.0, 1.0, n)  # raw RMSE ~ 1.0 everywhere
        if g in helped:
            comp = yv + rng.normal(0.0, 0.3, n)  # comp RMSE ~ 0.3 -> helps
        else:
            comp = yv + rng.normal(0.0, 2.0, n)  # comp RMSE ~ 2.0 -> hurts, and > lag
        lag = yv + rng.normal(0.0, 1.5, n)  # lag RMSE ~ 1.5
        ys.append(yv)
        raws.append(raw)
        comps.append(comp)
        lags.append(lag)
        gids.append(np.full(n, g))
    return (np.concatenate(ys), np.concatenate(raws), np.concatenate(comps), np.concatenate(lags), np.concatenate(gids), set(helped), set(hurt))


def test_biz_val_value_report_identifies_helped_and_hurt_split():
    """Biz val value report identifies helped and hurt split."""
    y, raw, comp, lag, g, helped, hurt = _make_split_synthetic()
    rep = build_composite_value_report(y, raw, comp, g, y_pred_lag=lag)
    by = {e["group"]: e for e in rep["per_group"]}

    # Every engineered helped group is reported "helped"; every hurt group "hurt".
    for gid in helped:
        assert by[gid]["verdict_vs_raw"] == "helped", f"group {gid} should be helped"
        assert by[gid]["lift_over_raw"] >= 0.55, f"helped lift floor (measured ~0.70): {by[gid]['lift_over_raw']}"
    for gid in hurt:
        assert by[gid]["verdict_vs_raw"] == "hurt", f"group {gid} should be hurt"
        assert by[gid]["lift_over_raw"] <= -0.7, f"hurt lift floor (measured ~-1.0): {by[gid]['lift_over_raw']}"

    vr = rep["aggregate"]["vs_raw"]
    assert vr["helped"] == len(helped)
    assert vr["hurt"] == len(hurt)


def test_biz_val_value_report_worse_than_lag_set():
    """Biz val value report worse than lag set."""
    y, raw, comp, lag, g, _helped, hurt = _make_split_synthetic()
    rep = build_composite_value_report(y, raw, comp, g, y_pred_lag=lag)
    agg = rep["aggregate"]
    # The hurt groups (comp RMSE ~2.0 > lag ~1.5) are exactly the "should not have deployed" set.
    assert set(agg["worse_than_lag_groups"]) == hurt
    assert agg["n_worse_than_lag"] == len(hurt)


def test_biz_val_value_report_net_lift_sign_positive_when_helped_rows_dominate():
    """Biz val value report net lift sign positive when helped rows dominate."""
    y, raw, comp, lag, g, _helped, _hurt = _make_split_synthetic()
    rep = build_composite_value_report(y, raw, comp, g, y_pred_lag=lag)
    net = rep["aggregate"]["net_weighted_lift_over_raw"]
    # Helped groups carry 8000 rows @ +0.70; hurt carry 1000 rows @ -1.0.
    # Net = (8000*0.70 + 1000*-1.0)/9000 ~= +0.51. Floor ~15% inside.
    assert net > 0.0, "net lift must be positive when helped rows dominate"
    assert net >= 0.43, f"net-lift floor (measured ~0.51): {net}"
    assert rep["aggregate"]["net_verdict"] == "composite helped overall"


def test_biz_val_value_report_net_lift_sign_flips_when_hurt_dominates():
    """Regression sensor: enlarge the hurt groups so hurt rows dominate -> net must go NEGATIVE.

    If a future change swapped raw/composite or mislabeled the split, the sign would not track the
    engineered dominance and this pair of sign tests would fail.
    """
    rng = np.random.default_rng(1)
    ys, raws, comps, gids = [], [], [], []
    for g in range(12):
        helped = g < 4
        n = 250 if helped else 1000  # hurt groups now dominate the row count
        yv = rng.normal(50.0, 5.0, n)
        raw = yv + rng.normal(0.0, 1.0, n)
        comp = yv + rng.normal(0.0, 0.3 if helped else 2.0, n)
        ys.append(yv)
        raws.append(raw)
        comps.append(comp)
        gids.append(np.full(n, g))
    rep = build_composite_value_report(np.concatenate(ys), np.concatenate(raws), np.concatenate(comps), np.concatenate(gids))
    net = rep["aggregate"]["net_weighted_lift_over_raw"]
    assert net < 0.0, f"net lift must be negative when hurt rows dominate: {net}"
    assert rep["aggregate"]["net_verdict"] == "composite hurt overall"
