"""Unit + biz_value tests for the MoE selection gate (``_moe_gate.MoESelectionGate``).

The gate takes validation-split expert predictions (composite / raw-y / lag failsafe) + true y (+ optional
groups), LEARNS a per-group (fallback: global) choice, and routes at predict time with a hard guarantee: on
the selection split the deployed prediction is never worse, in expectation, than the lag failsafe where lag
is available -- and, at ``shrink_rtol == 0``, never worse than ANY single expert. The unit tests pin the
argmin selection, the tie / shrink / fallback rules, and the degenerate cases; the biz_value test proves the
gate's HELD-OUT (disjoint-split) RMSE beats every single expert and always-composite on a synthetic where
composite wins on some groups and lag on others, and is never worse than lag on any lag-defined group.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite._moe_gate import MoESelectionGate


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# --------------------------------------------------------------------------- unit: per-group argmin


def test_per_group_argmin_selection_correct():
    # Group A: composite exact, raw/lag off -> pick composite. Group B: lag exact -> pick lag.
    y = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
    comp = np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0])  # exact on A, off on B
    raw = np.array([2.0, 2.0, 2.0, 12.0, 12.0, 12.0])  # off everywhere
    lag = np.array([3.0, 3.0, 3.0, 10.0, 10.0, 10.0])  # exact on B
    g = np.array(["A", "A", "A", "B", "B", "B"])
    gate = MoESelectionGate().fit(y, {"composite": comp, "raw": raw, "lag": lag}, group_ids=g)
    assert gate.group_choice_["A"] == "composite"
    assert gate.group_choice_["B"] == "lag"


def test_gate_never_selects_worse_than_lag_on_selection_split():
    rng = np.random.default_rng(1)
    n = 300
    y = rng.normal(size=n)
    comp = y + rng.normal(0, 0.2, n)
    raw = y + rng.normal(0, 0.5, n)
    lag = y + rng.normal(0, 0.3, n)
    g = rng.integers(0, 6, n)
    gate = MoESelectionGate().fit(y, {"composite": comp, "raw": raw, "lag": lag}, group_ids=g)
    assert gate.guarantee_["not_worse_than_lag"] is True
    assert gate.guarantee_["not_worse_than_best_single"] is True
    gate_rmse = gate.guarantee_["pooled_rmse_gate"]
    for name, r in gate.guarantee_["pooled_rmse_per_expert"].items():
        assert gate_rmse <= r * (1.0 + 1e-9), f"gate {gate_rmse} > expert {name} {r}"


def test_gate_pooled_rmse_exact_argmin_at_shrink_zero():
    # With shrink_rtol=0 the gate's pooled selection-split RMSE must equal the per-group argmin exactly.
    y = np.array([0.0, 0.0, 5.0, 5.0])
    comp = np.array([0.1, -0.1, 4.0, 6.0])
    raw = np.array([1.0, -1.0, 5.05, 4.95])
    lag = np.array([0.5, -0.5, 5.0, 5.0])
    g = np.array([0, 0, 1, 1])
    gate = MoESelectionGate(shrink_rtol=0.0).fit(y, {"composite": comp, "raw": raw, "lag": lag}, group_ids=g)
    routed = gate.predict({"composite": comp, "raw": raw, "lag": lag}, group_ids=g)
    assert _rmse(routed, y) == pytest.approx(gate.guarantee_["pooled_rmse_gate"])


def test_global_fallback_for_unseen_group_is_lag():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    comp = np.array([0.0, 0.0, 0.5, 0.5])
    raw = np.array([2.0, 2.0, 2.0, 2.0])
    lag = np.array([1.0, 1.0, 1.0, 1.0])
    g = np.array(["A", "A", "B", "B"])
    gate = MoESelectionGate().fit(y, {"composite": comp, "raw": raw, "lag": lag}, group_ids=g)
    assert gate.global_choice_ == "lag"
    # Unseen group Z -> global fallback (lag).
    out = gate.predict({"composite": np.array([9.0]), "raw": np.array([8.0]), "lag": np.array([7.0])}, group_ids=np.array(["Z"]))
    assert out[0] == 7.0


def test_degenerate_single_expert_no_groups():
    y = np.array([1.0, 2.0, 3.0])
    comp = np.array([1.1, 2.1, 2.9])
    gate = MoESelectionGate().fit(y, {"composite": comp})
    assert gate.global_choice_ == "composite"
    out = gate.predict({"composite": comp})
    assert np.allclose(out, comp)


def test_degenerate_no_groups_single_global_choice():
    y = np.array([0.0, 0.0, 0.0, 0.0])
    comp = np.array([0.1, -0.1, 0.1, -0.1])
    raw = np.array([1.0, 1.0, 1.0, 1.0])
    lag = np.array([2.0, 2.0, 2.0, 2.0])
    gate = MoESelectionGate().fit(y, {"composite": comp, "raw": raw, "lag": lag})
    # Single group -> composite is best -> beats lag -> deployed everywhere (groupless fallback = that choice).
    assert gate.group_choice_[None] == "composite"
    assert gate.global_choice_ == "composite"
    out = gate.predict({"composite": comp, "raw": raw, "lag": lag})
    assert np.allclose(out, comp)


def test_tie_prefers_lag():
    # composite and lag are BYTE-identical -> tie must resolve to the conservative lag failsafe.
    y = np.array([0.0, 0.0, 0.0])
    comp = np.array([1.0, -1.0, 1.0])
    raw = np.array([5.0, 5.0, 5.0])
    lag = np.array([1.0, -1.0, 1.0])
    gate = MoESelectionGate().fit(y, {"composite": comp, "raw": raw, "lag": lag}, group_ids=np.zeros(3))
    assert gate.group_choice_[0.0] == "lag"


def test_shrink_rtol_keeps_lag_on_marginal_win_but_stays_not_worse_than_lag():
    # composite beats lag by ~4% only; shrink_rtol=0.10 keeps lag; vs-lag guarantee stays exact.
    rng = np.random.default_rng(3)
    n = 400
    y = rng.normal(size=n)
    lag = y + rng.normal(0, 0.30, n)
    comp = y + rng.normal(0, 0.29, n)  # marginally better than lag
    raw = y + rng.normal(0, 0.8, n)
    g = np.zeros(n)
    gate = MoESelectionGate(shrink_rtol=0.10).fit(y, {"composite": comp, "raw": raw, "lag": lag}, group_ids=g)
    assert gate.group_choice_[0.0] == "lag"
    assert gate.guarantee_["not_worse_than_lag"] is True


def test_nan_lag_rows_routed_by_priority_fallback():
    # Group chose lag, but the first row's lag is NaN -> per-row fallback to next available (raw here).
    y = np.array([0.0, 0.0, 0.0])
    comp = np.array([5.0, 5.0, 5.0])
    raw = np.array([2.0, 2.0, 2.0])
    lag = np.array([np.nan, 0.1, -0.1])  # NaN first row; near-exact elsewhere
    g = np.zeros(3)
    gate = MoESelectionGate(prefer=("lag", "raw", "composite")).fit(y, {"composite": comp, "raw": raw, "lag": lag}, group_ids=g)
    assert gate.group_choice_[0.0] == "lag"
    out = gate.predict({"composite": comp, "raw": raw, "lag": lag}, group_ids=g)
    assert out[0] == 2.0  # raw, not NaN
    assert out[1] == 0.1 and out[2] == -0.1


def test_all_nan_lag_group_uses_nonlag_expert():
    # Whole group has lag == NaN -> tier-2 decides among raw/composite.
    y = np.array([0.0, 0.0, 0.0, 0.0])
    comp = np.array([0.1, -0.1, 0.1, -0.1])  # best
    raw = np.array([1.0, 1.0, 1.0, 1.0])
    lag = np.array([np.nan, np.nan, np.nan, np.nan])
    gate = MoESelectionGate().fit(y, {"composite": comp, "raw": raw, "lag": lag}, group_ids=np.zeros(4))
    assert gate.group_choice_[0.0] == "composite"


def test_no_lag_expert_is_plain_argmin_selector():
    y = np.array([0.0, 0.0, 5.0, 5.0])
    comp = np.array([0.1, -0.1, 4.0, 6.0])
    raw = np.array([1.0, -1.0, 5.0, 5.0])
    g = np.array([0, 0, 1, 1])
    gate = MoESelectionGate(failsafe="lag").fit(y, {"composite": comp, "raw": raw}, group_ids=g)
    assert gate.group_choice_[0] == "composite"
    assert gate.group_choice_[1] == "raw"


def test_route_labels_reports_per_row_choice():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    comp = np.array([0.0, 0.0, 5.0, 5.0])
    raw = np.array([2.0, 2.0, 2.0, 2.0])
    lag = np.array([3.0, 3.0, 1.0, 1.0])
    g = np.array(["A", "A", "B", "B"])
    gate = MoESelectionGate().fit(y, {"composite": comp, "raw": raw, "lag": lag}, group_ids=g)
    labels = gate.route_labels(g)
    assert list(labels) == ["composite", "composite", "lag", "lag"]


def test_sample_weight_shifts_group_choice():
    # Regression sensor: weights must enter the per-group SSE. Unweighted, composite wins (SSE 1 vs lag 8);
    # putting a heavy weight on the one row where lag is exact and composite is off flips the group to lag.
    y = np.array([0.0, 0.0, 0.0])
    comp = np.array([1.0, 0.0, 0.0])  # off by 1 on the heavy row, exact on the light rows
    lag = np.array([0.0, 2.0, 2.0])  # exact on the heavy row, off by 2 on the light rows
    raw = np.array([5.0, 5.0, 5.0])
    g = np.zeros(3)
    w = np.array([100.0, 1.0, 1.0])
    gate = MoESelectionGate().fit(y, {"composite": comp, "raw": raw, "lag": lag}, group_ids=g, sample_weight=w)
    assert gate.group_choice_[0.0] == "lag"
    gate_uw = MoESelectionGate().fit(y, {"composite": comp, "raw": raw, "lag": lag}, group_ids=g)
    assert gate_uw.group_choice_[0.0] == "composite"


def test_empty_input_guarded():
    gate = MoESelectionGate().fit(np.array([]), {"composite": np.array([]), "lag": np.array([])})
    assert gate.guarantee_["pooled_rmse_gate"] is None


# --------------------------------------------------------------------------- biz_value


def _make_split_synthetic(seed=7, n_per_group=400):
    """5 composite-favorable groups + 5 lag-favorable groups; return selection + disjoint holdout splits."""
    rng = np.random.default_rng(seed)
    n_groups = 10
    y, comp, raw, lag, g = [], [], [], [], []
    for gid in range(n_groups):
        yi = np.cumsum(rng.normal(0, 1.0, n_per_group))  # AR-ish level series
        comp_good = gid < 5
        comp_i = yi + rng.normal(0, 0.10 if comp_good else 1.0, n_per_group)
        lag_i = yi + rng.normal(0, 1.0 if comp_good else 0.10, n_per_group)
        raw_i = yi + rng.normal(0, 0.5, n_per_group)
        y.append(yi)
        comp.append(comp_i)
        raw.append(raw_i)
        lag.append(lag_i)
        g.append(np.full(n_per_group, gid))
    y = np.concatenate(y)
    comp = np.concatenate(comp)
    raw = np.concatenate(raw)
    lag = np.concatenate(lag)
    g = np.concatenate(g)
    idx = rng.permutation(y.shape[0])
    half = idx.shape[0] // 2
    sel, hold = idx[:half], idx[half:]

    def pack(ix):
        return dict(y=y[ix], comp=comp[ix], raw=raw[ix], lag=lag[ix], g=g[ix])

    return pack(sel), pack(hold)


def test_biz_val_moe_gate_beats_every_expert_and_composite_held_out():
    sel, hold = _make_split_synthetic()
    gate = MoESelectionGate(shrink_rtol=0.0).fit(sel["y"], {"composite": sel["comp"], "raw": sel["raw"], "lag": sel["lag"]}, group_ids=sel["g"])

    routed = gate.predict({"composite": hold["comp"], "raw": hold["raw"], "lag": hold["lag"]}, group_ids=hold["g"])
    gate_rmse = _rmse(routed, hold["y"])
    rmse_comp = _rmse(hold["comp"], hold["y"])
    rmse_raw = _rmse(hold["raw"], hold["y"])
    rmse_lag = _rmse(hold["lag"], hold["y"])
    min_single = min(rmse_comp, rmse_raw, rmse_lag)

    # Held-out: gate is no worse than the best single expert (tiny numeric slack) ...
    assert gate_rmse <= min_single * 1.02, f"gate {gate_rmse} vs min single {min_single}"
    # ... and STRICTLY, quantitatively better than always-composite (measured ~7x; floor 25% better).
    assert gate_rmse < rmse_comp * 0.75, f"gate {gate_rmse} not < 0.75*composite {rmse_comp}"

    # Never worse than lag on ANY group where lag is defined (held-out, per group).
    for gid in np.unique(hold["g"]):
        m = hold["g"] == gid
        gr = _rmse(routed[m], hold["y"][m])
        lr = _rmse(hold["lag"][m], hold["y"][m])
        assert gr <= lr * 1.05, f"group {gid}: gate {gr} worse than lag {lr}"


def test_biz_val_moe_gate_selection_split_guarantee_is_exact():
    sel, _ = _make_split_synthetic(seed=11)
    gate = MoESelectionGate(shrink_rtol=0.0).fit(sel["y"], {"composite": sel["comp"], "raw": sel["raw"], "lag": sel["lag"]}, group_ids=sel["g"])
    g = gate.guarantee_
    assert g["not_worse_than_lag"] and g["not_worse_than_best_single"]
    assert g["pooled_rmse_gate"] <= min(g["pooled_rmse_per_expert"].values()) * (1.0 + 1e-9)
    # The gate correctly learned the split: composite for groups 0-4, lag for groups 5-9.
    for gid in range(5):
        assert gate.group_choice_[gid] == "composite"
    for gid in range(5, 10):
        assert gate.group_choice_[gid] == "lag"
