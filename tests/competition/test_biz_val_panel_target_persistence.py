"""biz_value test for ``mlframe.competition.panel_target_persistence``.

COMPETITION/EXPLORATORY USE ONLY -- see ``mlframe.competition`` package docstring.

The win: given a candidate grouping key + within-group ordering key,
``check_target_persistence`` should clearly distinguish a genuinely persistent
panel (target rarely flips within a group's ordered sequence) from a control panel
where the same grouping key carries no real panel structure (target within a group
is effectively random). When persistence is high, the (leak-prone) lag/lead-target
features should carry strong signal about the current row's target -- demonstrating
why the trick was informative on Kaggle, even though it must never be shipped as a
production feature.
"""

from __future__ import annotations

import numpy as np

from mlframe.competition.panel_target_persistence import (
    TargetPersistenceResult,
    check_target_persistence,
    lag_target_within_group,
    lead_target_within_group,
)


def _make_persistent_panel(rng: np.random.Generator, n_groups: int, group_size: int, flip_prob: float = 0.02):
    """Each group has one latent binary state that rarely flips within the group."""
    group_ids = np.repeat(np.arange(n_groups), group_size)
    order = np.tile(np.arange(group_size), n_groups)
    y = np.empty(group_ids.size, dtype=float)
    for g in range(n_groups):
        state = float(rng.integers(0, 2))
        seq = np.empty(group_size, dtype=float)
        for i in range(group_size):
            if i > 0 and rng.random() < flip_prob:
                state = 1.0 - state
            seq[i] = state
        seq_perm = rng.permutation(group_size)  # shuffle row storage order; `order` still recovers true sequence
        y[g * group_size : (g + 1) * group_size] = seq[np.argsort(seq_perm)]
        order[g * group_size : (g + 1) * group_size] = seq_perm
    return group_ids, order, y


def _make_control_panel(rng: np.random.Generator, n_groups: int, group_size: int):
    """Same grouping cardinality/shape, but target is i.i.d. noise -- no real panel structure."""
    group_ids = np.repeat(np.arange(n_groups), group_size)
    order = np.tile(np.arange(group_size), n_groups)
    y = rng.integers(0, 2, size=group_ids.size).astype(float)
    return group_ids, order, y


def test_biz_val_check_target_persistence_distinguishes_persistent_from_control():
    rng = np.random.default_rng(0)
    n_groups, group_size = 300, 10

    g_p, o_p, y_p = _make_persistent_panel(rng, n_groups, group_size, flip_prob=0.02)
    g_c, o_c, y_c = _make_control_panel(rng, n_groups, group_size)

    result_persistent = check_target_persistence(g_p, o_p, y_p)
    result_control = check_target_persistence(g_c, o_c, y_c)

    assert isinstance(result_persistent, TargetPersistenceResult)
    assert isinstance(result_control, TargetPersistenceResult)

    # persistent panel: very low flip rate, very high lag-1 autocorrelation.
    assert result_persistent.flip_rate < 0.1, f"persistent-panel flip rate should be low, got {result_persistent.flip_rate:.4f}"
    assert result_persistent.lag1_autocorrelation > 0.7, f"persistent-panel lag-1 autocorr should be high, got {result_persistent.lag1_autocorrelation:.4f}"
    assert result_persistent.is_persistent is True

    # control panel: flip rate near 0.5 (random binary), near-zero autocorrelation.
    assert result_control.flip_rate > 0.35, f"control-panel flip rate should be near-random, got {result_control.flip_rate:.4f}"
    assert abs(result_control.lag1_autocorrelation) < 0.15, f"control-panel lag-1 autocorr should be near zero, got {result_control.lag1_autocorrelation:.4f}"
    assert result_control.is_persistent is False

    # the diagnostic must separate the two scenarios by a wide, real margin -- not just cross an
    # arbitrary threshold by noise.
    assert result_persistent.lag1_autocorrelation - result_control.lag1_autocorrelation > 0.6
    assert result_control.flip_rate - result_persistent.flip_rate > 0.25


def test_biz_val_lag_lead_target_within_group_carry_strong_signal_when_persistent():
    rng = np.random.default_rng(1)
    n_groups, group_size = 300, 10
    g_p, o_p, y_p = _make_persistent_panel(rng, n_groups, group_size, flip_prob=0.02)

    lag = lag_target_within_group(g_p, o_p, y_p)
    lead = lead_target_within_group(g_p, o_p, y_p)

    valid_lag = ~np.isnan(lag)
    valid_lead = ~np.isnan(lead)
    assert valid_lag.sum() > 0 and valid_lead.sum() > 0

    lag_agreement = float(np.mean(lag[valid_lag] == y_p[valid_lag]))
    lead_agreement = float(np.mean(lead[valid_lead] == y_p[valid_lead]))
    assert lag_agreement > 0.9, f"lag(target) should match current target most of the time in a persistent panel, got {lag_agreement:.4f}"
    assert lead_agreement > 0.9, f"lead(target) should match current target most of the time in a persistent panel, got {lead_agreement:.4f}"

    # a naive "predict target = lag(target)" rule should trounce the unconditional base rate --
    # this is the (leak-driven) signal that made the trick informative on Kaggle.
    base_rate = float(np.mean(y_p))
    majority_baseline_acc = max(base_rate, 1 - base_rate)
    assert lag_agreement > majority_baseline_acc + 0.05, (
        f"lag-based accuracy ({lag_agreement:.4f}) should beat the majority-class baseline ({majority_baseline_acc:.4f}) by a real margin"
    )

    # control: on a non-persistent panel, lag(target) carries no such signal.
    g_c, o_c, y_c = _make_control_panel(rng, n_groups, group_size)
    lag_c = lag_target_within_group(g_c, o_c, y_c)
    valid_lag_c = ~np.isnan(lag_c)
    lag_agreement_c = float(np.mean(lag_c[valid_lag_c] == y_c[valid_lag_c]))
    assert abs(lag_agreement_c - 0.5) < 0.1, f"control-panel lag agreement should be near chance, got {lag_agreement_c:.4f}"


def test_check_target_persistence_no_pairs_returns_nan_and_not_persistent():
    group_ids = np.array([1, 2, 3])  # every group has size 1: no within-group pairs
    order = np.array([0, 0, 0])
    y = np.array([1.0, 0.0, 1.0])
    result = check_target_persistence(group_ids, order, y)
    assert result.n_pairs == 0
    assert result.flip_rate != result.flip_rate  # NaN
    assert result.lag1_autocorrelation != result.lag1_autocorrelation  # NaN
    assert result.is_persistent is False


def test_lag_lead_target_within_group_shapes_and_edges():
    group_ids = np.array([1, 1, 1, 2, 2])
    order = np.array([2, 0, 1, 5, 4])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    lag = lag_target_within_group(group_ids, order, y)
    lead = lead_target_within_group(group_ids, order, y)
    assert lag.shape == y.shape
    assert lead.shape == y.shape

    # group 1 sorted by order: idx with order 0,1,2 -> y values 20,30,10 (rows 1,2,0)
    # so row0 (order=2, 3rd in seq) should have lag = y of order=1 row (30.0)
    assert lag[0] == 30.0
    assert np.isnan(lag[1])  # first in its group's sequence
    assert lead[1] == 30.0  # next in sequence after row1 (order=1) is row2... wait check via order

    # group 2 sorted by order: order 4,5 -> rows (idx4, idx3); row3 (order=5) is 2nd in seq
    assert lag[3] == 50.0
    assert np.isnan(lead[3])


def test_lag_target_within_group_periods_zero_raises():
    group_ids = np.array([1, 1])
    order = np.array([0, 1])
    y = np.array([1.0, 2.0])
    raised = False
    try:
        lag_target_within_group(group_ids, order, y, periods=0)
    except ValueError:
        raised = True
    assert raised
