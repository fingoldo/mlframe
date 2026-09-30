"""Deterministic unit tests of the RFECV futility stop's decision function (synthetic per-fold curves, no model fits)."""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.wrappers.rfecv._futility_stop import futility_armed, futility_verdict, patience_for, winners_from_trace

P = 60
K = 5


def _trace(deltas, *, k=K, seed=0, base=0.80, fold_sd=0.03, noise=0.002, sizes=None):
    """Full-set baseline first, then one subset per delta; every subset shares the baseline's fold-difficulty effect (folds are shared in a real run)."""
    rng = np.random.default_rng(seed)
    fold_eff = rng.normal(0.0, fold_sd, k)
    sizes = sizes or [P - 4 * (i + 1) for i in range(len(deltas))]
    tr = [(P, tuple(base + fold_eff + rng.normal(0.0, noise, k)))]
    for n, d in zip(sizes, deltas):
        tr.append((n, tuple(base + d + fold_eff + rng.normal(0.0, noise, k))))
    return tr


def _verdict(tr, **kw):
    kw.setdefault("remaining", 20)
    return futility_verdict(tr, full_n=P, **kw)


def test_flat_curve_stops_once_min_iters_are_evaluated():
    tr = _trace([-0.001, 0.0005, -0.002, 0.001, -0.0005, 0.0, -0.001, 0.0005], noise=0.0008)
    assert _verdict(tr, min_iters=5).stop
    assert not _verdict(tr[:5], min_iters=5).stop  # only 4 subsets evaluated so far


def test_min_iters_is_a_hard_floor():
    tr = _trace([0.0] * 10)
    assert not _verdict(tr[:4], min_iters=5).stop
    assert _verdict(tr, min_iters=5).stop


def test_clearly_improving_curve_never_stops():
    tr = _trace([0.002, 0.01, 0.02, 0.035, 0.05, 0.06, 0.07, 0.08])
    assert not any(_verdict(tr[:t], min_iters=3).stop for t in range(2, len(tr) + 1))


def test_dip_then_rise_does_not_stop_prematurely():
    tr = _trace([-0.12, -0.09, -0.06, -0.04, -0.02, -0.008, -0.002, 0.04])
    assert not any(_verdict(tr[:t], min_iters=3).stop for t in range(2, len(tr) + 1))


def test_rising_but_still_negative_trend_is_held_by_the_trend_guard():
    tr = _trace([-0.10, -0.07, -0.05, -0.03, -0.015, -0.006], sizes=[5, 10, 20, 30, 40, 50])
    v = _verdict(tr, min_iters=5)
    assert not v.stop and "trending up" in v.why


def test_subset_that_moved_the_pick_disables_the_stop():
    tr = _trace([0.0, 0.0, 0.08, 0.0, 0.0, 0.0])
    v = _verdict(tr, min_iters=3)
    assert not v.stop and ("moved" in v.why or "could still" in v.why)


def test_single_fold_is_conservative():
    tr = _trace([0.0] * 10, k=1)
    assert not _verdict(tr, min_iters=3).stop
    assert not _verdict(tr, min_iters=3, anchor="pick").stop


def test_nan_fold_is_conservative():
    tr = _trace([0.0] * 8)
    n, scores = tr[3]
    tr[3] = (n, (float("nan"),) + scores[1:])
    assert not _verdict(tr, min_iters=3).stop


def test_pick_anchor_fires_when_the_pick_sits_below_the_full_set_and_nothing_beats_it():
    tr = _trace([0.0, -0.002, 0.0015, -0.001, 0.0, -0.0015, 0.0005, -0.001], noise=0.0008)
    assert _verdict(tr, min_iters=5, anchor="pick").stop


def test_unknown_anchor_is_rejected():
    with pytest.raises(ValueError):
        _verdict(_trace([0.0] * 8), min_iters=3, anchor="nope")


def test_trace_not_starting_at_full_set_is_rejected():
    tr = _trace([0.0] * 8)[1:]
    assert not _verdict(tr, min_iters=3).stop


def test_noisy_unpaired_folds_do_not_stop():
    """Fold-score noise as large as the band (pairing gives no tightening) leaves the upper bound above the bar."""
    tr = _trace([0.0] * 8, noise=0.03, fold_sd=0.005)
    assert not _verdict(tr, min_iters=5).stop


def test_patience_scales_with_remaining_iterations():
    assert patience_for(0, 0.1) == 2
    assert patience_for(20, 0.1) == 2
    assert patience_for(80, 0.1) == 8
    tr = _trace([0.0] * 9, noise=0.0008)
    assert _verdict(tr, min_iters=5, remaining=10, patience_frac=0.5).stop
    assert not _verdict(tr, min_iters=5, remaining=40, patience_frac=0.5).stop  # patience 20 > 9 evaluated sizes


def test_verdict_carries_the_evidence():
    v = _verdict(_trace([-0.001, 0.001, 0.0, -0.002, 0.0005, 0.0]), min_iters=5)
    assert v.stop and v.n_evaluated == 6 and v.baseline_mean == pytest.approx(0.80, abs=0.05)
    assert v.upper_bound < v.bar
    assert v.describe().startswith("futility (")


def test_winners_keep_best_revisit_per_size():
    curve, order = winners_from_trace([(10, (0.5, 0.5)), (5, (0.6, 0.6)), (10, (0.7, 0.7)), (10, (0.4, 0.4))])
    assert order == [10, 5] and curve[10].tolist() == [0.7, 0.7]


@pytest.mark.parametrize(
    "kw,armed",
    [
        ({}, True),
        ({"n_features_selection_rule": "one_se_max"}, True),
        ({"n_features_selection_rule": "one_se_max_foldstd"}, True),
        ({"n_features_selection_rule": "argmax"}, False),
        ({"n_features_selection_rule": "one_se_min"}, False),
        ({"feature_cost": 0.001}, False),
        ({"max_nfeatures": 10}, False),
        ({"futility_stop": False}, False),
    ],
)
def test_armed_only_for_one_se_max_without_cost_or_cap(kw, armed):
    class _S:
        futility_stop = True
        n_features_selection_rule = "auto"
        feature_cost = 0.0
        max_nfeatures = None
        special_feature_indices = None

    s = _S()
    for k, v in kw.items():
        setattr(s, k, v)
    assert futility_armed(s) is armed
