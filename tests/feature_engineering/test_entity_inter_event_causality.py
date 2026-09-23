"""Entity group statistics must not be determined by events that have not happened yet.

`group_mean/std/median_time_delta` (and the `_value` triple) were computed over the entity's WHOLE segment while the
docstring called the default "whole-history-to-date": with gaps [1, 1, 1, 100], row 0 received
`group_mean_time_delta = 25.75`, a number set entirely by a gap three events LATER. Backtests scored optimistically,
the serve-time value (history only) differed, and the value also changed when the batch was split differently.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_engineering.entity_inter_event import entity_inter_event_features


def _bed():
    """One entity with four one-second gaps and a final 100-second gap, so a leak from the future is unmistakable."""
    ids = np.array(["u", "u", "u", "u", "u"])
    ts = np.array([0.0, 1.0, 2.0, 3.0, 103.0])
    return ids, ts


def test_an_early_row_cannot_see_a_late_gap():
    """Rows before the 100-second gap must average 1.0; seeing 25.75 would mean the statistic read the future."""
    ids, ts = _bed()
    out = entity_inter_event_features(ids, ts)
    mean = out["group_mean_time_delta"]
    assert np.isnan(mean[0]), "the first row of an entity has no gap yet"
    np.testing.assert_allclose(mean[1:4], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(mean[4], (1 + 1 + 1 + 100) / 4)


def test_the_value_triple_is_causal_too():
    """The value statistics take the same causal path as the time ones, not just the gaps."""
    ids, ts = _bed()
    values = np.array([10.0, 10.0, 10.0, 10.0, 1000.0])
    out = entity_inter_event_features(ids, ts, value_col=values)
    np.testing.assert_allclose(out["group_mean_value"][:4], [10.0, 10.0, 10.0, 10.0])
    np.testing.assert_allclose(out["group_median_value"][:4], [10.0, 10.0, 10.0, 10.0])
    np.testing.assert_allclose(out["group_mean_value"][4], (10 * 4 + 1000) / 5)


def test_splitting_the_batch_does_not_change_a_row_s_features():
    """The serving-time property the leak broke: scoring a prefix must give that prefix's rows the same numbers."""
    ids, ts = _bed()
    full = entity_inter_event_features(ids, ts)
    prefix = entity_inter_event_features(ids[:4], ts[:4])
    for key in ("group_mean_time_delta", "group_std_time_delta", "group_median_time_delta"):
        np.testing.assert_allclose(prefix[key], full[key][:4], err_msg=key)


def test_entities_do_not_bleed_into_each_other():
    """Interleaved entities each see only their own history, so the running window is keyed per entity."""
    ids = np.array(["a", "b", "a", "b"])
    ts = np.array([0.0, 0.0, 5.0, 50.0])
    out = entity_inter_event_features(ids, ts)
    np.testing.assert_allclose(out["group_mean_time_delta"][2], 5.0)
    np.testing.assert_allclose(out["group_mean_time_delta"][3], 50.0)


def test_the_whole_segment_behaviour_is_still_reachable():
    """``causal=False`` restores the old whole-segment statistic for callers doing offline analysis rather than modelling."""
    ids, ts = _bed()
    out = entity_inter_event_features(ids, ts, causal=False)
    np.testing.assert_allclose(out["group_mean_time_delta"], np.full(5, 25.75))


def test_a_single_event_entity_has_no_gap_statistics():
    """One event means no gap to describe: NaN, not a zero that reads as a measured interval."""
    out = entity_inter_event_features(np.array(["solo"]), np.array([7.0]))
    assert np.isnan(out["group_mean_time_delta"][0])
    assert np.isnan(out["group_median_time_delta"][0])


def test_the_causal_median_tracks_the_running_middle():
    """The median is recomputed over the gaps seen so far, which a running-window implementation can easily get wrong."""
    ids = np.array(["u"] * 6)
    ts = np.cumsum([0.0, 1.0, 5.0, 2.0, 100.0, 3.0])
    out = entity_inter_event_features(ids, ts)
    gaps = [1.0, 5.0, 2.0, 100.0, 3.0]
    expected = [np.nan] + [float(np.median(gaps[: i + 1])) for i in range(len(gaps))]
    np.testing.assert_allclose(out["group_median_time_delta"], expected)
