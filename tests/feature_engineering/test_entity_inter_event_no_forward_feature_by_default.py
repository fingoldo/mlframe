"""The default output of entity_inter_event_features holds nothing a row could not know at its own time."""

import numpy as np

from mlframe.feature_engineering.entity_inter_event import entity_inter_event_features

_IDS = np.array([1, 1, 1, 2, 2])
_TS = np.array([0.0, 1.0, 5.0, 0.0, 2.0])


def test_the_forward_looking_gap_is_not_emitted_by_default():
    """time_to_next_event is the timestamp of an event that has not happened yet: leakage in any training frame."""
    out = entity_inter_event_features(_IDS, _TS)
    assert "time_to_next_event" not in out
    assert "time_since_prev_event" in out


def test_it_is_available_when_asked_for_explicitly():
    out = entity_inter_event_features(_IDS, _TS, include_forward_looking=True)
    np.testing.assert_allclose(out["time_to_next_event"], [1.0, 4.0, np.nan, 2.0, np.nan])


def test_the_backward_gap_is_unchanged():
    np.testing.assert_allclose(entity_inter_event_features(_IDS, _TS)["time_since_prev_event"], [np.nan, 1.0, 4.0, np.nan, 2.0])
