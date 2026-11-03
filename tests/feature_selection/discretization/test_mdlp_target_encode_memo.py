"""The MDLP target encoding is remembered by content: the same labels as encoding afresh, also for a target shuffled in place between calls."""

from __future__ import annotations

import numpy as np

import mlframe.feature_selection.filters.supervised_binning as sb


def test_memo_returns_fresh_labels_and_tracks_content():
    """Repeat calls equal the uncached encoding; an in-place shuffle of the same array gives the shuffled labels; mutating a returned array does not reach the memo."""
    rng = np.random.default_rng(0)
    y = rng.standard_normal(5000)
    args = (64, False, 16, 8)
    first = sb._encode_mdlp_target(y, *args)
    want = sb._encode_mdlp_target_uncached(y, *args)
    assert np.array_equal(first[0], want[0]) and first[1] == want[1]
    first[0][:] = -1  # a caller changing its copy
    again = sb._encode_mdlp_target(y, *args)
    assert np.array_equal(again[0], want[0])
    rng.shuffle(y)  # same object, new content
    shuffled = sb._encode_mdlp_target(y, *args)
    assert np.array_equal(shuffled[0], sb._encode_mdlp_target_uncached(y, *args)[0])


def test_other_settings_are_other_keys():
    """A different pseudo-class count is encoded afresh, not served from the memo."""
    y = np.random.default_rng(1).standard_normal(3000)
    a = sb._encode_mdlp_target(y, 64, False, 16, 8)[0]
    b = sb._encode_mdlp_target(y, 64, False, 4, 8)[0]
    assert np.array_equal(b, sb._encode_mdlp_target_uncached(y, 64, False, 4, 8)[0]) and not np.array_equal(a, b)
