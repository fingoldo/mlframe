"""Cluster 22 regression: integer 0/1 arrays are sets unless as_mask=True says they are masks (audit 2026-11-20, finding 22-F18)."""

import pytest

from mlframe.core.set_similarity import dice, jaccard, kuncheva


def test_int_arrays_are_sets_by_default_and_masks_on_request():
    """jaccard([1,0,1],[1,1,0]) is 1.0 as sets of values and 1/3 as masks."""
    assert jaccard([1, 0, 1], [1, 1, 0]) == 1.0
    assert jaccard([1, 0, 1], [1, 1, 0], as_mask=True) == pytest.approx(1 / 3)
    assert dice([1, 0, 1], [1, 1, 0], as_mask=True) == pytest.approx(0.5)


def test_as_mask_rejects_non_binary_integers():
    """Item ids that are not 0/1 cannot be masks."""
    with pytest.raises(ValueError, match="0/1"):
        jaccard([1, 2, 3], [1, 2, 3], as_mask=True)


def test_kuncheva_accepts_as_mask():
    """Two identical k-subsets given as int masks score 1.0."""
    assert kuncheva([1, 1, 0, 0], [1, 1, 0, 0], 4, as_mask=True) == 1.0
