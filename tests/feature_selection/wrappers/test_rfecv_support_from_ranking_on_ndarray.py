"""Final voting and the p>=n cap pick features by a ranking of integer columns; the support must still mark them on ndarray fits (string names)."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.wrappers.rfecv._stability_select import _support_from_ranking


def test_integer_ranking_marks_string_named_columns():
    """Columns 4 and 1 ranked on a 6-column ndarray fit (names '0'..'5') give a mask with exactly those two set."""
    names = [str(i) for i in range(6)]
    mask = _support_from_ranking(names, [4, 1])
    assert mask.tolist() == [False, True, False, False, True, False]


def test_string_ranking_on_a_dataframe_fit_still_works():
    """Named columns keep working (negative control)."""
    assert _support_from_ranking(["a", "b", "c"], ["c"]).tolist() == [False, False, True]


def test_empty_ranking_gives_all_false():
    """No ranked features, no support."""
    assert not np.any(_support_from_ranking(["a", "b"], []))
