"""RFECV on ndarray input stores string fit-time names, yet pins and group members are integer columns: both must still resolve."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlframe.feature_selection.wrappers.rfecv._finalize import _expand_support_by_feature_groups, _glue_must_include_into_support


def _fitted(mask):
    """Stand-in for an RFECV fitted on a 6-column ndarray."""
    return SimpleNamespace(feature_names_in_=[str(i) for i in range(6)], support_=np.asarray(mask, dtype=bool), n_features_=int(np.sum(mask)), feature_groups=None)


def test_integer_must_include_is_added_to_the_support_of_an_ndarray_fit():
    """Column 3 is pinned although the optimiser kept only columns 0 and 5."""
    est = _fitted([True, False, False, False, False, True])
    _glue_must_include_into_support(est, [3])
    assert est.support_.tolist() == [True, False, False, True, False, True]
    assert est.n_features_ == 3


def test_integer_group_members_are_expanded_on_an_ndarray_fit():
    """A group of columns 1 and 2 is kept whole once column 1 is selected."""
    est = _fitted([False, True, False, False, False, False])
    est.feature_groups = {"g": [1, 2]}
    _expand_support_by_feature_groups(est, verbose=0)
    assert np.asarray(est.support_, dtype=bool)[[1, 2]].all()
