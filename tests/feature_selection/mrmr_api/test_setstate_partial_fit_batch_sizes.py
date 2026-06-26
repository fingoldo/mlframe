"""Regression (B3): __setstate__ must inject _partial_fit_batch_sizes_ for an attribute-less
legacy pickle, like its sibling partial-fit buffers. Omitting it made a pickled-then-resumed
partial_fit fall back to a single fictional batch, collapsing multi-batch history (zeroing decay).
"""
from __future__ import annotations

from mlframe.feature_selection.filters.mrmr import MRMR


def test_setstate_injects_partial_fit_batch_sizes():
    m = MRMR.__new__(MRMR)
    m.__setstate__({})
    assert hasattr(m, "_partial_fit_batch_sizes_")
    assert m._partial_fit_batch_sizes_ == []


def test_partial_fit_batch_sizes_not_aliased_across_instances():
    a = MRMR.__new__(MRMR); a.__setstate__({})
    b = MRMR.__new__(MRMR); b.__setstate__({})
    a._partial_fit_batch_sizes_.append(1)
    assert b._partial_fit_batch_sizes_ == [], "mutable default aliased across unpickled instances"
