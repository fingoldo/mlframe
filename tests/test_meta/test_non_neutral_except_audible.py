"""Behavioural pins for the non_neutral_except_fallback fixes: substituted values are unchanged, and the unexpected-failure ones are now audible."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_subsample import _derive_seed
from mlframe.feature_selection.filters.engineered_recipes._recipe_core import _arrays_equal_nan_manual, _extra_equal
from mlframe.feature_selection.wrappers.rfecv._nan_policy import _array_has_nan
from mlframe.training._data_helpers import _finite_mask_or_none
from mlframe.utils.log_throttle import reset_throttle_counts


class _BrokenRng:
    """A generator whose integers() always fails."""

    def integers(self, *args, **kwargs):
        """Raise to simulate a broken RNG."""
        raise RuntimeError("boom")


def test_derive_seed_failure_is_audible_and_keeps_fallback(caplog):
    """A failed RNG draw still yields seed 0, but now says so at WARNING."""
    reset_throttle_counts()
    with caplog.at_level(logging.WARNING):
        assert _derive_seed(_BrokenRng()) == 0
    assert any(r.levelno >= logging.WARNING and "_derive_seed" in r.getMessage() for r in caplog.records)
    assert isinstance(_derive_seed(np.random.default_rng(1)), int)


@pytest.mark.parametrize(
    "x, expected",
    [
        (np.array([1.0, np.nan]), True),
        (np.array([1.0, 2.0]), False),
        (np.array([1, 2]), False),
        (np.array([1.0, float("nan")], dtype=object), True),
        (np.array(["a", "b"], dtype=object), False),
    ],
)
def test_array_has_nan(x, expected):
    """NaN detection per dtype family, including object arrays and non-numeric content."""
    assert _array_has_nan(x) is expected


def test_finite_mask_or_none():
    """Numeric input gets a mask; object input (unsupported by isfinite) gets None, not an exception."""
    assert _finite_mask_or_none(np.array([1.0, np.inf])).tolist() == [True, False]
    assert _finite_mask_or_none(np.array(["a"], dtype=object)) is None


def test_manual_nan_equal_matches_equal_nan_semantics():
    """The old-numpy fallback treats both-NaN as equal and rejects shape/value mismatches."""
    a = np.array([1.0, np.nan])
    assert _arrays_equal_nan_manual(a, a.copy())
    assert not _arrays_equal_nan_manual(a, np.array([1.0, 2.0]))
    assert not _arrays_equal_nan_manual(a, np.array([1.0]))
    assert _extra_equal({"k": a}, {"k": a.copy()})
    assert not _extra_equal({"k": a}, {"k": np.array([1.0, 3.0])})
