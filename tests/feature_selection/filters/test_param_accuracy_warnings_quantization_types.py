"""The coarse-quantization caveat fires for numpy integers and ignores booleans, which are ints in Python but never a bin count."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.feature_selection.filters._param_accuracy_warnings import warn_accuracy_suboptimal_params


def _triggered(value) -> bool:
    """Whether ``quantization_nbins=value`` raises the accuracy warning."""
    est = SimpleNamespace(quantization_nbins=value)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_accuracy_suboptimal_params(est)
    return any("quantization_nbins" in str(w.message) for w in caught)


@pytest.mark.parametrize("value", [3, np.int64(3), np.int32(4)])
def test_integer_types_below_five_are_flagged(value):
    """A python or numpy integer under 5 is a coarse discretisation."""
    assert _triggered(value)


@pytest.mark.parametrize("value", [10, np.int64(10), True, False, None, "auto"])
def test_good_or_non_count_values_are_not_flagged(value):
    """A healthy count, a boolean or a non-numeric value stays silent."""
    assert not _triggered(value)
