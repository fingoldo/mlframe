"""``active_budget``: the single reading of budget parameters, where 0 and None both mean no limit."""

from __future__ import annotations

import pytest

from mlframe.utils.budgets import active_budget


@pytest.mark.parametrize("value", [None, 0, 0.0, -1, -0.5])
def test_zero_none_and_negative_mean_no_limit(value):
    """None, 0 and negatives all read as no limit."""
    assert active_budget(value) is None


@pytest.mark.parametrize("value", [1, 30, 0.25, 1e-9])
def test_positive_values_are_returned_unchanged(value):
    """A positive limit comes back as given, type included, so ``min(limit, n)`` keeps int caps int."""
    out = active_budget(value)
    assert out == value and type(out) is type(value)
