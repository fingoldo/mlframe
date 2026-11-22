"""Audit 2026-11-20 finding 28-F41: a duplicate grid label is refused before any variant trains.

The check ran inside the loop, so a duplicate at the fifth entry raised after four variants had already run. The test uses a stub suite
function and checks it is never called when the labels collide.
"""

import pytest

from mlframe.training.grid import run_grid


def test_duplicate_label_refused_before_any_run():
    """A grid whose fourth entry repeats an earlier label raises, and the stub suite is called zero times."""
    calls = []

    def stub_suite(**kwargs):
        """Record a run; the duplicate check must prevent every one of these."""
        calls.append(kwargs)
        return {}

    grid = [{"a": 1}, {"a": 2}, {"a": 3}, ("variant_1", {"a": 4})]
    with pytest.raises(ValueError, match="duplicate grid label"):
        run_grid({}, grid, suite_fn=stub_suite)
    assert calls == []
