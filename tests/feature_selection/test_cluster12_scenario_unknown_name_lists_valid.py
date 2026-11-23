"""Cluster 12 regression: an unknown scenario name lists the valid names (audit 2026-11-20, finding 12-F101).

The lookup used to raise a bare KeyError with no choices. The error now names the request and lists every valid name.
"""

import pytest

from mlframe.feature_selection._benchmarks.fs_hybrid import scenarios


def test_unknown_name_error_lists_valid_names():
    """The KeyError names the bad scenario and includes a real valid name (12-F101)."""
    with pytest.raises(KeyError) as excinfo:
        scenarios.make("no_such_scenario_xyz")
    message = str(excinfo.value)
    assert "no_such_scenario_xyz" in message
    assert "valid names" in message
    valid = next(iter(scenarios.ALL_SCENARIOS))
    assert valid in message


def test_known_name_is_still_built():
    """Control: a valid name still builds a bed (12-F101)."""
    name = next(iter(scenarios.ALL_SCENARIOS))
    bed = scenarios.make(name, seed=0, n=50)
    assert bed is not None
