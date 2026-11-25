"""Cluster 04 regression: raising a small n_samples to the rare-class floor is logged and recorded (audit 2026-11-20, finding 04-F31)."""

import logging

from mlframe.data.datasets.scenarios._observation import RARE_MIN_ROWS, rare_class_spec


def test_small_n_samples_is_raised_with_a_warning(caplog):
    """A request below the floor is raised to RARE_MIN_ROWS and a WARNING names both values (04-F31)."""
    with caplog.at_level(logging.WARNING):
        spec = rare_class_spec(n_samples=100)
    assert spec.n_samples == RARE_MIN_ROWS
    assert any(r.levelno == logging.WARNING and "raised to the RARE_MIN_ROWS floor" in r.getMessage() for r in caplog.records)


def test_requested_size_is_recorded_in_provenance():
    """The caller's requested size survives in provenance even after the floor applies (04-F31)."""
    spec = rare_class_spec(n_samples=100)
    assert spec.provenance["requested_n_samples"] == "100"


def test_size_above_floor_is_unchanged_and_silent(caplog):
    """Control: a request at or above the floor is kept as given with no warning (04-F31)."""
    with caplog.at_level(logging.WARNING):
        spec = rare_class_spec(n_samples=RARE_MIN_ROWS + 500)
    assert spec.n_samples == RARE_MIN_ROWS + 500
    assert not any("raised to the RARE_MIN_ROWS floor" in r.getMessage() for r in caplog.records)
