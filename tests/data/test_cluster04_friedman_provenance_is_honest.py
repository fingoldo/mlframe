"""Audit 2026-11-20 finding 04-F11: the Friedman beds must not claim to be the published functions."""

from mlframe.data.datasets.scenarios._reference import friedman2_spec, friedman3_spec


def test_friedman2_provenance_says_it_is_an_approximation():
    """The provenance states the bed is not the published Friedman-2 function."""
    purpose = friedman2_spec().provenance["purpose"]
    assert "NOT the published Friedman-2" in purpose


def test_friedman3_provenance_says_it_is_an_approximation():
    """The provenance states the bed is not the published Friedman-3 function."""
    purpose = friedman3_spec().provenance["purpose"]
    assert "NOT the published Friedman-3" in purpose
