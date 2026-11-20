"""Audit 2026-11-20 finding 04-F13: every graded-cardinality column carries the same signal range, so each weight is 1/(levels-1)."""

import pytest

from mlframe.data.datasets.scenarios import graded_cardinality_spec


def test_each_informative_weight_scales_to_one_over_levels_minus_one():
    """coefficient x (levels - 1) == 1 for every k{levels} column, so a 100-level code and a 2-level code span the same score range."""
    spec = graded_cardinality_spec()
    coefficients = spec.targets[0].link.coefficients
    informative = {name: coef for name, coef in coefficients.items() if name.startswith("k")}
    assert informative, "no informative cardinality columns found"
    for name, coef in informative.items():
        levels = int(name[1:])
        assert coef * (levels - 1) == pytest.approx(1.0), f"{name}: weight {coef} does not scale to 1/(levels-1)"
