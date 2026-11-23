"""Cluster 27 regression: format_metric's documented examples match its output (audit 2026-11-20, finding 27-F20)."""

import pytest

from mlframe.training._format import format_metric


@pytest.mark.parametrize(
    "value, expected",
    [
        (11497.4655, "11497.47"),
        (13.93, "13.93"),
        (0.4655, "0.47"),
        (0.0034, "0.0034"),
        (1.23e-05, "0.000012"),
        (0.0, "0.00"),
    ],
)
def test_documented_examples_match_output(value, expected):
    """Each example in the docstring is what format_metric returns at the default ndigits (27-F20)."""
    assert format_metric(value) == expected


def test_nan_formats_as_nan():
    """The documented NaN case (27-F20)."""
    assert format_metric(float("nan")) == "nan"
