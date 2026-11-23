"""Cluster 29 regression: numbers are not prefixed by the CSV formula guard; real formulas still are (audit 2026-11-20, finding 29-F32)."""

import pytest

from mlframe.utils.csv_safe import neutralise_cell


@pytest.mark.parametrize("value", ["-3.5", "+2", "-1e-5"])
def test_plain_numbers_are_unchanged(value):
    """Numeric strings keep their value; they are not formulas (29-F32)."""
    assert neutralise_cell(value) == value


@pytest.mark.parametrize("value", ["=SUM(A1)", "-1+2", "+cmd", "@cmd", "\t=x"])
def test_formulas_are_still_prefixed(value):
    """Real formula-like strings keep the quote prefix (29-F32)."""
    assert neutralise_cell(value) == "'" + value
