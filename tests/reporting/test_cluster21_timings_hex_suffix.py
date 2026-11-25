"""Cluster 21 regression: the timing grouper strips only hex disambiguators that contain a digit (audit 2026-11-20, finding 21-F21).

A chart stem ending in an ordinary hex-letter word (decade, accede) keeps its name, so it does not merge with its prefix chart. A real
hex disambiguator with a digit is still removed, as are the backend and numeric suffixes.
"""

import pytest

from mlframe.reporting.renderers import _render_timings as rt


@pytest.mark.parametrize("stem", ["pit_decade", "pit_accede", "pit_facade"])
def test_hex_words_are_not_stripped(stem):
    """A stem ending in an all-letter hex word is its own chart type (21-F21)."""
    assert rt.chart_type_of(stem) == stem


@pytest.mark.parametrize("stem,expected", [("pit_3fa9c21b", "pit"), ("pit_12", "pit"), ("pit.matplotlib", "pit")])
def test_real_disambiguators_are_still_stripped(stem, expected):
    """Hex disambiguators with a digit, numeric and backend suffixes are removed (21-F21)."""
    assert rt.chart_type_of(stem) == expected
