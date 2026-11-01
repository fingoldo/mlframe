"""The plot-output DSL parses valid clauses the same way and rejects a long unterminated run of whitespace quickly (RDS2-2).

The clause pattern used to overlap its whitespace quantifiers with a class that also held whitespace, so a long run of spaces
inside an unterminated bracket was scanned from every starting position.
"""

import time

import pytest

from mlframe.reporting.output import parse_plot_output_dsl


def test_clause_with_spaces_between_formats_still_parses():
    """Whitespace around the brackets and between formats is accepted as before."""
    spec = parse_plot_output_dsl("plotly[html, png]")
    assert spec is not None


def test_unknown_format_is_still_rejected():
    """A bad format name still raises, so the stricter pattern did not weaken validation."""
    with pytest.raises(ValueError):
        parse_plot_output_dsl("plotly[nonsense]")


def test_long_whitespace_run_in_an_unterminated_bracket_is_rejected_quickly():
    """A pathological input fails in linear time instead of stalling the parser."""
    text = "plotly[" + " " * 200_000 + "html"
    start = time.perf_counter()
    with pytest.raises(ValueError):
        parse_plot_output_dsl(text)
    assert time.perf_counter() - start < 2.0
