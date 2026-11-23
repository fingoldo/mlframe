"""Cluster 28 regression: load_and_prepare_dataframe raises a typed error with a message, not a bare AssertionError (audit 2026-11-20, finding 28-F52).

Source-level pin: the bare assertion is gone (and would be stripped under python -O), and the replacement names the function and the type it
received.
"""

from pathlib import Path

from mlframe.training import preprocessing as pp

_SRC = Path(pp.__file__).resolve()


def test_no_bare_assertion_remains_in_the_loader():
    """The bare raise AssertionError at the end of load_and_prepare_dataframe is removed (28-F52)."""
    text = _SRC.read_text(encoding="utf-8")
    assert "        raise AssertionError\n" not in text


def test_replacement_error_names_the_function_and_type():
    """The new error names load_and_prepare_dataframe and the received type (28-F52)."""
    text = _SRC.read_text(encoding="utf-8")
    assert "load_and_prepare_dataframe: expected a polars DataFrame after collection, got" in text
