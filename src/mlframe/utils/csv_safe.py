"""Neutralise spreadsheet formula injection in frames exported to CSV.

A cell or header that begins with ``=``, ``+``, ``-``, ``@``, a tab or a carriage return is read as a formula by Excel and LibreOffice.
Feature, target and model names come from the data, so an exported report could carry one. A leading apostrophe turns the cell into text.
Numbers are left alone: only string cells and headers are touched.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

_FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")


def _is_plain_number(value: str) -> bool:
    """True when the string is a plain number such as ``-3.5`` or ``+2``, which spreadsheets read as a value, not a formula (29-F32)."""
    try:
        float(value)
    except ValueError:
        return False
    return True


def neutralise_cell(value: Any) -> Any:
    """Return ``value`` unchanged unless it is a string that starts like a formula, in which case prefix a single quote.

    A plain number with a leading sign (``-3.5``, ``+2``) is not a formula and is left alone, so numeric-looking object-column
    values are not changed (29-F32).
    """
    if isinstance(value, str) and value.startswith(_FORMULA_STARTS) and not _is_plain_number(value):
        return "'" + value
    return value


def csv_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``frame`` whose string cells and column names cannot be read as spreadsheet formulas (the data frame passed in is not modified)."""
    out = frame.copy(deep=False)
    for column in frame.columns:
        series = frame[column]
        if series.dtype == object or pd.api.types.is_string_dtype(series.dtype):
            out[column] = series.map(neutralise_cell)
    out.columns = [neutralise_cell(c) if isinstance(c, str) else c for c in frame.columns]
    return out
