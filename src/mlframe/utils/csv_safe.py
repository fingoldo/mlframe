"""Neutralise spreadsheet formula injection in frames exported to CSV.

A cell or header that begins with ``=``, ``+``, ``-``, ``@``, a tab or a carriage return is read as a formula by Excel and LibreOffice.
Feature, target and model names come from the data, so an exported report could carry one. A leading apostrophe turns the cell into text.
Numbers are left alone: only string cells and headers are touched.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

_FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")


def neutralise_cell(value: Any) -> Any:
    """Return ``value`` unchanged unless it is a string that starts like a formula, in which case prefix a single quote."""
    if isinstance(value, str) and value.startswith(_FORMULA_STARTS):
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
