"""Exported CSV reports cannot carry a spreadsheet formula from a feature, target or model name (INJ4-1, INJ4-2).

A cell or header that starts with ``=``, ``+``, ``-`` or ``@`` is evaluated by Excel and LibreOffice. The export sites now pass the frame through
``csv_safe_frame``, which prefixes such strings with an apostrophe and leaves numbers alone.
"""

import pandas as pd

from mlframe.training.core._phase_targets_performance import _persist
from mlframe.utils.csv_safe import csv_safe_frame, neutralise_cell


def test_formula_like_strings_get_an_apostrophe_and_numbers_are_untouched():
    """Strings that start like a formula are prefixed, a negative number is not."""
    assert neutralise_cell("=cmd|' /C calc'!A0") == "'=cmd|' /C calc'!A0"
    assert neutralise_cell("@SUM(A1)") == "'@SUM(A1)"
    assert neutralise_cell("-2+3") == "'-2+3"
    assert neutralise_cell("plain") == "plain"
    assert neutralise_cell(-1.5) == -1.5


def test_frame_cells_and_headers_are_neutralised_without_changing_the_input():
    """Names in cells and in the header are neutralised; the frame passed in keeps its original values."""
    frame = pd.DataFrame({"=header": ["=1+1", "ok"], "score": [-0.5, 0.25]})
    safe = csv_safe_frame(frame)
    assert list(safe.columns) == ["'=header", "score"]
    assert safe["'=header"].tolist() == ["'=1+1", "ok"]
    assert safe["score"].tolist() == [-0.5, 0.25]
    assert frame["=header"].tolist() == ["=1+1", "ok"]


def test_targets_performance_export_writes_no_live_formula(tmp_path):
    """The real export path writes the neutralised text: no line of the CSV starts with a formula character in a name cell."""
    frame = pd.DataFrame({"target": ['=HYPERLINK("http://x")', "y"], "auc": [0.7, 0.8]})
    metadata: dict = {}
    _persist(frame, str(tmp_path / "run.png"), metadata)
    path = metadata["targets_performance_path"]
    text = open(path, encoding="utf-8").read()
    assert "'=HYPERLINK" in text
    assert "\n=HYPERLINK" not in text
