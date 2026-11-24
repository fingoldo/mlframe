"""Cluster 41 regression: the MTR report logs are readable and survive a non-numeric metric (audit 2026-11-20, finding 41-F34).

The verbose warning printed literal ``{plot_file}`` / ``{k}`` placeholders, and a non-numeric metric made the ``:+.4f`` format raise, dropping the
whole metric block. The warning now uses ``<...>`` placeholders, and each metric falls back to ``str()`` when it is not numeric.
"""

from pathlib import Path

from mlframe.training.reporting._reporting_regression import _mtr


def test_warning_text_has_no_literal_format_placeholders():
    """The warning names its chart file pattern with angle placeholders, not braces (41-F34)."""
    src = Path(_mtr.__file__).resolve().read_text(encoding="utf-8")
    assert "rendered to {plot_file}" not in src
    assert "rendered to <plot_file>_target<k><ext>" in src


def test_metric_formatting_falls_back_for_non_numeric_values():
    """A non-numeric metric is formatted with str() instead of raising in the :+.4f format (41-F34)."""
    src = Path(_mtr.__file__).resolve().read_text(encoding="utf-8")
    assert "{_v:+.4f}" not in src
    assert "_fmt = str(_v)" in src
