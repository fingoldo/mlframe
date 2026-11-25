"""Cluster 20 regression: the error-bias binning warning describes a skip, not a fallback (audit 2026-11-20, finding 20-F32).

Source-level pin: the warning says the feature was skipped and no longer uses the copy-pasted 'fell back to its default' wording.
"""

from pathlib import Path

from mlframe.reporting.charts import _error_bias as eb

_SRC = Path(eb.__file__).resolve()


def test_warning_says_skipped_not_fell_back():
    """The log text names a skip and drops the copy-pasted fallback wording (20-F32)."""
    text = _SRC.read_text(encoding="utf-8")
    assert "fell back to its default" not in text
    assert "skipped this feature: the histogram could not be binned" in text
