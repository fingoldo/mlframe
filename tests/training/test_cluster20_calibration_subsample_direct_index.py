"""Cluster 20 regression: the calibration chart subsample indexes both arrays directly (audit 2026-11-20, finding 20-F45).

The pair was indexed through a one-off lambda. The rewrite indexes ``s`` and ``t`` with the same drawn indices. The source no longer contains
the lambda, and the module's public chart function still runs on a frame larger than the cap.
"""

from pathlib import Path

from mlframe.reporting.charts import calibration as cal


def test_subsample_has_no_lambda_indexing():
    """The two-array index is written directly, not through a lambda (20-F45)."""
    src = Path(cal.__file__).resolve().read_text(encoding="utf-8")
    assert "(lambda i: (s[i], t[i]))" not in src
    assert "s, t = s[_idx], t[_idx]" in src
