"""Cluster 20 regression: the binary composer docstring states the real default panel list (audit 2026-11-20, finding 20-F28).

The constant is "ROC PR SCORE_DIST THRESHOLD"; the docstring used to claim KS and GAIN are defaults too. Pinned: the constant is as expected
and the module source no longer carries the wrong default claim.
"""

from pathlib import Path

from mlframe.reporting.charts import binary as bn

_SRC = Path(bn.__file__).resolve()


def test_default_constant_is_the_documented_list():
    """The shipped default panel list is the one the docstring now names (20-F28)."""
    assert bn.DEFAULT_BINARY_PANELS == "ROC PR SCORE_DIST THRESHOLD"


def test_source_no_longer_claims_ks_and_gain_by_default():
    """The wrong 'defaults to ... KS THRESHOLD GAIN' claim is gone from the source (20-F28)."""
    text = _SRC.read_text(encoding="utf-8")
    assert '= "ROC PR SCORE_DIST KS THRESHOLD GAIN".' not in text
    assert "KS and GAIN are opt-in" in text
