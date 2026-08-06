"""X_OSS_HYGIENE_PACKAGING-4 regression test: docs/MRMR_RESEARCH.md's 'Critical gaps' section must not
contradict its own 'Recommendations'/backlog sections about the Fleuret synergy-rejection fix.

Pre-fix, 'Critical gaps' called the JMI/JMIM synergy fix "still unfixed" / said there was "No protection
against synergy destruction", while the same doc's Recommendations section and backlog table both mark
the JMIM redundancy_aggregator as shipped, and fleuret.py's own docstring confirms it.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_critical_gaps_does_not_claim_synergy_fix_is_unfixed():
    """The 'Critical gaps' section must not claim the JMI/JMIM synergy-rejection mitigation is unfixed."""
    doc = (REPO_ROOT / "docs" / "MRMR_RESEARCH.md").read_text(encoding="utf-8")
    assert "still unfixed" not in doc, "MRMR_RESEARCH.md still claims a fix is 'unfixed' -- check against the Recommendations/backlog sections for staleness"
    idx = doc.find("No protection against synergy destruction")
    assert idx != -1, "expected bullet text not found -- doc structure changed, update this test"
    preceding = doc[max(0, idx - 5) : idx]
    assert "~~" in preceding, "the synergy-destruction gap bullet must be struck through (shipped), matching the Recommendations section"


def test_fleuret_module_confirms_jmim_synergy_mitigation_shipped():
    """Sanity: fleuret.py's own docstring is the ground truth this doc must agree with."""
    src = (REPO_ROOT / "src" / "mlframe" / "feature_selection" / "filters" / "fleuret.py").read_text(encoding="utf-8")
    assert "is shipped" in src
    assert "redundancy_aggregator='jmim'" in src
