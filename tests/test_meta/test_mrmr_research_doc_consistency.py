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


def test_doc_does_not_cite_the_deleted_rfecv_monolith_path():
    """X_OSS_HYGIENE_PACKAGING-5: MRMR_RESEARCH.md must not cite wrappers/_rfecv.py -- RFECV was split
    into a wrappers/rfecv/ subpackage and that flat-file path no longer exists."""
    doc = (REPO_ROOT / "docs" / "MRMR_RESEARCH.md").read_text(encoding="utf-8")
    assert "wrappers/_rfecv.py" not in doc
    assert not (
        REPO_ROOT / "src" / "mlframe" / "feature_selection" / "wrappers" / "_rfecv.py"
    ).exists(), "wrappers/_rfecv.py exists again -- the doc citation may now be valid; re-check this test's premise"
    assert (REPO_ROOT / "src" / "mlframe" / "feature_selection" / "wrappers" / "rfecv").is_dir()


def test_dummy_baselines_guide_cites_a_real_logger_name():
    """X_OSS_HYGIENE_PACKAGING-7: dummy_baselines_guide.md must instruct users to raise a logger name
    that actually exists (mlframe.training.baselines), not the pre-split mlframe.training.dummy_baselines
    name, which is not a parent of any of the split submodules' loggers and so is a complete no-op."""
    doc = (REPO_ROOT / "docs" / "dummy_baselines_guide.md").read_text(encoding="utf-8")
    assert "raise\nthe logger level for `mlframe.training.dummy_baselines`" not in doc
    assert "raise\nthe logger level for `mlframe.training.baselines`" in doc
    baselines_dir = REPO_ROOT / "src" / "mlframe" / "training" / "baselines"
    assert baselines_dir.is_dir()
    modules_using_dunder_name_logger = [f for f in baselines_dir.glob("*.py") if "logging.getLogger(__name__)" in f.read_text(encoding="utf-8")]
    assert modules_using_dunder_name_logger, "no baselines submodule uses getLogger(__name__) -- re-check this test's premise"


def test_dummy_baselines_guide_profiling_commands_point_at_real_files():
    """X_OSS_HYGIENE_PACKAGING-8: the documented profiling/smoke commands must reference files that
    actually exist under src/mlframe/training/baselines/, not the pre-split mlframe/training/ paths."""
    doc = (REPO_ROOT / "docs" / "dummy_baselines_guide.md").read_text(encoding="utf-8")
    assert "python -m mlframe.training._profile_dummy_baselines" not in doc
    assert "python mlframe/training/_smoke_dummy_baselines_e2e.py" not in doc
    baselines_dir = REPO_ROOT / "src" / "mlframe" / "training" / "baselines"
    assert (baselines_dir / "_profile_dummy_baselines.py").exists()
    assert (baselines_dir / "_smoke_dummy_baselines_e2e.py").exists()
    assert "python -m mlframe.training.baselines._profile_dummy_baselines" in doc
    assert "python src/mlframe/training/baselines/_smoke_dummy_baselines_e2e.py" in doc
