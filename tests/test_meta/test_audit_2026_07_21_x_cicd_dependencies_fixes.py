"""Regression tests for audits/full_audit_2026-07-21/x_cicd_dependencies.md findings F1-F7.

PR1-PR4 are proposals (a workflow-level timeout meta-test suggestion, a lockfile-reproducibility
question, a lint-fail-fast ordering tradeoff, and a YAML-anchor de-duplication idea) with no reported
bug -- assessed, no fix required beyond what F1-F7 already cover.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    """Read a repo-relative file as UTF-8 text."""
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# F1: .pre-commit-config.yaml no longer has a duplicated tests/ lint bundle on a stale ruff pin
# ---------------------------------------------------------------------------


def test_f1_no_duplicate_tests_lint_bundle():
    """F1 no duplicate tests lint bundle."""
    text = _read(".pre-commit-config.yaml")
    assert text.count("id: interrogate-tests-blocking") == 1, "F1 REGRESSION: the tests/ blocking lint bundle must not be duplicated"
    assert text.count("id: codespell-tests-blocking") == 1


def test_f1_no_stale_ruff_pin_remains():
    """F1 no stale ruff pin remains."""
    text = _read(".pre-commit-config.yaml")
    assert "v0.8.6" not in text, "F1 REGRESSION: the stale ruff-pre-commit rev must not remain anywhere in the file"


def test_f1_precommit_config_is_valid_yaml():
    """F1 precommit config is valid yaml."""
    import yaml

    yaml.safe_load(_read(".pre-commit-config.yaml"))


# ---------------------------------------------------------------------------
# F2/F3: ci.yml's build job and release.yml's publish job now carry an explicit timeout-minutes
# ---------------------------------------------------------------------------


def test_f2_ci_build_job_has_timeout():
    """F2 ci build job has timeout."""
    import yaml

    doc = yaml.safe_load(_read(".github/workflows/ci.yml"))
    build_job = doc["jobs"]["build"]
    assert "timeout-minutes" in build_job, "F2 REGRESSION: ci.yml's build job must set an explicit timeout-minutes"
    assert build_job["timeout-minutes"] > 0


def test_f3_release_publish_job_has_timeout():
    """F3 release publish job has timeout."""
    import yaml

    doc = yaml.safe_load(_read(".github/workflows/release.yml"))
    publish_job = doc["jobs"]["publish"]
    assert "timeout-minutes" in publish_job, "F3 REGRESSION: release.yml's publish job must set an explicit timeout-minutes"
    assert publish_job["timeout-minutes"] > 0


def test_f2_f3_every_job_in_every_workflow_has_a_timeout():
    """Broader sweep: this repo's own documented convention is that EVERY job sets timeout-minutes."""
    import yaml

    workflow_dir = REPO_ROOT / ".github" / "workflows"
    missing = []
    for wf_path in sorted(workflow_dir.glob("*.yml")):
        doc = yaml.safe_load(wf_path.read_text(encoding="utf-8"))
        for job_name, job in (doc.get("jobs") or {}).items():
            if isinstance(job, dict) and "uses" not in job and "timeout-minutes" not in job:
                missing.append(f"{wf_path.name}::{job_name}")
    assert not missing, f"jobs missing timeout-minutes: {missing}"


# ---------------------------------------------------------------------------
# F4: dependabot's pip ecosystem is re-enabled at a small nonzero limit
# ---------------------------------------------------------------------------


def test_f4_dependabot_pip_ecosystem_reenabled():
    """F4 dependabot pip ecosystem re-enabled."""
    import yaml

    doc = yaml.safe_load(_read(".github/dependabot.yml"))
    pip_entries = [u for u in doc["updates"] if u["package-ecosystem"] == "pip"]
    assert len(pip_entries) == 1
    assert pip_entries[0]["open-pull-requests-limit"] > 0, "F4 REGRESSION: pip ecosystem must not be silently permanently disabled"


# ---------------------------------------------------------------------------
# F5: the inert "FUTURE SKETCH" commented-out job block is trimmed from numba-coverage.yml
# ---------------------------------------------------------------------------


def test_f5_no_inert_future_sketch_block():
    """F5 no inert future sketch block."""
    text = _read(".github/workflows/numba-coverage.yml")
    assert "FUTURE SKETCH" not in text
    # The rationale (why not implemented) must still be preserved, just compressed.
    assert "codecov-numba" in text


def test_f5_numba_coverage_workflow_is_valid_yaml():
    """F5 numba coverage workflow is valid yaml."""
    import yaml

    yaml.safe_load(_read(".github/workflows/numba-coverage.yml"))


# ---------------------------------------------------------------------------
# F6: covered by tests/test_meta/test_numba_coverage_workflow_exists.py's new
# test_numba_coverage_workflow_nightly_gate_is_intentionally_off -- imported here as a sanity check
# that it actually exists (not duplicating its assertions).
# ---------------------------------------------------------------------------


def test_f6_nightly_gate_meta_test_exists():
    """F6 nightly gate meta test exists."""
    from tests.test_meta.test_numba_coverage_workflow_exists import (
        test_numba_coverage_workflow_nightly_gate_is_intentionally_off,
    )

    test_numba_coverage_workflow_nightly_gate_is_intentionally_off()


# ---------------------------------------------------------------------------
# F7: gpu-extras-install-matrix.yml's cuda12x "gpu" row is now marked experimental at Python 3.14,
# matching the repo's blanket 3.14-experimental policy (previously only gpu-cuda11 was marked)
# ---------------------------------------------------------------------------


def test_f7_gpu_cuda12x_row_marked_experimental_at_py314():
    """F7 gpu cuda12x row marked experimental at py314."""
    import yaml

    doc = yaml.safe_load(_read(".github/workflows/gpu-extras-install-matrix.yml"))
    include = doc["jobs"]["resolve"]["strategy"]["matrix"]["include"]
    gpu_314_entries = [e for e in include if e.get("extra") == "gpu" and e.get("python-version") == "3.14"]
    assert len(gpu_314_entries) == 1, "F7 REGRESSION: the cuda12x 'gpu' extra at Python 3.14 must have an experimental include entry"
    assert gpu_314_entries[0]["experimental"] is True


def test_f7_gpu_extras_matrix_is_valid_yaml():
    """F7 gpu extras matrix is valid yaml."""
    import yaml

    yaml.safe_load(_read(".github/workflows/gpu-extras-install-matrix.yml"))
