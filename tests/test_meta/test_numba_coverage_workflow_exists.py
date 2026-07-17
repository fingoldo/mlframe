"""Meta-test: the nightly NUMBA_DISABLE_JIT coverage workflow must exist and be wired correctly.

Per project memory ``reference_numba_coverage_blind``, every ``@njit`` kernel body is invisible to ``sys.settrace`` / ``coverage.py``. The nightly workflow runs the kernel-heavy test suites with ``NUMBA_DISABLE_JIT=1`` so the bodies become visible. If anyone deletes the workflow or weakens its configuration (drops the env var, removes the cron schedule, narrows the test set), this sensor fails and the regression is caught at PR time rather than 6 months later when coverage silently regressed.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "numba-coverage.yml"


def _workflow_text() -> str:
    assert WORKFLOW_PATH.exists(), f"missing nightly numba-coverage workflow at {WORKFLOW_PATH}"
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def test_numba_coverage_workflow_file_exists() -> None:
    assert WORKFLOW_PATH.exists(), (
        f"nightly numba-coverage workflow missing at {WORKFLOW_PATH}. "
        "This workflow makes @njit kernel bodies visible to coverage.py by setting NUMBA_DISABLE_JIT=1. "
        "See architectural_proposals/numba_coverage_ci.md."
    )


def test_numba_coverage_workflow_sets_numba_disable_jit_env() -> None:
    text = _workflow_text()
    assert "NUMBA_DISABLE_JIT" in text, "workflow must set NUMBA_DISABLE_JIT env var to disable JIT"
    assert 'NUMBA_DISABLE_JIT: "1"' in text or "NUMBA_DISABLE_JIT: '1'" in text or "NUMBA_DISABLE_JIT: 1" in text, (
        "NUMBA_DISABLE_JIT must be set to 1 in the workflow env so kernels execute as pure Python"
    )


def test_numba_coverage_workflow_has_nightly_cron() -> None:
    text = _workflow_text()
    assert "schedule:" in text, "workflow must declare a cron schedule (nightly run)"
    assert "cron:" in text, "workflow must use cron scheduling"
    assert "0 3 * * *" in text, "expected nightly 03:00 UTC cron ('0 3 * * *'); change is fine but update this assertion"


def test_numba_coverage_workflow_supports_manual_dispatch() -> None:
    text = _workflow_text()
    assert "workflow_dispatch" in text, "workflow_dispatch trigger missing; operators must be able to run the nightly suite on demand"


def test_numba_coverage_workflow_targets_kernel_heavy_dirs() -> None:
    """The whole point of the workflow is to cover kernel bodies in these directories."""
    text = _workflow_text()
    for required_dir in (
        "tests/feature_engineering/",
        "tests/feature_selection/",
        "tests/metrics/",
        "tests/training/",
    ):
        assert required_dir in text, f"workflow must run tests under {required_dir} for kernel-body coverage to materialise"


def test_numba_coverage_workflow_collects_coverage() -> None:
    text = _workflow_text()
    assert "--cov=src/mlframe" in text, "workflow must collect coverage on src/mlframe"
    assert "--cov-report=xml" in text, "workflow must produce coverage.xml for upload/artifact"
