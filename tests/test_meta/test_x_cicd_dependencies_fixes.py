"""Regression tests for audits/full_audit_2026-07-21/x_cicd_dependencies.md findings F1-F7.

PR1-PR4 are proposals (a workflow-level timeout meta-test suggestion, a lockfile-reproducibility
question, a lint-fail-fast ordering tradeoff, and a YAML-anchor de-duplication idea) with no reported
bug -- assessed, no fix required beyond what F1-F7 already cover.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    """Read a repo-relative file as UTF-8 text."""
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# F1: .pre-commit-config.yaml no longer has a duplicated tests/ lint bundle on a stale ruff pin
# ---------------------------------------------------------------------------


def _precommit_repos() -> list[dict]:
    """The ``repos`` list of the parsed ``.pre-commit-config.yaml``."""
    return yaml.safe_load(_read(".pre-commit-config.yaml"))["repos"]


def test_f1_no_duplicate_tests_lint_bundle():
    """Each tests/ blocking lint hook id is declared exactly once across the pre-commit config."""
    ids = [hook["id"] for repo in _precommit_repos() for hook in repo["hooks"]]
    assert ids, "no hooks parsed from .pre-commit-config.yaml"
    assert ids.count("interrogate-tests-blocking") == 1, "F1 REGRESSION: the tests/ blocking lint bundle must not be duplicated"
    assert ids.count("codespell-tests-blocking") == 1


def test_f1_no_stale_ruff_pin_remains():
    """No ruff-pre-commit repo is pinned to the stale v0.8.6 rev, and every one of them carries the same rev."""
    ruff_revs = [repo["rev"] for repo in _precommit_repos() if repo["repo"].endswith("astral-sh/ruff-pre-commit")]
    assert ruff_revs, "expected at least one ruff-pre-commit repo"
    assert "v0.8.6" not in ruff_revs, "F1 REGRESSION: the stale ruff-pre-commit rev must not remain anywhere in the file"
    assert len(set(ruff_revs)) == 1, f"ruff-pre-commit revs disagree: {ruff_revs}"


def test_f1_precommit_config_is_valid_yaml():
    """The pre-commit config parses to a mapping with a non-empty ``repos`` list whose entries each declare hooks."""
    doc = yaml.safe_load(_read(".pre-commit-config.yaml"))
    assert isinstance(doc, dict)
    assert doc["repos"], "the config declares no repos"
    assert all(repo["hooks"] for repo in doc["repos"])


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
    """Broader sweep: this repo's own documented convention is that every REGULAR (non-reusable-
    workflow-call) job sets timeout-minutes. A ``uses:``-based job is exempt: GitHub's schema only
    allows ``name``/``uses``/``with``/``secrets``/``needs``/``if``/``permissions`` on that job
    shape -- adding ``timeout-minutes`` there is a real YAML syntax error (confirmed via
    ``actionlint`` after an earlier, incorrect version of this test flagged 9 such jobs and the
    proposed "fix" broke `actionlint` on every one of them)."""
    import yaml

    workflow_dir = REPO_ROOT / ".github" / "workflows"
    missing = []
    for wf_path in sorted(workflow_dir.glob("*.yml")):
        doc = yaml.safe_load(wf_path.read_text(encoding="utf-8"))
        for job_name, job in (doc.get("jobs") or {}).items():
            if isinstance(job, dict) and "uses" not in job and "timeout-minutes" not in job:
                missing.append(f"{wf_path.name}::{job_name}")
    assert not missing, f"jobs missing timeout-minutes: {missing}"


def test_ci_jobs_have_timeout_minutes():
    """Wires the shared py_ci_shared.ci_workflow_timeout_gate check (meta-test proposal #12 from
    audits/full_audit_2026-07-21/META_TEST_PROPOSALS.md) across every workflow file -- an
    independent, regex-based implementation of the same invariant as
    test_f2_f3_every_job_in_every_workflow_has_a_timeout above (also uses:-exempt, see that
    function's docstring)."""
    from py_ci_shared.ci_workflow_timeout_gate import assert_all_jobs_have_timeout

    workflow_dir = REPO_ROOT / ".github" / "workflows"
    for wf_path in sorted(workflow_dir.glob("*.yml")):
        assert_all_jobs_have_timeout(wf_path)


# ---------------------------------------------------------------------------
# F4: dependabot's pip ecosystem is re-enabled at a small nonzero limit
# ---------------------------------------------------------------------------


def test_f4_dependabot_python_ecosystem_reenabled():
    """F4: the Python dependency ecosystem must stay enabled at a non-zero PR limit.

    Originally written against `package-ecosystem: pip`. It became `uv` on 2026-09-08 when uv.lock landed:
    the pip ecosystem reads pyproject.toml and does not know uv.lock exists, so every security-patch PR
    would have left the lock describing the old graph. The invariant F4 actually protects is that this
    repo keeps an automated security-patch signal at all -- dependency-review only inspects new deps in an
    incoming diff, and pip-audit is continue-on-error and opens nothing -- so it is asserted on whichever
    Python ecosystem is configured, not on the name `pip`.
    """
    import yaml

    doc = yaml.safe_load(_read(".github/dependabot.yml"))
    python_entries = [u for u in doc["updates"] if u["package-ecosystem"] in {"pip", "uv"}]
    assert len(python_entries) == 1, f"expected exactly one Python dependency ecosystem, found {len(python_entries)}"
    assert python_entries[0]["open-pull-requests-limit"] > 0, "F4 REGRESSION: the Python dependency ecosystem must not be silently permanently disabled"


# ---------------------------------------------------------------------------
# F5: the inert "FUTURE SKETCH" commented-out job block is trimmed from numba-coverage.yml
# ---------------------------------------------------------------------------


def _has_key(node: object, key: str) -> bool:
    """True when ``key`` is a mapping key anywhere inside the parsed YAML value ``node``."""
    if isinstance(node, dict):
        return key in node or any(_has_key(v, key) for v in node.values())
    if isinstance(node, list):
        return any(_has_key(v, key) for v in node)
    return False


def _commented_out_job_lines(text: str) -> list[int]:
    """First line of every contiguous comment block that, once uncommented, parses as YAML containing a ``runs-on`` key: an inert job."""
    blocks: list[tuple[int, list[str]]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            if blocks and blocks[-1][0] + len(blocks[-1][1]) == number:
                blocks[-1][1].append(stripped[1:])
            else:
                blocks.append((number, [stripped[1:]]))
    out = []
    for first, body in blocks:
        if any(_parses_to_a_job("\n".join(body[i:j])) for i in range(len(body)) for j in range(i + 1, len(body) + 1)):
            out.append(first)
    return out


def _parses_to_a_job(snippet: str) -> bool:
    """True when ``snippet`` is YAML whose parsed value contains a ``runs-on`` key."""
    try:
        return _has_key(yaml.safe_load(snippet), "runs-on")
    except yaml.YAMLError:
        return False


def test_commented_out_job_detector_finds_an_inert_job_and_ignores_prose():
    """A commented-out job body is reported at its first line; prose comments and a live job are not."""
    inert = "name: x\n# Why this is not built yet, in prose.\n# sketch:\n#   runs-on: ubuntu-latest\n#   steps: []\n# trailing prose\njobs: {}\n"
    assert _commented_out_job_lines(inert) == [2]
    assert _commented_out_job_lines("# a plain explanation: with a colon\n# and more prose\njobs:\n  a:\n    runs-on: x\n") == []


def test_f5_no_inert_future_sketch_block():
    """The workflow's jobs are all live (each has steps) and no comment block is a commented-out job."""
    path = REPO_ROOT / ".github" / "workflows" / "numba-coverage.yml"
    jobs = yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]
    assert set(jobs) == {"numba-disabled-coverage", "test-heavy-serial-numba-disabled", "merge-test-durations"}
    assert all(job["steps"] for job in jobs.values())
    assert _commented_out_job_lines(path.read_text(encoding="utf-8")) == []


def test_f5_numba_coverage_workflow_is_valid_yaml():
    """The numba coverage workflow parses to a mapping that declares jobs."""
    doc = yaml.safe_load(_read(".github/workflows/numba-coverage.yml"))
    assert isinstance(doc, dict)
    assert doc["jobs"]


# ---------------------------------------------------------------------------
# F6: covered by tests/test_meta/test_numba_coverage_workflow_exists.py's new
# test_numba_coverage_workflow_nightly_gate_is_intentionally_off -- imported here as a sanity check
# that it actually exists (not duplicating its assertions).
# ---------------------------------------------------------------------------


def test_f6_nightly_gate_meta_test_exists():
    """The referenced nightly-gate meta-test exists as a test function and passes when run."""
    from tests.test_meta.test_numba_coverage_workflow_exists import (
        test_numba_coverage_workflow_nightly_gate_is_intentionally_on,
    )

    assert inspect.isfunction(test_numba_coverage_workflow_nightly_gate_is_intentionally_on)
    assert test_numba_coverage_workflow_nightly_gate_is_intentionally_on.__name__.startswith("test_")
    test_numba_coverage_workflow_nightly_gate_is_intentionally_on()


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
    """The GPU extras matrix workflow parses and its resolve job declares a non-empty include matrix."""
    doc = yaml.safe_load(_read(".github/workflows/gpu-extras-install-matrix.yml"))
    assert doc["jobs"]["resolve"]["strategy"]["matrix"]["include"]
