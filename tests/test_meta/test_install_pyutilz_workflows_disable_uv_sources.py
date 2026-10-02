"""Every job that installs pyutilz from a local clone must also set ``UV_NO_SOURCES=1``.

``[tool.uv.sources]`` points pyutilz at git, while the ``install-pyutilz`` action installs it from a sibling clone. With both in play uv
refuses to resolve ("conflicting URLs for package pyutilz"), so the job dies at install before running anything. The scheduled
fs-benchmark workflow omitted the variable and failed that way on every run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.test_meta._scan_guard import assert_scanned_enough

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def _jobs_installing_pyutilz():
    """Yield ``(workflow file, job id, workflow env, job)`` for each job with an install-pyutilz step."""
    for path in sorted(WORKFLOWS.glob("*.yml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for job_id, job in (doc.get("jobs") or {}).items():
            steps = job.get("steps") or []
            if any("install-pyutilz" in str(step.get("uses", "")) for step in steps):
                yield path.name, job_id, doc.get("env") or {}, job


@pytest.mark.parametrize("workflow, job_id, workflow_env, job", list(_jobs_installing_pyutilz()), ids=lambda v: v if isinstance(v, str) else "")
def test_a_job_installing_pyutilz_from_a_clone_ignores_the_uv_sources_table(workflow, job_id, workflow_env, job):
    """The job (or its workflow) sets ``UV_NO_SOURCES`` to 1."""
    env = {**workflow_env, **(job.get("env") or {})}
    assert str(env.get("UV_NO_SOURCES")) == "1", f"{workflow}::{job_id} installs pyutilz from a clone but leaves [tool.uv.sources] in force"


def test_the_scan_found_the_jobs_it_is_meant_to_check():
    """A scan that matches nothing would pass for the wrong reason."""
    assert_scanned_enough(len(list(WORKFLOWS.glob("*.yml"))), ".github/workflows", minimum=3)
    assert {name for name, *_ in _jobs_installing_pyutilz()} >= {"ci.yml", "fs-benchmark-nightly.yml"}
