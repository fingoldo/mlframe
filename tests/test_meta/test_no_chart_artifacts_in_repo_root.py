"""No rendered chart may be tracked at the repository root.

A test that renders to a bare filename instead of ``tmp_path`` writes into whatever the working directory
happens to be, which for a pytest run is the repo root. The file then looks like an ordinary untracked
change, and the next ``git add -A`` commits it. That is exactly how ``calibration.html`` reached master on
2026-09-08 -- a 7-line plotly document, added by a commit about linter configuration, noticed only because
a later run under a different plotly version rewrote it and the diff appeared again.

Two things are wrong with that and this gate covers the durable one: the artifact must not be tracked. The
other half, finding the test that renders without ``tmp_path``, is a hunt through ~60 reporting tests and
is tracked in audits/ci_review_2026-09-08/_TRACKER.md (X4). Until it is found, .gitignore keeps the file
out of ``git add`` and this gate keeps it out of the index.
"""

from __future__ import annotations

import subprocess  # nosec B404 - runs git against this repo, no external input
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Formats a chart renderer emits. Data files (.json, .toml, .cfg) are deliberately absent: several are
# legitimately tracked at the root.
ARTIFACT_SUFFIXES = {".html", ".png", ".svg", ".pdf", ".jpg", ".jpeg", ".gif", ".webp"}


def _tracked_root_files() -> list[str]:
    """Every file git tracks directly at the repository root."""
    proc = subprocess.run(  # nosec B603 - fixed argv, no shell
        ["git", "ls-files", "--full-name"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        pytest.skip(f"git unavailable or not a checkout: {proc.stderr.strip()[:200]}")
    names = [line for line in proc.stdout.splitlines() if line and "/" not in line]
    assert names, "git tracks no files at the repository root; this gate cannot run and must not report green"
    return names


def test_no_rendered_chart_is_tracked_at_the_repo_root():
    """A chart at the root is a test artifact someone committed by accident, not a source file."""
    stray = sorted(n for n in _tracked_root_files() if Path(n).suffix.lower() in ARTIFACT_SUFFIXES)
    assert not stray, (
        f"{len(stray)} rendered chart artifact(s) are tracked at the repository root. They come from a "
        f"test that rendered to a bare filename instead of tmp_path, and were committed by a `git add -A`. "
        f"Remove them from the index and give the test an explicit tmp_path destination:\n" + "\n".join(f"  {n}" for n in stray)
    )


def test_gitignore_keeps_root_chart_artifacts_out_of_git_add():
    """The gate above catches the commit; .gitignore is what stops it being staged in the first place."""
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    missing = [pattern for pattern in ("/*.html", "/*.png", "/*.svg") if pattern not in text]
    assert not missing, "`.gitignore` no longer excludes root-level chart artifacts, so a stray render is stageable again: " + ", ".join(missing)


def test_no_untracked_chart_artifact_is_sitting_in_the_root_right_now():
    """A stray render present but ignored still means some test is writing to the working directory."""
    stray = sorted(p.name for p in REPO_ROOT.iterdir() if p.is_file() and p.suffix.lower() in ARTIFACT_SUFFIXES)
    if stray:
        pytest.skip(
            "chart artifact(s) in the repo root from a test rendering without tmp_path "
            f"(ignored by .gitignore, so harmless to the index): {', '.join(stray)}. "
            "See _TRACKER.md X4 -- this skip is the standing reminder that the writer is still unfound."
        )


if __name__ == "__main__":  # pragma: no cover - convenience for a manual check
    sys.exit(pytest.main([__file__, "-v"]))
