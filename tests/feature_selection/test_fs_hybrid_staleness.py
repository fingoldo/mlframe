"""A published result has to carry its own expiry date, or it reads as true forever.

This repository already contains two hundred `bench_*` scripts whose results nobody can date. They are not
wrong; they went stale silently, which is worse, because a number with no expiry is quoted with the same
confidence on the day it stops being true as on the day it was measured.
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - builds a throwaway git repository for the test; fixed argument lists only
from pathlib import Path
from typing import Any, List

import pytest

from mlframe.feature_selection._benchmarks.fs_hybrid._staleness import STALE_COMMITS, WARN_COMMITS, WATCHED_PATHS, assess, commits_since, format_staleness


def _run(args: List[str], cwd: Path) -> None:
    """Run one git command in the scratch repository."""
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, timeout=60)  # nosec B603 B607 - fixed argument list, no shell


@pytest.fixture()
def repo(tmp_path: Path) -> Any:
    """Build a throwaway repository with commits on a watched path and on an unwatched one."""
    root = tmp_path / "repo"
    watched = root / WATCHED_PATHS[0]
    watched.mkdir(parents=True)
    _run(["init", "-q", "repo"], tmp_path)
    _run(["config", "user.email", "t@example.invalid"], root)
    _run(["config", "user.name", "test"], root)
    (watched / "seed.py").write_text("x = 0\n", encoding="utf-8")
    _run(["add", "-A"], root)
    _run(["commit", "-q", "-m", "seed"], root)
    return root


def _head(root: Path) -> str:
    """Return the repository's current HEAD sha."""
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True, timeout=60)  # nosec B603 B607 - fixed argument list, no shell
    return out.stdout.strip()


def _commit_to(root: Path, relative: str, count: int) -> None:
    """Make ``count`` commits touching one path."""
    target = root / relative
    target.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (target / f"f{index}.py").write_text(f"v = {index}\n", encoding="utf-8")
        _run(["add", "-A"], root)
        _run(["commit", "-q", "-m", f"change {index}"], root)


def _manifest(root: Path, sha: str) -> str:
    """Write a manifest recording one commit and return its path."""
    path = root / "MANIFEST.json"
    path.write_text(json.dumps({"git_sha": sha}), encoding="utf-8")
    return str(path)


def test_a_result_at_head_is_current(repo: Path) -> None:
    """Zero commits behind is the only state that needs no caveat."""
    verdict = assess(_manifest(repo, _head(repo)), repo=str(repo))

    assert verdict.verdict == "current"
    assert verdict.commits_behind == 0


def test_commits_to_unwatched_paths_do_not_age_a_result(repo: Path) -> None:
    """A month of work on the reporting layer changes nothing about which features a selector picks.

    Ageing on total repository activity would make every result stale within days and the check would be
    ignored, which is the failure mode a threshold is supposed to avoid.
    """
    base = _head(repo)
    _commit_to(repo, "src/mlframe/reporting", STALE_COMMITS + 5)

    verdict = assess(_manifest(repo, base), repo=str(repo))

    assert verdict.commits_behind == 0
    assert verdict.verdict == "current"


def test_enough_commits_to_a_watched_path_mark_a_result_drifting(repo: Path) -> None:
    """Between the two thresholds the result still means something but should be re-run."""
    base = _head(repo)
    _commit_to(repo, WATCHED_PATHS[0], WARN_COMMITS + 1)

    verdict = assess(_manifest(repo, base), repo=str(repo))

    assert verdict.verdict == "drifting"
    assert verdict.commits_behind == WARN_COMMITS + 1


def test_far_enough_behind_is_stale(repo: Path) -> None:
    """Past the upper threshold a document quoting the result has to say it no longer describes the code."""
    base = _head(repo)
    _commit_to(repo, WATCHED_PATHS[0], STALE_COMMITS + 2)

    verdict = assess(_manifest(repo, base), repo=str(repo))

    assert verdict.verdict == "stale"
    assert "no longer describes" in verdict.reason


def test_a_manifest_without_a_commit_is_unknown_rather_than_current(repo: Path) -> None:
    """Guessing is worse than saying so: an undatable result would otherwise read as a fresh one."""
    path = repo / "MANIFEST.json"
    path.write_text(json.dumps({"scenarios": ["a"]}), encoding="utf-8")

    verdict = assess(str(path), repo=str(repo))

    assert verdict.verdict == "unknown"
    assert "records no commit" in verdict.reason


def test_a_missing_manifest_is_unknown_and_does_not_raise(repo: Path) -> None:
    """A results file with no manifest beside it is a state to report, not a crash."""
    verdict = assess(str(repo / "absent.json"), repo=str(repo))

    assert verdict.verdict == "unknown"
    assert verdict.commits_behind is None


def test_counting_against_an_unknown_revision_returns_none(repo: Path) -> None:
    """A manifest naming a commit this checkout does not have cannot be dated, and says so."""
    assert commits_since("0" * 40, str(repo)) is None


def test_the_report_names_the_verdict_and_the_distance(repo: Path) -> None:
    """A table of shas nobody can interpret is the same as no table."""
    base = _head(repo)
    _commit_to(repo, WATCHED_PATHS[0], WARN_COMMITS + 1)

    lines = format_staleness({"scm_beds": assess(_manifest(repo, base), repo=str(repo))})

    assert any("scm_beds" in line and "drifting" in line for line in lines)


def test_the_report_says_so_when_there_is_nothing_to_date() -> None:
    """An empty table must read as "nothing is dated", never as "everything is current"."""
    assert any("nothing here is dated" in line for line in format_staleness({}))


def test_the_thresholds_are_ordered() -> None:
    """A warn threshold above the stale one would make the drifting state unreachable."""
    assert 0 < WARN_COMMITS < STALE_COMMITS


def test_the_watched_paths_exist_in_this_repository() -> None:
    """A watched path that was renamed silently stops ageing anything, which reads as permanent freshness."""
    root = Path(__file__).resolve().parents[2]

    assert WATCHED_PATHS
    for relative in WATCHED_PATHS:
        assert (root / relative).is_dir(), f"the staleness check watches {relative!r}, which this repository does not have"


def test_the_measured_paths_cover_the_code_the_atlas_is_about() -> None:
    """The atlas is a claim about selectors and about the beds they run on; both have to be watched."""
    joined = " ".join(WATCHED_PATHS)

    assert "feature_selection" in joined
    assert "datasets" in joined
    assert os.sep not in WATCHED_PATHS[0], "paths are stored posix-style so git accepts them on every platform"
