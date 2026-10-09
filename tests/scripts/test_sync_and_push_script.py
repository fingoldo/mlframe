"""scripts/sync_and_push.sh parses, and its loop merges what arrived and pushes, in a throwaway pair of repositories."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "sync_and_push.sh"
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None or shutil.which("git") is None, reason="needs bash and git")


def _git(cwd: Path, *args: str) -> str:
    """Run git in ``cwd`` and return stdout, failing the test on a non-zero exit."""
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", check=True)
    return done.stdout.strip()


def _clone_pair(tmp_path: Path) -> "tuple[Path, Path, Path]":
    """A bare remote, a working clone ``mine`` and a second clone ``other`` standing in for the other session."""
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", "-b", "master", str(remote))
    mine, other = tmp_path / "mine", tmp_path / "other"
    for clone in (mine, other):
        subprocess.run(["git", "clone", str(remote), str(clone)], check=True, capture_output=True)
        _git(clone, "config", "user.email", "t@example.com")
        _git(clone, "config", "user.name", "t")
        _git(clone, "config", "core.autocrlf", "false")
        _git(clone, "config", "core.hooksPath", str(tmp_path / "nohooks"))
    (mine / "base.txt").write_text("base\n", encoding="utf-8")
    _git(mine, "add", "base.txt")
    _git(mine, "commit", "-m", "base")
    _git(mine, "push", "origin", "master")
    _git(other, "pull", "origin", "master")
    return remote, mine, other


def _run(clone: Path, *args: str) -> subprocess.CompletedProcess:
    """Run the script from inside ``clone`` with a private log directory."""
    return subprocess.run(
        [str(BASH), str(SCRIPT), *args], cwd=clone, capture_output=True, text=True, encoding="utf-8", env={**os.environ, "TMPDIR": str(clone.parent)}
    )


def test_the_script_has_valid_bash_syntax() -> None:
    """``bash -n`` accepts the file."""
    done = subprocess.run([str(BASH), "-n", str(SCRIPT)], capture_output=True, text=True, encoding="utf-8")
    assert done.returncode == 0, done.stderr


def test_it_merges_a_remote_that_moved_and_pushes(tmp_path) -> None:
    """The remote is one commit ahead on a different file; the script merges it and the push lands, so the remote contains both commits."""
    remote, mine, other = _clone_pair(tmp_path)
    (other / "theirs.txt").write_text("t\n", encoding="utf-8")
    _git(other, "add", "theirs.txt")
    _git(other, "commit", "-m", "theirs")
    _git(other, "push", "origin", "master")
    (mine / "mine.txt").write_text("m\n", encoding="utf-8")
    _git(mine, "add", "mine.txt")
    _git(mine, "commit", "-m", "mine")

    done = _run(mine)

    assert done.returncode == 0, done.stdout + done.stderr
    files = _git(tmp_path, "--git-dir", str(remote), "ls-tree", "-r", "--name-only", "master").split()
    assert {"base.txt", "theirs.txt", "mine.txt"} <= set(files)


def test_a_conflict_stops_the_loop_without_pushing(tmp_path) -> None:
    """Both sides edit the same line: the script exits with 3, names the file and leaves the remote untouched."""
    remote, mine, other = _clone_pair(tmp_path)
    (other / "base.txt").write_text("theirs\n", encoding="utf-8")
    _git(other, "commit", "-am", "theirs")
    _git(other, "push", "origin", "master")
    (mine / "base.txt").write_text("mine\n", encoding="utf-8")
    _git(mine, "commit", "-am", "mine")
    before = _git(tmp_path, "--git-dir", str(remote), "rev-parse", "master")

    done = _run(mine)

    assert done.returncode == 3, done.stdout + done.stderr
    assert "base.txt" in done.stdout
    assert _git(tmp_path, "--git-dir", str(remote), "rev-parse", "master") == before
