"""The staleness check resolves git against the repository root, not the benchmark folder (audit 2026-11-20, finding 12-F1)."""
import os

from mlframe.feature_selection._benchmarks.fs_hybrid._staleness import _repo_root


def test_repo_root_contains_the_git_entry():
    """The default root must be the directory holding .git, so git paths resolve; the old default was the module folder."""
    root = _repo_root()
    assert os.path.exists(os.path.join(root, ".git"))
    assert os.path.isdir(os.path.join(root, "src", "mlframe"))
