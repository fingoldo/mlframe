"""Cluster 12 regression: bench results go to a configurable folder, not a hard-coded D:/Temp (audit 2026-11-20, finding 12-F98).

bench_out honours MLFRAME_BENCH_OUT and otherwise writes under _results next to the package and creates it. A source-level guard
keeps any code-level D:/Temp literal from returning to the fs_hybrid bench scripts (comments and docstrings are allowed).
"""

import os
import re
from pathlib import Path

from mlframe.feature_selection._benchmarks.fs_hybrid import _bench_out

_FS_HYBRID = Path(_bench_out.__file__).resolve().parent


def test_env_override_sets_the_folder(monkeypatch, tmp_path):
    """MLFRAME_BENCH_OUT redirects results and the folder is created (12-F98)."""
    target = tmp_path / "results"
    monkeypatch.setenv("MLFRAME_BENCH_OUT", str(target))
    path = _bench_out.bench_out("rows.csv")
    assert path == os.path.join(str(target), "rows.csv")
    assert target.is_dir()


def test_default_is_results_next_to_the_package(monkeypatch):
    """Without the override the default folder is _results beside the bench package (12-F98)."""
    monkeypatch.delenv("MLFRAME_BENCH_OUT", raising=False)
    monkeypatch.setattr(_bench_out.os, "makedirs", lambda *a, **k: None)  # check the path without creating _results in the tree
    path = _bench_out.bench_out("rows.csv")
    assert os.path.dirname(path) == str(_FS_HYBRID / "_results")
    assert "D:" not in path


def test_no_code_level_d_temp_literal_in_fs_hybrid():
    """No non-comment line in the fs_hybrid bench scripts writes to D:/Temp (12-F98)."""
    pattern = re.compile(r'"D:(?:/|\\)Temp')
    offenders = []
    for path in sorted(_FS_HYBRID.glob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if pattern.search(line) and "``" not in line and '"""' not in line:
                offenders.append(f"{path.name}:{number}")
    assert offenders == []
