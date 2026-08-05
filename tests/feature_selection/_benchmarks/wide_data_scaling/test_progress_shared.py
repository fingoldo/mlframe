"""FS_BENCHMARKS_C-1 (2026-08-05 audit): ``ck()`` (the shared progress-log checkpoint helper for the
wide_data_scaling H1/H2 bench family) hardcodes an absolute path (``D:/Temp/synergy_scale_bench/
progress.txt``) with no directory-existence check. ``open(path, "a")`` creates a missing FILE but raises
``FileNotFoundError`` if the parent DIRECTORY tree does not exist -- and since ``ck()`` is called as the
very first action of both ``h1_bench.py`` and ``h2_bench.py`` (before any real sweep work), a missing
``D:/Temp/synergy_scale_bench/`` directory aborted the whole multi-minute sweep before a single result was
produced, on any machine/session where that directory hadn't already been created.
"""

from __future__ import annotations

from mlframe.feature_selection._benchmarks.wide_data_scaling import _progress_shared


def test_ck_creates_missing_progress_dir(tmp_path, monkeypatch):
    """ck() must create its progress log's parent directory tree, not assume it already exists."""
    missing_dir = tmp_path / "does_not_exist_yet" / "nested"
    prog_path = missing_dir / "progress.txt"
    monkeypatch.setattr(_progress_shared, "PROG", str(prog_path))

    assert not missing_dir.exists()
    _progress_shared.ck("hello")  # must not raise FileNotFoundError
    assert prog_path.exists()
    assert "hello" in prog_path.read_text()


def test_ck_appends_across_multiple_calls_into_existing_dir(tmp_path, monkeypatch):
    """Sanity: once the directory exists (either pre-created or from a prior ck() call), subsequent calls
    keep appending rather than truncating."""
    prog_path = tmp_path / "progress.txt"
    monkeypatch.setattr(_progress_shared, "PROG", str(prog_path))

    _progress_shared.ck("first")
    _progress_shared.ck("second")

    content = prog_path.read_text()
    assert "first" in content
    assert "second" in content
    assert content.index("first") < content.index("second")
