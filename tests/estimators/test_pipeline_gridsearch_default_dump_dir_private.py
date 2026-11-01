"""The gridsearch dump written without an explicit ``output_dir`` is not at a name predictable from the shared temp root (FS-02).

Before the fix the dump went to ``tempfile.gettempdir()/cv_results-<title>.dump``. Any local user could pre-create that
path, and the file is read back with ``joblib.load``. The default location is now a fresh private directory.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from mlframe.estimators.pipelines import optimize_pipeline_by_gridsearch


def _fake_cv_func(X, Y, title, **constants):
    """Return a canned CV result so the gridsearch completes without fitting anything."""
    return {"results": {"cv_results": {"model_a": {"metrics": {"root_mean_squared_error": [0.1, 0.2, 0.15]}}}}}


def test_default_dump_is_not_at_the_predictable_shared_temp_path(monkeypatch, tmp_path):
    """Without output_dir the dump lands in a new private directory, never directly in the shared temp root."""
    import joblib

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    seen = []
    real_dump = joblib.dump

    def spy(obj, path, *a, **k):
        """Record the dump path and the private directory's mode while the dump is being written."""
        parent = Path(path).parent
        seen.append((Path(path), parent, parent.stat().st_mode))
        return real_dump(obj, path, *a, **k)

    monkeypatch.setattr(joblib, "dump", spy)
    optimize_pipeline_by_gridsearch(X=None, Y=None, title="fs02_t", cv_func=_fake_cv_func)
    assert len(seen) == 1, seen
    path, parent, mode = seen[0]
    assert path != tmp_path / "cv_results-fs02_t.dump", "dump written to the predictable shared temp path"
    assert parent != tmp_path
    if os.name == "posix":
        assert (mode & 0o077) == 0


def test_default_scratch_directory_is_removed_when_the_search_ends(monkeypatch, tmp_path):
    """The auto-created dump directory is unreachable for the caller, so it must not outlive the call (RES4-1)."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    result = optimize_pipeline_by_gridsearch(X=None, Y=None, title="res41", cv_func=_fake_cv_func)
    assert "res41" in result
    assert list(tmp_path.iterdir()) == []


def test_explicit_output_dir_keeps_its_dump(tmp_path):
    """A caller-supplied directory is the caller's: the dump stays."""
    optimize_pipeline_by_gridsearch(X=None, Y=None, title="res41k", cv_func=_fake_cv_func, output_dir=str(tmp_path))
    assert (tmp_path / "cv_results-res41k.dump").exists()
