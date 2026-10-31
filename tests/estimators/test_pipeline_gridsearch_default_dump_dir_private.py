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
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    optimize_pipeline_by_gridsearch(X=None, Y=None, title="fs02_t", cv_func=_fake_cv_func)
    predictable = tmp_path / "cv_results-fs02_t.dump"
    assert not predictable.exists(), "dump written to the predictable shared temp path"
    dumps = list(Path(tmp_path).rglob("cv_results-fs02_t.dump"))
    assert len(dumps) == 1, dumps
    assert dumps[0].parent != tmp_path
    if os.name == "posix":
        assert (dumps[0].parent.stat().st_mode & 0o077) == 0
