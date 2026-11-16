"""Regression: the per-paramset cv_results dump must be written atomically."""
import os

import joblib
import pytest

from mlframe.estimators import pipelines


def test_failed_dump_keeps_previous_file(tmp_path, monkeypatch):
    """A failed dump leaves the previous file byte-identical."""
    out = str(tmp_path)
    calls = {"n": 0}

    def cv_func(X, Y, title, **kw):
        """Return a fresh metric dict on every call."""
        calls["n"] += 1
        return {"m": calls["n"]}

    monkeypatch.setattr(pipelines, "compare_cv_metrics", lambda **kw: None)
    pipelines.optimize_pipeline_by_gridsearch(None, None, "t", cv_func, output_dir=out)
    dumps = [f for f in os.listdir(out) if f.endswith(".dump")]
    assert len(dumps) == 1
    path = os.path.join(out, dumps[0])
    before = open(path, "rb").read()

    real = joblib.dump

    def boom(obj, fh, *a, **k):
        """Write a partial payload then fail, as a full disk would."""
        fh.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(pipelines.joblib, "dump", boom)
    with pytest.raises(OSError):
        pipelines.optimize_pipeline_by_gridsearch(None, None, "t", cv_func, output_dir=out)
    monkeypatch.setattr(pipelines.joblib, "dump", real)
    assert open(path, "rb").read() == before
