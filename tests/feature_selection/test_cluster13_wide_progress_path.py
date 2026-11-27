"""Cluster 13 regression: the wide-data-scaling progress log honours MLFRAME_BENCH_OUT and has no hard-coded D: path (audit 2026-11-20, finding 13-F32).

A private copy of the module is imported under a temporary override to check where PROG points, and the default is checked not to name D:.
"""

import importlib.util
import os

from mlframe.feature_selection._benchmarks.wide_data_scaling import _progress_shared as shared


def _fresh_copy():
    """A private import of the progress module under another name, so the shared module object is never rebound."""
    spec = importlib.util.spec_from_file_location("_progress_shared_fresh_copy", shared.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_progress_path_follows_the_env_override(monkeypatch, tmp_path):
    """With MLFRAME_BENCH_OUT set, the progress log is written under that folder (13-F32)."""
    monkeypatch.setenv("MLFRAME_BENCH_OUT", str(tmp_path))
    fresh = _fresh_copy()
    assert fresh.PROG.startswith(str(tmp_path))
    assert fresh.PROG.endswith(os.path.join("synergy_scale_bench", "progress.txt"))


def test_default_path_has_no_d_drive(monkeypatch):
    """Without the override the default does not name a D: drive (13-F32)."""
    monkeypatch.delenv("MLFRAME_BENCH_OUT", raising=False)
    assert "D:" not in _fresh_copy().PROG
