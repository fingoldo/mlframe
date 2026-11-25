"""Cluster 13 regression: the wide-data-scaling progress log honours MLFRAME_BENCH_OUT and has no hard-coded D: path (audit 2026-11-20, finding 13-F32).

The module is reloaded under a temporary override to check where PROG points, and the default is checked not to name D:.
"""

import importlib
import os

from mlframe.feature_selection._benchmarks.wide_data_scaling import _progress_shared as shared


def test_progress_path_follows_the_env_override(monkeypatch, tmp_path):
    """With MLFRAME_BENCH_OUT set, the progress log is written under that folder (13-F32)."""
    monkeypatch.setenv("MLFRAME_BENCH_OUT", str(tmp_path))
    reloaded = importlib.reload(shared)
    try:
        assert reloaded.PROG.startswith(str(tmp_path))
        assert reloaded.PROG.endswith(os.path.join("synergy_scale_bench", "progress.txt"))
    finally:
        monkeypatch.delenv("MLFRAME_BENCH_OUT", raising=False)
        importlib.reload(shared)


def test_default_path_has_no_d_drive():
    """Without the override the default does not name a D: drive (13-F32)."""
    assert "D:" not in importlib.reload(shared).PROG
