"""A native extension that crashes the interpreter on import must cost a fallback, not the training run.

hnswlib's DLL faulted on load (0xC0000005) on a conda Windows box whose MSVC runtime next to python.exe was older than
the wheel's. That is not an ImportError: the process died mid-suite. ``native_module_importable`` probes the import in
a child interpreter first; these tests crash a real child to prove the parent survives and reports it.
"""

from __future__ import annotations

import logging
import sys
import textwrap

import numpy as np
import pytest

import mlframe.utils.native_import_probe as probe


@pytest.fixture
def fresh_probe(monkeypatch, tmp_path):
    """Empty verdict cache, a module dir on the child's PYTHONPATH, and no inherited verdicts."""
    monkeypatch.setattr(probe, "_verdicts", {})
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    return tmp_path


def _write(dirpath, name, body):
    (dirpath / f"{name}.py").write_text(textwrap.dedent(body), encoding="ascii")


def test_module_that_crashes_on_import_is_reported_unusable_and_parent_survives(fresh_probe, monkeypatch, caplog):
    # A real native crash during import: no Python exception reaches the caller, the interpreter dies.
    _write(fresh_probe, "mlframe_probe_crasher", "import faulthandler\nfaulthandler._sigsegv()\n")
    monkeypatch.delenv(probe._env_key("mlframe_probe_crasher"), raising=False)
    with caplog.at_level(logging.WARNING, logger=probe.__name__):
        assert probe.native_module_importable("mlframe_probe_crasher") is False
    assert "unusable" in caplog.text and "crashed" in caplog.text
    # Workers spawned later inherit the verdict instead of crashing a probe of their own.
    assert probe.os.environ[probe._env_key("mlframe_probe_crasher")] == "0"


def test_importable_module_is_usable_and_cached(fresh_probe, monkeypatch):
    _write(fresh_probe, "mlframe_probe_fine", "VALUE = 1\n")
    monkeypatch.delenv(probe._env_key("mlframe_probe_fine"), raising=False)
    assert probe.native_module_importable("mlframe_probe_fine") is True

    def _no_second_probe(*a, **k):
        raise AssertionError("the verdict is cached per process")

    monkeypatch.setattr(probe.subprocess, "run", _no_second_probe)
    assert probe.native_module_importable("mlframe_probe_fine") is True


def test_missing_module_is_unusable_without_a_warning(fresh_probe, monkeypatch, caplog):
    monkeypatch.delenv(probe._env_key("mlframe_probe_missing_xyz"), raising=False)
    with caplog.at_level(logging.WARNING, logger=probe.__name__):
        assert probe.native_module_importable("mlframe_probe_missing_xyz") is False
    assert "unusable" not in caplog.text


def test_already_imported_module_is_never_probed(fresh_probe, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("an imported module needs no probe")

    import json  # noqa: F401

    monkeypatch.setattr(probe.subprocess, "run", _boom)
    assert "json" in sys.modules and probe.native_module_importable("json") is True


def test_inherited_negative_verdict_is_trusted(fresh_probe, monkeypatch):
    monkeypatch.setenv(probe._env_key("mlframe_probe_inherited"), "0")

    def _boom(*a, **k):
        raise AssertionError("an inherited verdict needs no probe")

    monkeypatch.setattr(probe.subprocess, "run", _boom)
    assert probe.native_module_importable("mlframe_probe_inherited") is False


def test_knn_search_falls_back_to_exact_sklearn_when_hnswlib_is_unusable(monkeypatch):
    import mlframe.feature_engineering.transformer._knn_helper as kh

    monkeypatch.setattr(kh, "_HNSW_AVAILABLE", None)
    monkeypatch.setattr(probe, "native_module_importable", lambda name: False)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    dists, ids = kh.knn_search(X, X[:10], k=3, prefer_hnsw_at_n=1)
    assert kh._HNSW_AVAILABLE is False
    assert np.array_equal(ids[:, 0], np.arange(10)) and np.allclose(dists[:, 0], 0.0)
