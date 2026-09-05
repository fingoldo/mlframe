"""A transient device fault must not pin a subsystem to its CPU path for the rest of the process.

Three probes cached a broad `except Exception` as `False` for the process lifetime, reporting it only at
debug. The exceptions they actually see are not facts about the machine -- another process holding the
card, a CUDA OOM at probe time, a driver reset, a fault raised out of `getDeviceCount` under contention --
so the first one to land decided the whole run. `ImportError` is the one genuine absence and stays cached,
because cupy not being installed is a fact about the install rather than a moment.

What each latch cost: `_core_auc_brier` gives back the ~10% end-to-end win its own A/B header records at
200k (CPU 9.37/8.29s against GPU 7.98/7.89s) for every later metric call; `cluster_su_gpu_available` puts
the whole ShapProxiedFS cluster-SU pair loop on the CPU for every later fit.

The MRMR fit-entry re-arm is the same class one level up: it re-armed three of the five process-global GPU
circuit breakers in the package, leaving `_ksg` and `_permutation_null_resident` with exactly the
process-lifetime stickiness its own docstring describes as the defect. Both reset functions already
existed with no production caller.
"""

from __future__ import annotations

import builtins
import logging
import types

import pytest


def _cupy_raising(monkeypatch, exc: BaseException | None):
    """Make `import cupy` raise `exc`, or succeed with a stub reporting one working device."""
    real_import = builtins.__import__

    class _Arr:
        """The tiny array a probe allocates and reduces."""

        def sum(self):
            """Support the allocation round-trip."""
            return self

        def get(self):
            """Return a host scalar, completing the round-trip."""
            return 0.0

    stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(runtime=types.SimpleNamespace(getDeviceCount=lambda: 1)),
        zeros=lambda *a, **k: _Arr(),
        float32="float32",
    )

    def _fake_import(name, *args, **kwargs):
        """Intercept only cupy; everything else imports normally."""
        if name == "cupy":
            if exc is not None:
                raise exc
            return stub
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


class TestMetricsArgsortProbe:
    """`_core_auc_brier._gpu_argsort_available`."""

    @pytest.fixture(autouse=True)
    def _clean(self):
        """Start and finish unprobed."""
        from mlframe.metrics import _core_auc_brier as m

        m.reset_gpu_argsort_probe()
        yield
        m.reset_gpu_argsort_probe()

    def test_a_transient_failure_is_not_cached(self, monkeypatch, caplog):
        """A device-count fault must leave the cache unset so a later call can succeed."""
        from mlframe.metrics import _core_auc_brier as m

        _cupy_raising(monkeypatch, RuntimeError("device busy"))
        with caplog.at_level(logging.WARNING, logger=m.logger.name):
            assert m._gpu_argsort_available() is False
        assert any("transiently" in r.getMessage() for r in caplog.records), "a transient probe failure was not reported above debug"
        assert m._GPU_ARGSORT_AVAILABLE is None, "a transient failure pinned the process to the CPU argsort"

        _cupy_raising(monkeypatch, None)
        assert m._gpu_argsort_available() is True, "the probe never re-ran after a transient failure"

    def test_an_absent_cupy_is_cached(self, monkeypatch):
        """ImportError is a fact about the install, so caching it is correct."""
        from mlframe.metrics import _core_auc_brier as m

        _cupy_raising(monkeypatch, ImportError("no module named cupy"))
        assert m._gpu_argsort_available() is False
        assert m._GPU_ARGSORT_AVAILABLE is False

    def test_a_successful_probe_is_cached(self, monkeypatch):
        """Re-probing every call would be its own regression."""
        from mlframe.metrics import _core_auc_brier as m

        _cupy_raising(monkeypatch, None)
        assert m._gpu_argsort_available() is True
        assert m._GPU_ARGSORT_AVAILABLE is True


class TestClusterSuProbe:
    """`_shap_proxy_cluster_su.cluster_su_gpu_available`."""

    @pytest.fixture(autouse=True)
    def _clean(self):
        """Start and finish unprobed."""
        from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_cluster_su as m

        m.reset_cluster_su_gpu_probe()
        yield
        m.reset_cluster_su_gpu_probe()

    def test_a_failing_allocation_is_not_cached(self, monkeypatch, caplog):
        """The tiny alloc faults under VRAM contention; that is a moment, not a missing device."""
        from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_cluster_su as m

        _cupy_raising(monkeypatch, RuntimeError("out of memory"))
        with caplog.at_level(logging.WARNING, logger=m.logger.name):
            assert m.cluster_su_gpu_available() is False
        assert any("transiently" in r.getMessage() for r in caplog.records)
        assert m._GPU_AVAILABLE_CACHE is None, "a transient failure disabled cluster-SU GPU for the process"

        _cupy_raising(monkeypatch, None)
        assert m.cluster_su_gpu_available() is True

    def test_an_absent_cupy_is_cached(self, monkeypatch):
        """ImportError still latches."""
        from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_cluster_su as m

        _cupy_raising(monkeypatch, ImportError("no module named cupy"))
        assert m.cluster_su_gpu_available() is False
        assert m._GPU_AVAILABLE_CACHE is False


def test_the_fit_entry_rearm_covers_every_circuit_breaker_in_the_package():
    """Every reset_*_gpu_circuit_breaker the package defines must be called by the fit-entry re-arm.

    Two existed with no production caller at all, so the flags they clear kept exactly the
    process-lifetime stickiness the re-arm exists to bound to one fit.
    """
    import ast
    import pathlib

    import mlframe.feature_selection.filters as filters_pkg

    root = pathlib.Path(filters_pkg.__file__).parent
    defined = set()
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_bytes().decode("utf-8"))):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("reset_") and node.name.endswith("_gpu_circuit_breaker"):
                defined.add(node.name)

    helpers = root / "mrmr" / "_mrmr_class_fit_helpers.py"
    tree = ast.parse(helpers.read_bytes().decode("utf-8"))
    rearm = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_rearm_gpu_circuit_breakers")
    called = {n.func.id for n in ast.walk(rearm) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}

    assert defined, "no circuit-breaker resets found; this test has lost its subject"
    assert defined <= called, f"defined but never re-armed at fit entry: {sorted(defined - called)}"
