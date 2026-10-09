"""Tuner discovery reads files; it must not run the package's benchmark scripts.

``mlframe-tune-kernels`` used to find tuners by importing every module of the package. Benchmark scripts change the process at import (``CUDA_VISIBLE_DEVICES=''``,
``NUMBA_DISABLE_CUDA=1``, ``sys.modules["cupy"] = None`` ...), so the process that was meant to measure the GPU saw none: its hardware fingerprint read "no-gpu", GPU sweeps built no
variants and persisted zero regions. These tests build a small package that contains both kinds of module.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

from mlframe.system.kernel_tuning_cache import _discovery

_REGISTERING = 'def kernel_tuner(**kwargs):\n    return kwargs\n\n\nSPEC = kernel_tuner(kernel_name="{name}")\nimport sys\nSEEN_CUPY = sys.modules.get("cupy")\n'


@pytest.fixture
def package(tmp_path, monkeypatch):
    """`ktcpkg`: two registering modules (one in a subpackage, one that raises), a benchmark script that poisons the environment and cupy, and a module that only MENTIONS kernel_tuner."""
    pkg = tmp_path / "ktcpkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "sub" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "reg_a.py").write_text(_REGISTERING.format(name="a"), encoding="utf-8")
    (pkg / "sub" / "reg_b.py").write_text(_REGISTERING.format(name="b"), encoding="utf-8")
    (pkg / "reg_c_raises.py").write_text('def kernel_tuner(**kw):\n    return kw\n\n\nSPEC = kernel_tuner(kernel_name="c")\nraise RuntimeError("optional extra missing")\n', encoding="utf-8")
    (pkg / "bench_poison.py").write_text('import os, sys\nos.environ["CUDA_VISIBLE_DEVICES"] = ""\nsys.modules["cupy"] = None\n', encoding="utf-8")
    (pkg / "mentions_only.py").write_text('"""Docs: a module registers with ``kernel_tuner(...)`` at import."""\n# kernel_tuner(x) is explained here\nVALUE = 1\n', encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "ktcpkg"
    for name in [m for m in sys.modules if m == "ktcpkg" or m.startswith("ktcpkg.")]:
        del sys.modules[name]


def test_only_modules_that_call_kernel_tuner_are_selected(package):
    """Found by reading the syntax tree: the registering modules, including one in a subpackage; not the benchmark script and not a module that only mentions the name."""
    assert sorted(_discovery.registering_modules(package)) == ["ktcpkg.reg_a", "ktcpkg.reg_c_raises", "ktcpkg.sub.reg_b"]


def test_a_benchmark_script_is_never_run_so_the_process_keeps_its_gpu(package, monkeypatch):
    """The poisoning script is not imported: CUDA_VISIBLE_DEVICES stays unset and cupy stays usable."""
    fake_cupy = types.ModuleType("cupy")
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _discovery.discover_specs(package)
    assert "ktcpkg.bench_poison" not in sys.modules
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert sys.modules["cupy"] is fake_cupy
    assert sys.modules["ktcpkg.reg_a"].SEEN_CUPY is fake_cupy


def test_a_registering_module_that_fails_is_reported_and_does_not_stop_the_others(package, monkeypatch, caplog):
    """A broken registering module is skipped with a WARNING naming it (its kernels will not be tuned); the others are still imported."""
    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    with caplog.at_level("WARNING", logger=_discovery.logger.name):
        _discovery.discover_specs(package)
    assert "ktcpkg.reg_a" in sys.modules and "ktcpkg.sub.reg_b" in sys.modules
    assert any("ktcpkg.reg_c_raises" in r.getMessage() and "optional extra missing" in r.getMessage() for r in caplog.records)


def test_cupy_is_protected_in_a_fresh_process_too(package, tmp_path, monkeypatch):
    """In a new process cupy is not in sys.modules yet. It is imported when discovery starts, so a registering module that blocks it cannot leave the process without."""
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    (stub_dir / "cupy.py").write_text("IS_STUB = True" + chr(10), encoding="utf-8")
    (tmp_path / "ktcpkg" / "reg_poison.py").write_text('import sys\n\n\ndef kernel_tuner(**kw):\n    return kw\n\n\nSPEC = kernel_tuner(kernel_name="p")\nsys.modules["cupy"] = None\n', encoding="utf-8")
    monkeypatch.syspath_prepend(str(stub_dir))
    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    _discovery.discover_specs(package)
    assert getattr(sys.modules["cupy"], "IS_STUB", False) is True
    sys.modules.pop("cupy", None)


def test_a_cupy_that_was_blocked_before_discovery_stays_blocked(package, monkeypatch):
    """An explicit CPU-only choice made by the caller (cupy already None) is not undone."""
    monkeypatch.setitem(sys.modules, "cupy", None)
    _discovery.discover_specs(package)
    assert sys.modules["cupy"] is None


def test_the_real_package_is_found_by_reading_not_by_importing_everything():
    """On mlframe itself: the known registering module is found, and nothing is imported to find them."""
    before = set(sys.modules)
    found = set(_discovery.registering_modules("mlframe"))
    assert "mlframe.feature_selection.filters.batch_pair_mi_gpu" in found
    assert set(sys.modules) == before, "finding the modules must not import any"


def test_the_cli_discovers_through_the_guarded_walk():
    """main() resolves its specs through discover_specs, not through the unguarded pyutilz walk."""
    from mlframe.system import kernel_tuning_cache as cli

    assert cli.discover_specs is _discovery.discover_specs
    assert not hasattr(cli, "discover_tuners")
