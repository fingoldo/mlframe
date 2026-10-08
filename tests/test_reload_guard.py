"""``reload_guard`` puts a reloaded module's original classes back, so later pickling stays by reference."""

from __future__ import annotations

import importlib
import sys

import dill  # nosec B403 - pickling an object this test builds

from tests._reload_guard import reload_guard


def _write_probe(tmp_path, monkeypatch, name: str):
    """A throwaway module defining one class, importable as ``name``."""
    (tmp_path / f"{name}.py").write_text('class Probe:\n    """A class whose identity a reload changes."""\n', encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(name, None)
    module = importlib.import_module(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_a_reload_inside_the_guard_is_undone_on_exit(tmp_path, monkeypatch) -> None:
    """The class object is new inside the guard and the original again after it."""
    module = _write_probe(tmp_path, monkeypatch, "reload_guard_probe_a")
    original = module.Probe
    with reload_guard(prefixes=("reload_guard_probe_a",)) as saved:
        importlib.reload(module)
        assert module.Probe is not original
        assert "reload_guard_probe_a" in saved
    assert module.Probe is original


def test_reloading_twice_still_restores_the_first_bindings(tmp_path, monkeypatch) -> None:
    """A second reload (the usual fake 'restore') must not become the state that survives."""
    module = _write_probe(tmp_path, monkeypatch, "reload_guard_probe_b")
    original = module.Probe
    with reload_guard(prefixes=("reload_guard_probe_b",)):
        importlib.reload(module)
        importlib.reload(module)
    assert module.Probe is original


def test_the_class_pickles_by_reference_after_the_guard(tmp_path, monkeypatch) -> None:
    """After the guard an instance pickles without the by-value type constructor the restricted loader refuses."""
    module = _write_probe(tmp_path, monkeypatch, "reload_guard_probe_c")
    with reload_guard(prefixes=("reload_guard_probe_c",)):
        importlib.reload(module)
    assert b"_create_type" not in dill.dumps(module.Probe())


def test_modules_outside_the_prefixes_are_left_alone(tmp_path, monkeypatch) -> None:
    """A module that is not under a guarded prefix keeps its reloaded state, and the real reload is back on exit."""
    module = _write_probe(tmp_path, monkeypatch, "reload_guard_probe_d")
    original = module.Probe
    real = importlib.reload
    with reload_guard(prefixes=("mlframe",)) as saved:
        importlib.reload(module)
    assert saved == {}
    assert module.Probe is not original
    assert importlib.reload is real
