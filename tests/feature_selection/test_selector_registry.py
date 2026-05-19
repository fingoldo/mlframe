"""Sensor tests for the unified FeatureSelectorSpec registry (A-Arch-002).

Verifies MRMR / RFECV / BorutaShap are registered, instantiation routes via the registry,
and unknown names raise KeyError. The registry is the single edit point for adding new
selectors so adding sklearn RFE / boruta-py later is one class registration.
"""
from __future__ import annotations

import pytest

from mlframe.feature_selection import registry as fs_registry


def test_builtin_registrations_present():
    available = fs_registry.available()
    assert "MRMR" in available
    assert "RFECV" in available
    assert "BorutaShap" in available


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        fs_registry.get("definitely_not_a_real_selector")


def test_register_then_get_roundtrip():
    """Custom registrations are gettable. Uses _SimpleSpec internally."""
    from mlframe.feature_selection.registry import _SimpleSpec

    sentinel = object()

    def _make(**_kw):
        return sentinel

    spec = _SimpleSpec(name="__test_spec__", instantiate=_make)
    fs_registry.register(spec)
    try:
        got = fs_registry.get("__test_spec__")
        assert got.name == "__test_spec__"
        assert got.instantiate() is sentinel
    finally:
        # cleanup
        fs_registry._REGISTRY.pop("__test_spec__", None)


def test_mrmr_spec_instantiate_returns_mrmr():
    """The MRMR spec actually builds an MRMR instance (not just a sentinel)."""
    spec = fs_registry.get("MRMR")
    inst = spec.instantiate()
    from mlframe.feature_selection.filters import MRMR
    assert isinstance(inst, MRMR)


def test_register_without_name_raises():
    from mlframe.feature_selection.registry import _SimpleSpec
    spec = _SimpleSpec(name="", instantiate=lambda: None)
    with pytest.raises(ValueError):
        fs_registry.register(spec)
