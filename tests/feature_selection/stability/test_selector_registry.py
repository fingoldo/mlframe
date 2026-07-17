"""Sensor tests for the unified FeatureSelectorSpec registry (A-Arch-002).

Verifies MRMR / RFECV / BorutaShap are registered, instantiation routes via the registry,
and unknown names raise KeyError. The registry is the single edit point for adding new
selectors so adding sklearn RFE / boruta-py later is one class registration.
"""

from __future__ import annotations

import pytest

from mlframe.feature_selection import registry as fs_registry


def test_builtin_registrations_present():
    """Builtin registrations present."""
    available = fs_registry.available()
    assert "MRMR" in available
    assert "RFECV" in available
    assert "BorutaShap" in available
    assert "ShapProxiedFS" in available  # was registered but had no presence sensor


def test_every_registered_selector_has_contract_spec():
    """Tripwire: every production-registered selector MUST have a contract spec in
    tests/feature_selection/_selector_factories.SELECTOR_SPECS, so registration
    implies shared-contract coverage (a 5th registry entry without a spec fails
    here instead of silently escaping the cross-selector battery). Closes the
    "two hand-rolled factory lists can drift" gap (shared_lift-02)."""
    from tests.feature_selection._selector_factories import SELECTOR_SPECS

    missing = set(fs_registry.available()) - set(SELECTOR_SPECS)
    assert not missing, (
        f"registry selectors without a contract spec: {sorted(missing)} -- add a "
        f"SelectorSpec in _selector_factories.py so registration implies contract coverage"
    )


def test_rfecv_and_shap_proxied_specs_instantiate_real_types():
    """The RFECV / ShapProxiedFS specs build the real classes via the registry
    (only MRMR had an instantiate-type sensor before)."""
    from mlframe.feature_selection.wrappers import RFECV
    from sklearn.linear_model import LogisticRegression

    rfecv = fs_registry.get("RFECV").instantiate(estimator=LogisticRegression(max_iter=50), cluster_reduce=False)
    assert isinstance(rfecv, RFECV)

    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    sp = fs_registry.get("ShapProxiedFS").instantiate()
    assert isinstance(sp, ShapProxiedFS)


def test_get_unknown_raises():
    """Get unknown raises."""
    with pytest.raises(KeyError):
        fs_registry.get("definitely_not_a_real_selector")


def test_register_then_get_roundtrip():
    """Custom registrations are gettable. Uses _SimpleSpec internally."""
    from mlframe.feature_selection.registry import _SimpleSpec

    sentinel = object()

    def _make(**_kw):
        """Helper that make."""
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
    """Register without name raises."""
    from mlframe.feature_selection.registry import _SimpleSpec

    spec = _SimpleSpec(name="", instantiate=lambda: None)
    with pytest.raises(ValueError):
        fs_registry.register(spec)


def test_every_builtin_spec_satisfies_protocol():
    """Each registered spec MUST be a runtime instance of FeatureSelectorSpec. Catches a future drift where a class is
    registered with a non-matching attribute shape (e.g. missing instantiate, or instantiate is a non-callable)."""
    from mlframe.feature_selection.registry import FeatureSelectorSpec

    for name in fs_registry.available():
        spec = fs_registry.get(name)
        assert isinstance(spec, FeatureSelectorSpec), f"registered spec {name!r} fails FeatureSelectorSpec protocol"
        assert callable(spec.instantiate), f"{name!r}.instantiate is not callable"
        assert isinstance(spec.name, str) and spec.name, f"{name!r} has empty or non-str name"
        # report_extract is optional, but if present MUST be callable
        if spec.report_extract is not None:
            assert callable(spec.report_extract), f"{name!r}.report_extract present but not callable"


def test_register_rejects_non_protocol_object():
    """A bare object missing the required attribute set must not slip through ``register``. ``register`` already
    validates name; this pins that ``getattr`` access on a non-conforming type fails the contract rather than silently
    storing a broken spec."""

    class _NoInstantiate:
        """Groups tests covering NoInstantiate."""
        name = "broken_spec"

    with pytest.raises(AttributeError):
        # Accessing .instantiate must blow up; if a future register() refactor uses getattr with a default and
        # silently stores None, this test surfaces it.
        fs_registry.register(_NoInstantiate())  # type: ignore[arg-type]
