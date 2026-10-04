"""Every fitted attribute a selector assigns must be backfilled by ``__setstate__`` or be declared core or fit-scratch.

Generalises ``test_every_fitted_attr_is_setstate_safe`` (MRMR) to RFECV, BorutaShap and ShapProxiedFS. An estimator pickled by an older release lacks
any attribute added since; methods that read it without a ``getattr`` default then raise ``AttributeError`` on first use. Each estimator keeps a
``_SETSTATE_LEGACY_DEFAULTS`` roster next to its ``__setstate__``; a new ``self.<name>_ = ...`` has to join the roster, or the module's
``CORE_FITTED_ATTRS`` (state whose absence must keep the estimator unfitted), or ``SCRATCH_FITTED_ATTRS`` / a leading underscore (re-derived each fit).
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

from tests.test_meta._scan_guard import assert_scanned_enough

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"

# (label, source package directory, roster module, estimator module, estimator class, core marker attribute)
_SELECTORS = (
    ("RFECV", "mlframe/feature_selection/wrappers/rfecv", "mlframe.feature_selection.wrappers.rfecv._state_compat",
     "mlframe.feature_selection.wrappers.rfecv", "RFECV", "support_"),
    ("BorutaShap", "mlframe/feature_selection/boruta_shap", "mlframe.feature_selection.boruta_shap._estimator_protocol",
     "mlframe.feature_selection.boruta_shap", "BorutaShap", "selected_features_"),
    ("ShapProxiedFS", "mlframe/feature_selection/shap_proxied_fs", "mlframe.feature_selection.shap_proxied_fs._state_compat",
     "mlframe.feature_selection.shap_proxied_fs", "ShapProxiedFS", "support_"),
)
_IDS = [s[0] for s in _SELECTORS]


def _assigned_fitted_attrs(package: str) -> set:
    """Every ``self.<name>_ = ...`` target in the package, excluding the double-underscore dunder form."""
    names: set = set()
    files = [p for p in sorted((_SRC / package).rglob("*.py")) if "_benchmarks" not in p.parts]
    assert_scanned_enough(len(files), str(_SRC / package), minimum=2)
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                    if target.attr.endswith("_") and not target.attr.endswith("__"):
                        names.add(target.attr)
    return names


@pytest.mark.parametrize("label,package,roster_module,est_module,est_class,marker", _SELECTORS, ids=_IDS)
def test_every_assigned_fitted_attribute_is_in_the_roster_or_declared_core_or_scratch(label, package, roster_module, est_module, est_class, marker):
    """A fitted attribute assigned by the selector's fit must be backfilled, core, or fit-scratch."""
    roster = importlib.import_module(roster_module)
    covered = set(roster._SETSTATE_LEGACY_DEFAULTS) | set(roster.CORE_FITTED_ATTRS) | set(getattr(roster, "SCRATCH_FITTED_ATTRS", ()))
    uncovered = sorted(a for a in _assigned_fitted_attrs(package) - covered if not a.startswith("_"))
    assert not uncovered, (
        f"{label}: fitted attribute(s) {uncovered} are assigned by fit but are neither in _SETSTATE_LEGACY_DEFAULTS (with the default an older "
        f"release's pickle should get), CORE_FITTED_ATTRS, nor SCRATCH_FITTED_ATTRS"
    )


@pytest.mark.parametrize("label,package,roster_module,est_module,est_class,marker", _SELECTORS, ids=_IDS)
def test_setstate_backfills_the_whole_roster_on_a_fitted_legacy_pickle(label, package, roster_module, est_module, est_class, marker):
    """Loading a fitted state that carries only core attributes yields every roster attribute with its declared default."""
    roster = importlib.import_module(roster_module)
    cls = getattr(importlib.import_module(est_module), est_class)
    est = cls.__new__(cls)
    est.__setstate__({marker: [True]})
    missing = [name for name in roster._SETSTATE_LEGACY_DEFAULTS if name not in est.__dict__]
    assert not missing, f"{label}.__setstate__ left {missing} unset on a fitted legacy pickle"
    assert roster._SETSTATE_LEGACY_DEFAULTS, "the legacy-default roster must not be empty"
    for name, default in roster._SETSTATE_LEGACY_DEFAULTS.items():
        assert est.__dict__[name] == default


@pytest.mark.parametrize("label,package,roster_module,est_module,est_class,marker", _SELECTORS, ids=_IDS)
def test_setstate_does_not_invent_fitted_state_for_an_unfitted_pickle(label, package, roster_module, est_module, est_class, marker):
    """A state without the core fitted marker is restored untouched, so the estimator keeps reporting itself unfitted."""
    roster = importlib.import_module(roster_module)
    cls = getattr(importlib.import_module(est_module), est_class)
    est = cls.__new__(cls)
    est.__setstate__({})
    assert not (set(roster._SETSTATE_LEGACY_DEFAULTS) & set(est.__dict__))


def test_the_roster_defaults_are_not_shared_between_instances():
    """Mutable defaults must be copied per instance, not aliased across every unpickled estimator."""
    from mlframe.feature_selection.wrappers.rfecv import RFECV

    a, b = RFECV.__new__(RFECV), RFECV.__new__(RFECV)
    a.__setstate__({"support_": [True]})
    b.__setstate__({"support_": [True]})
    a.estimators_.append("x")
    assert b.estimators_ == []
