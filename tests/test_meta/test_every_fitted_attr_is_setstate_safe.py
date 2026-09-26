"""A fitted attribute added today must still be present on an estimator unpickled from an older release.

``__setstate__`` backfills missing attributes from two sources. Constructor parameters are covered structurally, because the second pass reads
``_ctor_defaults()``. Trailing-underscore FITTED attributes are covered only by the hand-maintained roster in ``_mrmr_setstate_defaults``, so a
new one added without a roster entry is simply absent from an old pickle. The roster's own comment records an instance where exactly that
happened and silently collapsed multi-batch history, so the hazard is demonstrated rather than hypothetical.

Every consumer of the currently-uncovered attributes reads them through ``getattr(..., default)``, which is why nothing is broken today. That
is a property of the reading code, not of the attribute, and it is not something the next person adding an attribute will know to preserve.
This gate pins the current set: a NEW fitted attribute has to be given a roster entry, or be added to the baseline deliberately, which is the
moment to check that every read of it carries a default.
"""

from __future__ import annotations

from tests.test_meta._scan_guard import assert_scanned_enough

import ast
import pathlib
import sys

import orjson

_TESTS_META = pathlib.Path(__file__).resolve().parent
_SRC = _TESTS_META.parents[1] / "src"
_BASELINE = _TESTS_META / "_fitted_attr_setstate_baseline.json"

# Where the estimator's fitted attributes are assigned: the class itself and the fit implementation it delegates to.
_PACKAGES = (
    "mlframe/feature_selection/filters/mrmr",
    "mlframe/feature_selection/filters/_mrmr_fit_impl",
)


def _assigned_fitted_attrs() -> set:
    """Every ``self.<name>_ = ...`` assignment across the estimator's own packages, by attribute name."""
    names: set = set()
    for package in _PACKAGES:
        _files = sorted((_SRC / package).rglob("*.py"))
        assert_scanned_enough(len(_files), str(_SRC / package), minimum=1)
        for path in _files:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                    targets = [node.target]
                else:
                    continue
                for target in targets:
                    if not isinstance(target, ast.Attribute) or not isinstance(target.value, ast.Name):
                        continue
                    if target.value.id != "self":
                        continue
                    # sklearn's convention: one trailing underscore marks a fitted attribute; two marks a dunder.
                    if target.attr.endswith("_") and not target.attr.endswith("__"):
                        names.add(target.attr)
    return names


def _roster_keys() -> set:
    """The hand-maintained set of attributes ``__setstate__`` backfills."""
    sys.path.insert(0, str(_SRC))
    from mlframe.feature_selection.filters.mrmr._mrmr_setstate_defaults import _SETSTATE_LEGACY_DEFAULTS

    return set(_SETSTATE_LEGACY_DEFAULTS)


def _uncovered() -> set:
    """Fitted attributes assigned by a fit but not backfilled when an old pickle is loaded."""
    return _assigned_fitted_attrs() - _roster_keys()


_REFRESH_FLAG = "--refresh-fitted-attr-setstate-baseline"


def test_no_new_fitted_attribute_escapes_the_setstate_backfill():
    """The uncovered set may shrink freely; growing it requires a deliberate decision, recorded in the baseline."""
    if _REFRESH_FLAG in sys.argv:
        regenerate_baseline(_BASELINE)
        return
    baseline = set(orjson.loads(_BASELINE.read_bytes()))
    new = sorted(_uncovered() - baseline)
    assert not new, (
        "fitted attribute(s) assigned by a fit but not backfilled by __setstate__, so they are absent from an estimator unpickled from an "
        "older release: " + ", ".join(new) + ". Add each to _SETSTATE_LEGACY_DEFAULTS with the default a fit would have produced, or, if "
        "every read of it already passes a default to getattr, add it to " + _BASELINE.name + "."
    )


def test_the_baseline_does_not_list_attributes_that_are_now_covered():
    """A baselined attribute that has since been given a roster entry must be removed, so the list stays a real inventory."""
    stale = sorted(set(orjson.loads(_BASELINE.read_bytes())) & _roster_keys())
    assert not stale, f"baselined attributes that are now in the roster and should be dropped from {_BASELINE.name}: {stale}"


def test_the_detector_sees_a_fitted_attribute_assignment():
    """Teeth-check: the collector must actually recognise the assignment form it exists to find."""
    attrs = _assigned_fitted_attrs()
    assert "support_" in attrs, "the collector no longer sees even support_, so this gate would pass vacuously"
    assert "fit" not in attrs and not any(a.endswith("__") for a in attrs), "the collector is picking up non-fitted names"


def regenerate_baseline(path: pathlib.Path) -> None:
    """Rewrite the baseline with the currently-uncovered fitted attributes (invoked by ``regen_baselines.py``)."""
    path.write_bytes(orjson.dumps(sorted(_uncovered()), option=orjson.OPT_INDENT_2) + b"\n")
