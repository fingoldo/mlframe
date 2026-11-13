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

_TESTS_META = pathlib.Path(__file__).resolve().parent
_SRC = _TESTS_META.parents[1] / "src"

# Fitted attributes a fit assigns that the roster deliberately does not backfill, each with the reason. A backfilled default would be wrong for every one: these are the
# markers of "this estimator was fitted", present in every pickle of a fitted estimator from any release, so an estimator that lacks them was never fitted.
_NOT_BACKFILLED = {
    "feature_names_in_": "sklearn's fitted-input-names marker, set by every fit since the first release; an unfitted estimator must not gain it, and transform reads it through getattr(..., None)",
    "n_features_in_": "sklearn's fitted-input-width marker, set by every fit since the first release; a backfilled value would make an unfitted estimator pass check_is_fitted",
    "support_": "the selected column indices, the defining product of a fit and present in every fitted pickle; a backfilled empty array would silently select nothing",
    "support_nonlinear_": "an alias of support_ set in the same step and re-synced under hasattr; no reader exists outside that re-sync, so a pickle without it needs no value",
}

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


def test_no_new_fitted_attribute_escapes_the_setstate_backfill():
    """The uncovered set may shrink freely; growing it requires a deliberate decision, recorded in _NOT_BACKFILLED."""
    new = sorted(_uncovered() - set(_NOT_BACKFILLED))
    assert not new, (
        "fitted attribute(s) assigned by a fit but not backfilled by __setstate__, so they are absent from an estimator unpickled from an "
        "older release: " + ", ".join(new) + ". Add each to _SETSTATE_LEGACY_DEFAULTS with the default a fit would have produced, or, if "
        "it is a fitted-state marker no default can stand in for, add it to _NOT_BACKFILLED with the reason."
    )


def test_the_baseline_does_not_list_attributes_that_are_now_covered():
    """A baselined attribute that has since been given a roster entry must be removed, so the list stays a real inventory."""
    stale = sorted(set(_NOT_BACKFILLED) & _roster_keys())
    assert not stale, f"attributes now in the roster that should be dropped from _NOT_BACKFILLED: {stale}"


def test_the_detector_sees_a_fitted_attribute_assignment():
    """Teeth-check: the collector must actually recognise the assignment form it exists to find."""
    attrs = _assigned_fitted_attrs()
    assert "support_" in attrs, "the collector no longer sees even support_, so this gate would pass vacuously"
    assert "fit" not in attrs and not any(a.endswith("__") for a in attrs), "the collector is picking up non-fitted names"


def test_every_not_backfilled_attribute_states_its_reason_and_is_assigned():
    """Each exempt attribute carries a reason and is still assigned by a fit, so the list cannot keep entries nothing sets any more."""
    assert all(len(why) > 30 for why in _NOT_BACKFILLED.values())
    assert set(_NOT_BACKFILLED) <= _assigned_fitted_attrs()
