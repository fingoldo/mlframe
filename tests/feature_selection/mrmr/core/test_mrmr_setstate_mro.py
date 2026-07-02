"""Regression test: MRMR.__setstate__ must resolve to MRMR's own implementation, not BaseEstimator's.

The mrmr mixin split moved most methods into sibling mixins that MRMR inherits AFTER BaseEstimator in the
MRO. That is correct for methods MRMR merely ADDS, but __setstate__ OVERRIDES BaseEstimator.__setstate__ -- so
if it ever gets relocated into a post-BaseEstimator mixin again, MRMR.__setstate__ silently resolves to
BaseEstimator's plain __dict__-update and the custom legacy-pickle default injection stops running. hasattr
and a fresh-instance pickle round-trip both hide that; these tests pin the real contract.
"""
from sklearn.base import BaseEstimator

from mlframe.feature_selection.filters.mrmr import MRMR


def _defining_class(cls, name):
    for klass in cls.__mro__:
        if name in klass.__dict__:
            return klass
    return None


def test_setstate_defined_on_mrmr_not_shadowed_by_baseestimator():
    assert "__setstate__" in BaseEstimator.__dict__, "precondition: BaseEstimator defines __setstate__"
    owner = _defining_class(MRMR, "__setstate__")
    assert owner is MRMR, (
        f"MRMR.__setstate__ resolves in {owner.__name__}, not MRMR -- a post-BaseEstimator mixin is shadowing "
        f"the custom legacy-pickle __setstate__"
    )


def test_setstate_injects_missing_legacy_ctor_default():
    """A legacy pickle missing a roster ctor param must have it backfilled by the custom __setstate__ (the
    behaviour BaseEstimator.__setstate__ would NOT provide)."""
    m = MRMR()
    param = "fe_wavelet_enable"  # a _SETSTATE_LEGACY_OVERRIDES roster key
    assert param in MRMR._SETSTATE_LEGACY_OVERRIDES
    state = dict(m.__dict__)
    state.pop(param, None)

    restored = MRMR.__new__(MRMR)
    restored.__setstate__(state)
    assert hasattr(restored, param), "custom __setstate__ did not inject the missing legacy ctor param"
    # roster comment: "fe_wavelet_enable -- legacy OFF; ctor ON" -> a legacy pickle must get the legacy False.
    assert getattr(restored, param) is False, "legacy-override value not applied on unpickle"
