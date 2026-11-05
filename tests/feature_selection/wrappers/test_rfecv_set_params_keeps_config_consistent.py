"""``RFECV.set_params`` and a grouped config object agree, so ``clone`` works and a passed config takes effect.

A config object (``SearchConfig``) holds the same settings as the flat constructor arguments. Changing a flat argument with ``set_params``
left the config's old value in place, and ``clone`` then re-applied it over the new value and failed sklearn's parameter check, which breaks
any ``GridSearchCV`` over such a parameter. Passing a config object to ``set_params`` also did nothing, because the flat attributes were
read only at construction.
"""

from __future__ import annotations

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from mlframe.feature_selection.wrappers.rfecv import RFECV
from mlframe.feature_selection.wrappers.rfecv._configs import SearchConfig


def test_clone_after_setting_a_flat_field_keeps_the_new_value():
    """A flat field changed with set_params survives clone, rather than the config's older value winning."""
    r = RFECV(estimator=LogisticRegression(), search_config=SearchConfig(max_refits=20))
    r.set_params(max_refits=5)
    cloned = clone(r)
    assert cloned.max_refits == 5


def test_set_params_with_a_config_object_updates_the_flat_attributes():
    """Passing a SearchConfig to set_params applies its explicitly-set fields to the estimator."""
    r = RFECV(estimator=LogisticRegression(), search_config=SearchConfig(max_refits=20))
    r.set_params(search_config=SearchConfig(max_refits=3))
    assert r.max_refits == 3
