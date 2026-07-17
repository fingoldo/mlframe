"""Regression: native-cat / CB detection that drives the CB ordinal-encoding auto-flip must be instance/alias equivalent.

Before the fix, ``_phase_fit_pipeline`` detected CB / all-native-cat via ``str(m).lower()`` against a string set. A model passed as an
estimator INSTANCE (a first-class input -- ``get_strategy`` / ``_resolve_model_spec`` / ``strategy_by_model`` all accept instances)
stringifies to ``"catboostclassifier()"``, which is absent from the set, so an instance-passed CatBoost was mis-classed non-CB /
non-native and the CB native-categorical auto-flip (skip ordinal encoding, keep CB's native cat handling) silently never fired.
"""

import pytest

from mlframe.training.core._phase_helpers_fit_pipeline import _detect_native_cat_models
from mlframe.training.strategies import get_strategy


def _strats(models):
    """Strats."""
    return [get_strategy(m) for m in models]


def test_cb_instance_detected_same_as_cb_alias():
    """Cb instance detected same as cb alias."""
    cb = pytest.importorskip("catboost")
    inst = cb.CatBoostClassifier()

    has_cb_alias, all_native_alias = _detect_native_cat_models(_strats(["cb"]))
    has_cb_inst, all_native_inst = _detect_native_cat_models(_strats([inst]))

    assert has_cb_alias is True and all_native_alias is True
    # The bug: pre-fix str-based detection returned (False, False) for the instance.
    assert has_cb_inst is True, "CB passed as an instance must be detected as CB (was missed by str-name check)"
    assert all_native_inst is True, "a lone CB instance is all-native-cat (was missed by str-name check)"


def test_mixed_cb_instance_plus_linear_is_not_all_native():
    """Mixed cb instance plus linear is not all native."""
    cb = pytest.importorskip("catboost")
    has_cb, all_native = _detect_native_cat_models(_strats([cb.CatBoostClassifier(), "linear"]))
    assert has_cb is True
    assert all_native is False, "linear needs an encoder, so the suite is not all-native-cat -- auto-flip must stay off"


def test_empty_models_detects_nothing():
    """Empty models detects nothing."""
    assert _detect_native_cat_models([]) == (False, False)
