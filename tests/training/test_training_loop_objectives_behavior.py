"""Behavioral coverage for _training_loop_objectives.py's two dispatch helpers.

test_training_loop_calibration_split.py already pins that these symbols are importable from
the right module after a monolith split, but never exercises their actual branching logic
(both dispatch on type(model).__name__, so they're testable with lightweight fake classes --
no real XGBoost/sklearn model needed).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training._training_loop_objectives import (
    _ensure_xgb_classification_objective,
    _maybe_wrap_for_2d_target,
)

pytestmark = pytest.mark.fast


class _FakeXGBClassifier:
    """Stand-in for xgboost.XGBClassifier -- only the get_params/set_params surface used by
    _ensure_xgb_classification_objective, name-matched via 'XGB' + 'Classifier'."""

    def __init__(self, **params):
        self._params = dict(params)
        self.set_params_calls: list[dict] = []

    def get_params(self):
        """Return the current params dict."""
        return dict(self._params)

    def set_params(self, **kwargs):
        """Record and apply a set_params call."""
        self.set_params_calls.append(kwargs)
        self._params.update(kwargs)


class _FakeNonXGBModel:
    """A model whose class name does not match 'XGB'+'Classifier' -- must be a pure no-op."""

    def __init__(self, **params):
        self._params = dict(params)
        self.set_params_calls: list[dict] = []

    def get_params(self):
        """Return the current params dict."""
        return dict(self._params)

    def set_params(self, **kwargs):
        """Record a set_params call (should never be invoked in the non-XGB tests)."""
        self.set_params_calls.append(kwargs)


class TestEnsureXgbClassificationObjective:
    """_ensure_xgb_classification_objective."""

    def test_none_model_is_noop(self):
        """model=None must not raise."""
        _ensure_xgb_classification_objective(None, np.array([0, 1, 0]))

    def test_non_xgb_model_is_noop(self):
        """A non-XGB-classifier model's set_params is never called."""
        model = _FakeNonXGBModel()
        _ensure_xgb_classification_objective(model, np.array([0, 1, 2, 1]))
        assert model.set_params_calls == []

    def test_already_multiclass_objective_is_noop(self):
        """An objective already containing 'multi' is left alone."""
        model = _FakeXGBClassifier(objective="multi:softprob")
        _ensure_xgb_classification_objective(model, np.array([0, 1, 2]))
        assert model.set_params_calls == []

    def test_1d_multiclass_target_sets_multi_softprob(self):
        """A 1-D target with >2 unique values sets multi:softprob + num_class."""
        model = _FakeXGBClassifier()
        _ensure_xgb_classification_objective(model, np.array([0, 1, 2, 1, 0, 3]))
        assert model.set_params_calls == [{"objective": "multi:softprob", "num_class": 4}]

    def test_1d_binary_target_is_noop(self):
        """A 1-D target with exactly 2 unique values needs no objective change (binary default suffices)."""
        model = _FakeXGBClassifier()
        _ensure_xgb_classification_objective(model, np.array([0, 1, 0, 1]))
        assert model.set_params_calls == []

    def test_2d_target_sets_binary_logistic_multilabel(self):
        """A 2-D (N, K>=2) target (multilabel one-hot) sets binary:logistic + logloss."""
        model = _FakeXGBClassifier()
        target = np.array([[0, 1], [1, 0], [1, 1]])
        _ensure_xgb_classification_objective(model, target)
        assert model.set_params_calls == [{"objective": "binary:logistic", "eval_metric": "logloss"}]

    def test_get_params_exception_is_swallowed(self):
        """A get_params() failure is caught and treated as a no-op, not propagated."""

        class _BrokenGetParams:
            """A model whose get_params raises."""

            def get_params(self):
                """Raise to simulate a broken estimator."""
                raise RuntimeError("boom")

        # Class name must still match the XGB+Classifier dispatch gate to exercise the try/except.
        _BrokenGetParams.__name__ = "XGBClassifierBroken"
        model = _BrokenGetParams()
        _ensure_xgb_classification_objective(model, np.array([0, 1, 2]))  # must not raise


class TestMaybeWrapFor2dTarget:
    """_maybe_wrap_for_2d_target."""

    def test_none_model_is_noop(self):
        """model=None returns None unchanged."""
        assert _maybe_wrap_for_2d_target(None, np.array([[0, 1]])) is None

    def test_none_target_is_noop(self):
        """train_target=None returns the model unchanged."""
        model = _FakeNonXGBModel()
        assert _maybe_wrap_for_2d_target(model, None) is model

    def test_1d_target_is_noop(self):
        """A 1-D target never needs wrapping (only 2-D targets do)."""
        model = _FakeNonXGBModel()
        out = _maybe_wrap_for_2d_target(model, np.array([0, 1, 0]))
        assert out is model

    def test_already_multioutput_wrapped_is_noop(self):
        """A model whose class name is already a recognized multi-output wrapper is returned unchanged."""

        class MultiOutputClassifier:
            """Stand-in matching the recognized wrapper class name."""

        model = MultiOutputClassifier()
        out = _maybe_wrap_for_2d_target(model, np.array([[0, 1], [1, 0]]))
        assert out is model

    def test_catboost_classifier_is_noop(self):
        """CatBoostClassifier has native multilabel support -- never wrapped."""

        class CatBoostClassifier:
            """Stand-in matching CatBoost's class name."""

        model = CatBoostClassifier()
        out = _maybe_wrap_for_2d_target(model, np.array([[0, 1], [1, 1]]))
        assert out is model

    def test_generic_classifier_gets_wrapped(self):
        """A generic sklearn-style classifier (unrecognized name) with a 2-D target gets wrapped in
        MultiOutputClassifier."""
        pytest.importorskip("sklearn")
        from sklearn.multioutput import MultiOutputClassifier as RealMOC

        model = _FakeNonXGBModel()
        out = _maybe_wrap_for_2d_target(model, np.array([[0, 1], [1, 0], [1, 1]]))
        assert isinstance(out, RealMOC)
        assert out.estimator is model

    def test_wrapped_model_disables_incompatible_early_stopping_params(self):
        """A wrapped model carrying early_stopping/callbacks/early_stopping_rounds gets those
        neutralized (MultiOutputClassifier can't propagate eval_set per label)."""
        pytest.importorskip("sklearn")

        model = _FakeNonXGBModel(early_stopping=True, callbacks=["cb1"], early_stopping_rounds=50)
        out = _maybe_wrap_for_2d_target(model, np.array([[0, 1], [1, 0]]))
        assert out.estimator is model
        assert model.set_params_calls == [{"early_stopping": False, "callbacks": None, "early_stopping_rounds": None}]
