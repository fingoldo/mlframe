"""The native XGBoost contribution path passes the model's missing sentinel, categorical flag and best iteration, so contributions describe the served model."""

from __future__ import annotations

from types import SimpleNamespace

from mlframe.inference.native_gpu_shap import native_contrib_kwargs


def test_missing_sentinel_categorical_flag_and_best_iteration_are_forwarded():
    """missing=-999, enable_categorical=True, best_iteration=39 -> DMatrix(missing, enable_categorical) and predict(iteration_range=(0, 40))."""
    stub = SimpleNamespace(missing=-999.0, enable_categorical=True, best_iteration=39)
    matrix, predict = native_contrib_kwargs(stub)
    assert matrix == {"missing": -999.0, "enable_categorical": True}
    assert predict == {"iteration_range": (0, 40)}


def test_defaults_add_nothing():
    """NaN missing, no categorical flag, no early stopping: the historical call (negative control)."""

    class _NoEarlyStop:
        """Stub model without early stopping."""

        missing = float("nan")
        enable_categorical = False

        @property
        def best_iteration(self):
            """xgboost raises AttributeError without early stopping."""
            raise AttributeError("no best_iteration")

    assert native_contrib_kwargs(_NoEarlyStop()) == ({}, {})
