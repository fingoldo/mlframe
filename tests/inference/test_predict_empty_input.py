"""Regression: ``predict_from_models`` must raise (not silently return
empty) when every supplied model fails at predict.

Pre-fix path (iter-45 500k cb-regression seed=99):
1. ``train_mlframe_models_suite`` returned 1 trained CB model + metadata.
2. ``predict_from_models`` ran the per-model loop, the cb model raised
   inside ``_try_predict``, the ``except Exception`` swallow at the
   bottom of the loop logged ``logger.error("Error predicting with
   model regression_y: <root cause>")`` and ``continue``-d.
3. The function fell through to ``return results`` with
   ``results["predictions"] == {}`` and
   ``results["probabilities"] == {}``.
4. The harness in ``_profile_fuzz_1m.py`` caught this empty dict and
   raised its own ``RuntimeError("predict_from_models returned empty
   predictions+probabilities (...; per_target_probs keys: [])")`` -- a
   non-actionable surface that hid the actual per-model exception.

Post-fix: predict_from_models tracks per-model errors and, if EVERY
attempted model failed, raises a single ``RuntimeError`` whose message
contains the original per-model exception text. Surviving-model cases
still return normally (partial success is the intended best-effort
contract for multi-model suites).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

# Synthetic SimpleNamespace stand-in models + tiny pandas frames; no estimator fits, wall <0.5s.
pytestmark = [pytest.mark.fast]


class _AlwaysFailingModel:
    """Stand-in for a fitted estimator whose ``.predict`` always raises.

    Mimics the production-time failure that prompted iter-46: a CB
    regression model whose predict call tripped at the pipeline / NaN
    guard / pool-cache layer."""

    def __init__(self, msg: str = "synthetic predict failure"):
        self._msg = msg
        # feature_names_in_ makes the wrapper's expected-cols probe satisfied.
        self.feature_names_in_ = np.array(["x0", "x1"], dtype=object)

    def predict(self, X):
        raise RuntimeError(self._msg)


def _build_models_dict(*, n_models: int = 1, fail: bool = True):
    """Build a single-target single-tier models dict that
    ``predict_from_models`` accepts: ``models[target_type][target_name]
    -> [obj_with_model_attr, ...]``."""
    obj_list = []
    for i in range(n_models):
        if fail:
            inner = _AlwaysFailingModel(f"synthetic predict failure #{i}")
        else:
            class _Ok:
                feature_names_in_ = np.array(["x0", "x1"], dtype=object)
                def predict(self, X):
                    return np.zeros(len(X), dtype=np.float64)
            inner = _Ok()
        obj_list.append(SimpleNamespace(model=inner, pre_pipeline=None))
    return {"regression": {"y": obj_list}}


def _build_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x0": rng.standard_normal(50).astype(np.float64),
        "x1": rng.standard_normal(50).astype(np.float64),
    })


def test_all_models_failing_raises_with_aggregated_message() -> None:
    """All supplied models error -> predict_from_models raises
    RuntimeError whose message includes the per-model exception text."""
    from mlframe.training.core.predict import predict_from_models
    df = _build_df()
    models = _build_models_dict(n_models=1, fail=True)
    metadata = {"columns": ["x0", "x1"], "raw_input_columns": ["x0", "x1"]}
    with pytest.raises(RuntimeError) as excinfo:
        predict_from_models(
            df=df, models=models, metadata=metadata,
            features_and_targets_extractor=None,
            return_probabilities=False, verbose=0,
        )
    msg = str(excinfo.value)
    # Aggregated header naming the supplied count.
    assert "all 1 supplied model(s) failed at predict" in msg
    # Original per-model failure message must be embedded -- this is what
    # the harness in _profile_fuzz_1m.py was missing.
    assert "synthetic predict failure" in msg


def test_multi_model_all_failing_raises_with_truncated_message() -> None:
    """6 failing models -> message names first 5 + ``(+1 more)``."""
    from mlframe.training.core.predict import predict_from_models
    df = _build_df()
    models = _build_models_dict(n_models=6, fail=True)
    metadata = {"columns": ["x0", "x1"], "raw_input_columns": ["x0", "x1"]}
    with pytest.raises(RuntimeError) as excinfo:
        predict_from_models(
            df=df, models=models, metadata=metadata,
            features_and_targets_extractor=None,
            return_probabilities=False, verbose=0,
        )
    msg = str(excinfo.value)
    assert "all 6 supplied model(s) failed at predict" in msg
    # Truncation marker proves >5 failures got accumulated.
    assert "(+1 more)" in msg


def test_partial_success_does_not_raise() -> None:
    """Mixed-success case: 1 ok + 1 failing -> returns normally with
    the ok model's prediction; the failing one is logged + swallowed
    (back-compat: this is the documented best-effort behaviour for
    multi-model suites)."""
    from mlframe.training.core.predict import predict_from_models
    df = _build_df()
    # 1 failing + 1 working
    failing = SimpleNamespace(
        model=_AlwaysFailingModel("synthetic predict failure mixed"),
        pre_pipeline=None,
    )

    class _Ok:
        feature_names_in_ = np.array(["x0", "x1"], dtype=object)
        def predict(self, X):
            return np.full(len(X), 1.5, dtype=np.float64)

    working = SimpleNamespace(model=_Ok(), pre_pipeline=None)
    models = {"regression": {"y": [failing, working]}}
    metadata = {"columns": ["x0", "x1"], "raw_input_columns": ["x0", "x1"]}
    result = predict_from_models(
        df=df, models=models, metadata=metadata,
        features_and_targets_extractor=None,
        return_probabilities=False, verbose=0,
    )
    # Must have exactly the working model's prediction.
    assert isinstance(result, dict)
    assert len(result["predictions"]) >= 1
    # At least one prediction array should match the working model's output.
    assert any(
        isinstance(_v, np.ndarray) and np.allclose(_v, 1.5)
        for _v in result["predictions"].values()
    )


def test_zero_models_supplied_does_not_raise() -> None:
    """Empty models dict -> empty results; do NOT raise (this is the
    legitimate "no models" path used by integration tests). The fix is
    scoped to "models were supplied AND all failed at predict"."""
    from mlframe.training.core.predict import predict_from_models
    df = _build_df()
    metadata = {"columns": ["x0", "x1"], "raw_input_columns": ["x0", "x1"]}
    result = predict_from_models(
        df=df, models={}, metadata=metadata,
        features_and_targets_extractor=None,
        return_probabilities=False, verbose=0,
    )
    assert isinstance(result, dict)
    assert result["predictions"] == {}
    assert result["probabilities"] == {}
