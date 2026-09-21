"""The two pipeline stages a composite wrapper needs, checked against an oracle at every predict entry point.

``CompositeTargetEstimator`` fits ``alpha, beta`` on the RAW base column at discovery time, while the inner model it wraps is
trained on the entry's ``pre_pipeline`` output (StandardScaler / imputer / encoder). One frame cannot serve both: handing the
wrapper the pre-pipeline-scaled frame collapses ``y = t_hat + alpha*base + beta`` to residual scale, and handing it the raw frame
feeds the inner features it was not trained on. The wrapper therefore carries ``inner_pre_pipeline_`` and derives the inner's
frame itself (or takes it as ``inner_X``), and every entry point must hand it the raw, suite-stage frame.

The oracle each test compares against is "inner on pre_pipeline(X), base read raw" computed directly from the fitted pieces.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.training.composite.estimator._estimator import CompositeTargetEstimator
from mlframe.training.composite.post_shim import PrePipelinePredictShim
from mlframe.training.composite.transforms import _linear_residual_fit, get_transform


def _scaled_inner_scenario(n: int = 2000, seed: int = 0):
    """A composite target whose inner was trained behind a fitted scaler: frame, y, spec params, pipeline, inner and the oracle."""
    rng = np.random.default_rng(seed)
    base = rng.normal(1000.0, 50.0, n)
    feature = rng.normal(0.0, 1.0, n)
    y = 0.9 * base + 10.0 * feature + rng.normal(0.0, 1.0, n)
    params = _linear_residual_fit(y, base)
    transform = get_transform("linear_residual")
    t = transform.forward(y, base, params)
    X = pd.DataFrame({"base": base, "feature": feature})
    pre_pipeline = Pipeline([("scaler", StandardScaler())]).fit(X)
    inner = Ridge().fit(pre_pipeline.transform(X), t)
    oracle = np.asarray(transform.inverse(np.asarray(inner.predict(pre_pipeline.transform(X))), base, params), dtype=np.float64)
    return X, y, params, pre_pipeline, inner, oracle


def _wrap(inner, params, pre_pipeline, y, **kwargs):
    """Wrap ``inner`` as the suite does, optionally without handing the wrapper the inner's pipeline."""
    return CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=inner,
        transform_name="linear_residual",
        base_column="base",
        transform_fitted_params=params,
        y_train=y,
        inner_pre_pipeline=pre_pipeline,
        **kwargs,
    )


def _rmse(pred, y) -> float:
    """Root mean squared error of ``pred`` against ``y``."""
    return float(np.sqrt(np.mean((np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2)))


def test_oracle_beats_both_single_stage_frames():
    """The oracle (inner on scaled X, base raw) must be far better than either single-frame choice, else the test is toothless."""
    X, y, params, pre_pipeline, inner, oracle = _scaled_inner_scenario()
    wrapper_no_pp = _wrap(inner, params, None, y)
    raw_x_everywhere = np.asarray(wrapper_no_pp.predict(X))
    X_scaled = pd.DataFrame(pre_pipeline.transform(X), columns=X.columns)
    scaled_x_everywhere = np.asarray(wrapper_no_pp.predict(X_scaled))
    assert _rmse(oracle, y) < 0.05 * float(np.std(y))
    assert _rmse(raw_x_everywhere, y) > 10.0 * _rmse(oracle, y)
    assert _rmse(scaled_x_everywhere, y) > 10.0 * _rmse(oracle, y)


def test_wrapper_predict_on_raw_frame_matches_oracle():
    """A wrapper carrying the inner's pipeline reproduces the oracle exactly from the raw frame alone."""
    X, y, params, pre_pipeline, inner, oracle = _scaled_inner_scenario()
    wrapper = _wrap(inner, params, pre_pipeline, y)
    np.testing.assert_allclose(np.asarray(wrapper.predict(X)), oracle, rtol=1e-9, atol=1e-9)


def test_wrapper_predict_with_inner_x_override_matches_oracle():
    """``inner_X`` lets a caller that already applied the pipeline keep the base on its raw frame."""
    X, y, params, pre_pipeline, inner, oracle = _scaled_inner_scenario()
    wrapper = _wrap(inner, params, None, y)
    got = np.asarray(wrapper.predict(X, inner_X=pre_pipeline.transform(X)))
    np.testing.assert_allclose(got, oracle, rtol=1e-9, atol=1e-9)


def test_pre_pipeline_shim_routes_composite_components_to_the_oracle():
    """The ensemble's component shim must not scale the frame the wrapper reads its base from, whoever owns the pipeline."""
    X, y, params, pre_pipeline, inner, oracle = _scaled_inner_scenario()
    for wrapper in (_wrap(inner, params, pre_pipeline, y), _wrap(inner, params, None, y)):
        shim = PrePipelinePredictShim(wrapper, pre_pipeline, "composite#0")
        np.testing.assert_allclose(np.asarray(shim.predict(X)), oracle, rtol=1e-9, atol=1e-9)


def test_predict_from_models_matches_oracle():
    """The in-memory entry point feeds the wrapper the suite-stage frame, not its own per-model pre_pipeline output."""
    from mlframe.training.core.predict import predict_from_models

    X, y, params, pre_pipeline, inner, oracle = _scaled_inner_scenario()
    wrapper = _wrap(inner, params, pre_pipeline, y)
    entry = SimpleNamespace(model=wrapper, model_name="cte", columns=list(X.columns), pre_pipeline=pre_pipeline, metrics={})
    models = {"regression": {"y-linres-base": [entry]}}
    metadata = {"columns": list(X.columns), "pipeline": None, "extensions_pipeline": None, "schema_version": 2}
    result = predict_from_models(X, models, metadata, return_probabilities=False, verbose=0)
    assert len(result["predictions"]) == 1
    np.testing.assert_allclose(np.asarray(next(iter(result["predictions"].values()))), oracle, rtol=1e-9, atol=1e-9)


def _save_threads_zero(model, file, zstd_kwargs=None, verbose=0, lean=False, durable=False):
    """Single-threaded zstd write bypassing the Windows ``flush of closed file`` quirk (test-only)."""
    import dill  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
    import zstandard as zstd

    try:
        with open(file, "wb") as f:
            compressor = zstd.ZstdCompressor(level=4, write_checksum=True, write_content_size=True, threads=0)
            with compressor.stream_writer(f) as zf:
                dill.dump(model, zf)
        return True
    except Exception:
        return False


@pytest.mark.parametrize("wrapper_owns_pipeline", [True, False])
def test_predict_mlframe_models_suite_matches_oracle(tmp_path, wrapper_owns_pipeline):
    """The disk entry point must reach the same oracle for a dumped wrapper, whether or not it carries the pipeline itself."""
    import pickle
    from unittest.mock import patch

    import zstandard

    from mlframe.training.core._predict_main_suite import predict_mlframe_models_suite

    X, y, params, pre_pipeline, inner, oracle = _scaled_inner_scenario(n=1500, seed=1)
    wrapper = _wrap(inner, params, pre_pipeline if wrapper_owns_pipeline else None, y)
    if not wrapper_owns_pipeline:
        # A legacy dump whose inner pipeline lives on the entry only: the entry point must still keep the base unscaled.
        inner.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
    entry = SimpleNamespace(model=wrapper, model_name="cte_model", columns=list(X.columns), pre_pipeline=pre_pipeline, metrics={})

    models_path = str(tmp_path)
    model_dir = os.path.join(models_path, "regression", "y")
    os.makedirs(model_dir, exist_ok=True)
    with patch("mlframe.training.io.save_mlframe_model", side_effect=_save_threads_zero):
        from mlframe.training.io import save_mlframe_model

        save_mlframe_model(entry, os.path.join(model_dir, "cte_model.dump"))

    meta_payload = {
        "pipeline": None,
        "extensions_pipeline": None,
        "slug_to_original_target_type": {"regression": "regression"},
        "slug_to_original_target_name": {"y": "y"},
        "columns": list(X.columns),
        "schema_version": 2,
    }
    with open(os.path.join(models_path, "metadata.pkl.zst"), "wb") as f:
        f.write(zstandard.ZstdCompressor(level=3, threads=0).compress(pickle.dumps(meta_payload, protocol=5)))

    result = predict_mlframe_models_suite(X, models_path, return_probabilities=False, verbose=0)
    preds = np.asarray(result["predictions"]["cte_model"])
    if wrapper_owns_pipeline:
        np.testing.assert_allclose(preds, oracle, rtol=1e-9, atol=1e-9)
        return
    # A legacy dump reaches the oracle only approximately (its inner pipeline is reapplied from the entry).
    assert _rmse(preds, y) <= 1.5 * _rmse(oracle, y)
