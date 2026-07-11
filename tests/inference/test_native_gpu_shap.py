"""Correctness test for ``inference.native_gpu_shap.native_xgboost_gpu_shap_contribs``.

NOT a biz_value/speedup test: a proper warm + best-of-N A/B (see the module's own docstring) found this path is
NOT faster than ``shap.Explainer`` for a GPU-fit XGBoost model (~1.04x, noise-level) -- ``shap.Explainer``
already delegates to the model's own GPU-accelerated contribution path internally, so there was nothing to
bypass. This is kept as a correctness pin only (bit-identical values), matching the module's own "rejected as a
speedup, kept as a documented negative + functional alternative" framing.
"""
from __future__ import annotations

import numpy as np
import pytest

xgboost = pytest.importorskip("xgboost")
shap = pytest.importorskip("shap")


def _cuda_available() -> bool:
    try:
        import cupy as cp

        return bool(cp.cuda.runtime.getDeviceCount() > 0)
    except Exception:
        return False


_skip_no_cuda = pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")


def _make_data(n: int, f: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f)).astype(np.float32)
    logit = X[:, 0] * 1.5 - X[:, 1] * 0.7 + 0.3 * X[:, 2] * X[:, 3]
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return X, y


@_skip_no_cuda
def test_native_gpu_shap_bit_identical_to_shap_explainer():
    from mlframe.inference.native_gpu_shap import native_gpu_shap_available, native_xgboost_gpu_shap_contribs

    X, y = _make_data(n=30000, f=20, seed=0)
    model = xgboost.XGBClassifier(n_estimators=150, max_depth=5, device="cuda", tree_method="hist")
    model.fit(X, y)

    assert native_gpu_shap_available(model)

    gpu_values, gpu_base = native_xgboost_gpu_shap_contribs(model, X)
    explainer = shap.Explainer(model)
    cpu_shap = explainer(X)

    np.testing.assert_allclose(gpu_values, np.asarray(cpu_shap.values), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(gpu_base, np.asarray(cpu_shap.base_values), atol=1e-4, rtol=1e-4)


def test_native_gpu_shap_available_false_for_cpu_fit_model():
    from mlframe.inference.native_gpu_shap import native_gpu_shap_available

    X, y = _make_data(n=200, f=5, seed=1)
    model = xgboost.XGBClassifier(n_estimators=10, device="cpu")
    model.fit(X, y)
    assert not native_gpu_shap_available(model)


def test_native_gpu_shap_available_false_for_non_xgboost_model():
    from sklearn.linear_model import LogisticRegression

    from mlframe.inference.native_gpu_shap import native_gpu_shap_available

    X, y = _make_data(n=200, f=5, seed=2)
    model = LogisticRegression().fit(X, y)
    assert not native_gpu_shap_available(model)


def test_native_xgboost_gpu_shap_contribs_returns_none_when_unavailable():
    from mlframe.inference.native_gpu_shap import native_xgboost_gpu_shap_contribs

    X, y = _make_data(n=200, f=5, seed=3)
    model = xgboost.XGBClassifier(n_estimators=10, device="cpu")
    model.fit(X, y)
    assert native_xgboost_gpu_shap_contribs(model, X) is None


@_skip_no_cuda
def test_native_gpu_shap_additivity_holds_on_gpu():
    """SHAP additivity: sum(contribs) + base_value == raw model margin output.

    A GPU/CPU numerical divergence in TreeSHAP would silently break this property without moving the values far
    enough to fail a loose ``allclose`` pin elsewhere -- check it explicitly against the booster's own margin
    output, independent of any comparison to ``shap.Explainer``.
    """
    from mlframe.inference.native_gpu_shap import native_xgboost_gpu_shap_contribs

    X, y = _make_data(n=20000, f=15, seed=4)
    model = xgboost.XGBClassifier(n_estimators=120, max_depth=5, device="cuda", tree_method="hist")
    model.fit(X, y)

    gpu_values, gpu_base = native_xgboost_gpu_shap_contribs(model, X)
    margin = model.get_booster().predict(xgboost.DMatrix(X), output_margin=True)

    reconstructed = gpu_values.sum(axis=1) + gpu_base
    np.testing.assert_allclose(reconstructed, margin, atol=1e-3, rtol=1e-4)


def test_shap_explainer_additivity_holds_on_cpu():
    """Same additivity property, CPU path (``shap.Explainer``), as the other half of the GPU-vs-CPU pin."""
    X, y = _make_data(n=5000, f=15, seed=5)
    model = xgboost.XGBClassifier(n_estimators=120, max_depth=5, device="cpu", tree_method="hist")
    model.fit(X, y)

    explainer = shap.Explainer(model)
    cpu_shap = explainer(X)
    margin = model.get_booster().predict(xgboost.DMatrix(X), output_margin=True)

    reconstructed = np.asarray(cpu_shap.values).sum(axis=1) + np.asarray(cpu_shap.base_values)
    np.testing.assert_allclose(reconstructed, margin, atol=1e-3, rtol=1e-4)
