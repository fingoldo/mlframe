"""GPU TreeSHAP *interaction*-tensor parity + biz_value tests. Skipped unless cupy AND a CUDA device.

The cupy/CUDA interaction kernel mirrors the numba interaction kernel exactly (same conditioned-scan
recurrence, same float32 routing). These tests are the correctness gate for the GPU path:
  * PARITY: GPU tensor matches the numba tensor to ~1e-6 (machine-ish),
  * EXACT symmetry  ``Phi_ij == Phi_ji``,
  * row-sum identity ``sum_k Phi_ik == phi_i`` (the main-effect SHAP value from the same kernel),
  * EXACT additivity ``base + Phi.sum((1,2)) == model margin``,
plus a biz_value test quantifying the GPU-vs-numba speedup on a wide proxy and a routing test
(``compute_interaction_tensor`` picks the GPU path on a CUDA box for large enough problems).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

cupy = pytest.importorskip("cupy")
pytest.importorskip("xgboost")
pytest.importorskip("shap")


def _has_device():
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _has_device(), reason="no CUDA device available")]


def _fit_xgb(X, y, *, classification, n_estimators=40, max_depth=5, seed=0):
    from xgboost import XGBClassifier, XGBRegressor

    params = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.2, random_state=seed, tree_method="hist")
    m = XGBClassifier(**params, eval_metric="logloss") if classification else XGBRegressor(**params)
    m.fit(X, y)
    return m


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("max_depth", [3, 6])
def test_gpu_interaction_matches_numba(classification, max_depth):
    """GPU tensor == numba tensor to ~1e-6 + exact symmetry + row-sum identity + additivity."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions import interaction_tensor_numba
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions_gpu import interaction_tensor_gpu

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 9)), columns=[f"f{i}" for i in range(9)])
    signal = X["f0"] * X["f1"] + 0.5 * X["f2"] - X["f3"]  # genuine pairwise interaction f0*f1
    y = (signal + 0.3 * rng.normal(size=200) > 0).astype(int) if classification else (signal + 0.1 * rng.normal(size=200)).to_numpy()
    model = _fit_xgb(X, y, classification=classification, n_estimators=60, max_depth=max_depth)

    ens = extract_ensemble(model)
    Phi_n, phi_n, base_n = interaction_tensor_numba(ens, X.values)
    Phi_g, phi_g, base_g = interaction_tensor_gpu(ens, X.values)

    # PARITY GPU vs numba to machine-ish precision (both are float64 with the same recurrence).
    np.testing.assert_allclose(Phi_g, Phi_n, rtol=0, atol=1e-6)
    np.testing.assert_allclose(phi_g, phi_n, rtol=0, atol=1e-6)
    assert base_g == base_n

    # EXACT symmetry (symmetrised inside the kernel before the diagonal fill).
    np.testing.assert_allclose(Phi_g, np.transpose(Phi_g, (0, 2, 1)), rtol=0, atol=1e-12)

    # Row-sum identity against the GPU kernel's own main-effect phi.
    np.testing.assert_allclose(Phi_g.sum(axis=2), phi_g, rtol=0, atol=1e-10)

    # EXACT additivity in margin space.
    margin = model.predict(X, output_margin=True)
    np.testing.assert_allclose(base_g + Phi_g.sum(axis=(1, 2)), margin, rtol=0, atol=1e-4)


def test_gpu_interaction_handles_missing_values():
    """NaN features route via the node ``missing`` child and still match the numba tensor + additivity."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions import interaction_tensor_numba
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions_gpu import interaction_tensor_gpu

    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(150, 6)), columns=[f"f{i}" for i in range(6)])
    y = (X["f0"] * X["f1"] + 0.5 * X["f2"] + 0.1 * rng.normal(size=150)).to_numpy()
    model = _fit_xgb(X, y, classification=False, max_depth=5)
    Xn = X.copy()
    Xn.iloc[::7, 0] = np.nan
    Xn.iloc[::5, 3] = np.nan

    ens = extract_ensemble(model)
    Phi_n, _phi_n, _ = interaction_tensor_numba(ens, Xn.values)
    Phi_g, _phi_g, base_g = interaction_tensor_gpu(ens, Xn.values)
    np.testing.assert_allclose(Phi_g, Phi_n, rtol=0, atol=1e-6)
    margin = model.predict(Xn, output_margin=True)
    np.testing.assert_allclose(base_g + Phi_g.sum(axis=(1, 2)), margin, rtol=0, atol=1e-4)


def test_gpu_interaction_depth_cap_raises():
    """Beyond the local-scratch depth cap the GPU kernel raises so the caller falls back to numba."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions_gpu import _MAX_SUPPORTED_DEPTH, interaction_tensor_gpu

    class _FakeEnsemble:
        max_depth = _MAX_SUPPORTED_DEPTH + 1
        n_features = 4
        features = np.array([0, 1, -1], dtype=np.int32)

    with pytest.raises(ValueError):
        interaction_tensor_gpu(_FakeEnsemble(), np.zeros((2, 4), dtype=np.float32))


def test_compute_interaction_tensor_gpu_matches_shap():
    """compute_interaction_tensor's GPU backend matches its shap backend (same contract, ~1e-4)."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interactions import compute_interaction_tensor

    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(300, 12)), columns=[f"f{i}" for i in range(12)])
    y = ((X["f0"] > 0) ^ (X["f1"] > 0)).astype(int).to_numpy()
    tmpl = make_default_estimator(classification=True, n_estimators=60)

    Phi_g, base_g = compute_interaction_tensor(tmpl, X, y, classification=True, rng=np.random.default_rng(0), backend="treeshap_gpu")
    Phi_s, base_s = compute_interaction_tensor(tmpl, X, y, classification=True, rng=np.random.default_rng(0), backend="shap")
    np.testing.assert_allclose(Phi_g, Phi_s, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(base_g, base_s, rtol=1e-4, atol=1e-4)


def test_compute_interaction_tensor_routes_to_gpu_on_large_problem():
    """auto routes to the GPU kernel for a supported xgboost model once n*P^2 clears the threshold."""
    from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_interactions as mod
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interactions import compute_interaction_tensor

    rng = np.random.default_rng(4)
    P = 40
    X = pd.DataFrame(rng.normal(size=(800, P)), columns=[f"f{i}" for i in range(P)])
    y = ((X["f0"] > 0) ^ (X["f1"] > 0)).astype(int).to_numpy()
    tmpl = make_default_estimator(classification=True, n_estimators=30)

    # Force the routing threshold low enough that this problem selects the GPU path, and confirm the
    # GPU helper is the one invoked (not numba / shap).
    called = {"gpu": 0}
    orig_gpu = mod._interaction_tensor_gpu

    def _spy(*a, **k):
        called["gpu"] += 1
        return orig_gpu(*a, **k)

    mono = pytest.MonkeyPatch()
    mono.setattr(mod, "_interaction_gpu_min_cells", lambda: 0)
    mono.setattr(mod, "_interaction_tensor_gpu", _spy)
    try:
        Phi, _base = compute_interaction_tensor(tmpl, X, y, classification=True, rng=np.random.default_rng(0), backend="auto")
    finally:
        mono.undo()
    assert called["gpu"] == 1
    assert Phi.shape == (800, P, P)


@pytest.mark.slow
def test_biz_val_gpu_interaction_faster_than_numba():
    """biz_value: the GPU interaction kernel beats the numba kernel on a wide proxy (P~40, ~3000 rows).

    The interaction tensor is the search hotspot on wide proxies; the per-sample x per-conditioning-
    feature passes are embarrassingly parallel, so the GPU path should win once the tensor is large. We
    quantify GPU-vs-numba (and report vs shap) and assert a conservative >=1.15x with parity intact.
    """
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions import interaction_tensor_numba
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions_gpu import interaction_tensor_gpu

    X, y, _ = make_regime_dataset(n_samples=3000, n_informative=6, n_noise=34, task="regression", interaction_order=2, interaction_strength=0.6, seed=5)
    model = _fit_xgb(X, y, classification=False, n_estimators=200, max_depth=4)
    ens = extract_ensemble(model)
    P = ens.n_features

    # JIT/compile warmups (excluded from timing).
    interaction_tensor_numba(ens, X.values[:16])
    interaction_tensor_gpu(ens, X.values[:16])
    cupy.cuda.runtime.deviceSynchronize()

    t0 = time.perf_counter()
    Phi_n, _phi_n, _bn = interaction_tensor_numba(ens, X.values)
    t_numba = time.perf_counter() - t0

    t0 = time.perf_counter()
    Phi_g, _phi_g, _bg = interaction_tensor_gpu(ens, X.values)
    cupy.cuda.runtime.deviceSynchronize()
    t_gpu = time.perf_counter() - t0

    np.testing.assert_allclose(Phi_g, Phi_n, rtol=0, atol=1e-6)
    speedup = t_numba / max(t_gpu, 1e-9)
    print(f"\n[biz_value] P={P} n={X.shape[0]}: numba {t_numba:.3f}s vs gpu {t_gpu:.3f}s -> {speedup:.2f}x GPU speedup")
    assert speedup >= 1.15, f"expected >=1.15x GPU-vs-numba speedup on {P} features, got {speedup:.2f}x (numba {t_numba:.3f}s vs gpu {t_gpu:.3f}s)"
