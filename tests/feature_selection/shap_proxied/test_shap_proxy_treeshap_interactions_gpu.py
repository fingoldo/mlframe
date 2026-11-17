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


import numpy as np
import pandas as pd
import pytest

from tests._perf_paired import assert_paired_speedup
from tests.conftest import skip_if_host_contended

cupy = pytest.importorskip("cupy")
pytest.importorskip("xgboost")
pytest.importorskip("shap")


def _has_device():
    """Has device."""
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _has_device(), reason="no CUDA device available")]


@pytest.fixture(autouse=True)
def _exact_real(monkeypatch, request):
    """The 1e-6 parity tests run the device table kernel in its float64 reference mode; tests named ``test_*float32*`` and the speed test keep the float32 default."""
    if "float32" not in request.node.name and "faster_than_numba" not in request.node.name:
        monkeypatch.setenv("MLFRAME_SHAP_TREESHAP_REAL", "float64")


def _fit_xgb(X, y, *, classification, n_estimators=40, max_depth=5, seed=0):
    """Fit xgb."""
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
        """Groups tests covering FakeEnsemble."""
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

    # Force the tuner's answer to the device so this problem selects the GPU path, and confirm the
    # GPU helper is the one invoked (not numba / shap).
    called = {"gpu": 0}
    orig_gpu = mod._interaction_tensor_gpu

    def _spy(*a, **k):
        """Helper that spy."""
        called["gpu"] += 1
        return orig_gpu(*a, **k)

    mono = pytest.MonkeyPatch()
    mono.setattr(mod, "_interaction_gpu_pays", lambda n, p, t: True)
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

    # The numba arm uses every core and the GPU arm shares the card with whatever else is running, so the ratio carries no signal while the host is busy.
    skip_if_host_contended("a GPU-versus-numba speed ratio is not measurable while other processes use the cores and the card")

    X, y, _ = make_regime_dataset(n_samples=3000, n_informative=6, n_noise=34, task="regression", interaction_order=2, interaction_strength=0.6, seed=5)
    model = _fit_xgb(X, y, classification=False, n_estimators=200, max_depth=4)
    ens = extract_ensemble(model)
    P = ens.n_features

    # JIT/compile warmups (excluded from timing).
    interaction_tensor_numba(ens, X.values[:16])
    interaction_tensor_gpu(ens, X.values[:16])
    cupy.cuda.runtime.deviceSynchronize()

    def _run_numba():
        """One numba interaction-tensor pass over the full frame."""
        return interaction_tensor_numba(ens, X.values)

    def _run_gpu():
        """One GPU interaction-tensor pass, synchronised so the timer sees the finished kernels."""
        out = interaction_tensor_gpu(ens, X.values)
        cupy.cuda.runtime.deviceSynchronize()
        return out

    # Paired, interleaved rounds: a stall that lands on one arm in one round cannot decide the verdict.
    (Phi_n, _phi_n, _bn), (Phi_g, _phi_g, _bg) = assert_paired_speedup(
        _run_numba, _run_gpu, base_ratio=1.15, n_trials=3, warmup=False, what=f"the GPU interaction kernel against numba on {P} features"
    )
    np.testing.assert_allclose(Phi_g, Phi_n, rtol=0, atol=1e-4)  # float32 device tables


def _nan_ensemble(depth, n=400, P=12, seed=7):
    """An xgboost ensemble trained on NaN-bearing data, with a NaN-bearing evaluation frame."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, P)).astype(np.float32)
    X[rng.random(X.shape) < 0.07] = np.nan
    y = np.nan_to_num(X[:, 0] * X[:, 1] + X[:, 2] - X[:, 3])
    model = _fit_xgb(X, y, classification=False, n_estimators=50, max_depth=depth)
    return extract_ensemble(model), model, X


@pytest.mark.parametrize("depth", [3, 6])
def test_gpu_float32_table_matches_numba_with_nan(depth):
    """The float32 pair-table kernel (the default) matches the numba tensor to 1e-4 with NaN routing, keeps the mean|phi| rank, symmetry and the row-sum identity."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions import interaction_tensor_numba
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions_table_gpu import interaction_tensor_table_gpu

    ens, model, X = _nan_ensemble(depth)
    Phi_n, phi_n, _ = interaction_tensor_numba(ens, X)
    out = interaction_tensor_table_gpu(ens, X)
    assert out is not None
    Phi_g, phi_g, base_g = out
    np.testing.assert_allclose(Phi_g, Phi_n, rtol=0, atol=1e-4)
    np.testing.assert_allclose(phi_g, phi_n, rtol=0, atol=1e-4)
    assert (np.argsort(-np.abs(phi_g).mean(0)) == np.argsort(-np.abs(phi_n).mean(0))).all()
    np.testing.assert_allclose(Phi_g, np.transpose(Phi_g, (0, 2, 1)), rtol=0, atol=1e-6)
    np.testing.assert_allclose(Phi_g.sum(axis=2), phi_g, rtol=0, atol=1e-5)
    np.testing.assert_allclose(base_g + Phi_g.sum(axis=(1, 2)), model.predict(X, output_margin=True), rtol=0, atol=1e-3)


def test_gpu_float64_table_equals_numba_to_1e9_with_nan():
    """Float64 reference mode of the pair-table kernel equals the numba tensor to 1e-9 with NaNs."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions import interaction_tensor_numba
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions_table_gpu import interaction_tensor_table_gpu

    ens, _model, X = _nan_ensemble(5)
    Phi_n, phi_n, _ = interaction_tensor_numba(ens, X)
    Phi_g, phi_g, _ = interaction_tensor_table_gpu(ens, X)
    np.testing.assert_allclose(Phi_g, Phi_n, rtol=0, atol=1e-9)
    np.testing.assert_allclose(phi_g, phi_n, rtol=0, atol=1e-9)


def test_gpu_float32_table_row_chunks_equal_single_launch(monkeypatch):
    """Row chunking of the device accumulator (a tiny budget forces many launches) gives the single-launch tensor."""
    from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_treeshap_interactions_table_gpu as mod

    ens, _model, X = _nan_ensemble(4, n=700)
    whole = mod.interaction_tensor_table_gpu(ens, X)
    monkeypatch.setattr(mod, "_PHI_CHUNK_BYTES", 12 * 12 * 4 * 128)
    chunked = mod.interaction_tensor_table_gpu(ens, X)
    np.testing.assert_array_equal(whole[0], chunked[0])
    np.testing.assert_array_equal(whole[1], chunked[1])


def test_gpu_interaction_falls_back_to_path_kernel_when_pair_table_too_big(monkeypatch):
    """A pair table above the entry cap returns None from the table form and ``interaction_tensor_gpu`` keeps the conditioned-scan kernel (same tensor)."""
    from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_treeshap_interactions_table_gpu as mod
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions_gpu import interaction_tensor_gpu

    ens, _model, X = _nan_ensemble(3, n=150)
    ref = mod.interaction_tensor_table_gpu(ens, X)
    monkeypatch.setattr(mod, "MAX_PAIR_TABLE_ENTRIES", 1)
    assert mod.interaction_tensor_table_gpu(ens, X) is None
    Phi, phi, _ = interaction_tensor_gpu(ens, X)
    np.testing.assert_allclose(Phi, ref[0], rtol=0, atol=1e-4)
    np.testing.assert_allclose(phi, ref[1], rtol=0, atol=1e-4)


def test_interaction_backend_choice_follows_the_tuner_entry(monkeypatch):
    """The device-or-host choice asks the ``shap_proxy_treeshap_interactions_backend`` entry with cells = rows x features^2 and the tree count; its fallback floor sits between the measured crossover points."""
    from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_interactions as mod
    from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_treeshap_interactions_ktc as ktc

    assert ktc._SPEC is not None and set(ktc._SPEC.axes) == {"cells", "trees"}
    assert ktc._fallback(900_000, 100) == "numba" and ktc._fallback(3_600_000, 100) == "gpu"
    seen = []
    monkeypatch.setattr(ktc, "use_gpu", lambda cells, trees: (seen.append((cells, trees)), True)[1])
    assert mod._interaction_gpu_pays(800, 20, 30) is True
    assert seen == [(800 * 20 * 20, 30)]


def test_path_interaction_kernel_streams_row_blocks_with_the_same_tensor(monkeypatch):
    """The conditioned-scan interaction kernel (the fallback and reference) streams the rows in blocks sized from the free memory: with a (1, 1) memory report it runs 1024-row blocks over 2500 rows and equals the one-block tensor."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_treeshap_interactions_gpu as g
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble

    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2500, 6)).astype(np.float32)
    y = X[:, 0] * X[:, 1] + X[:, 2]
    ens = extract_ensemble(xgb.XGBRegressor(n_estimators=8, max_depth=3, verbosity=0, n_jobs=1).fit(X, y))
    X[::9, 1] = np.nan
    whole = g.interaction_tensor_gpu_path(ens, X)
    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: (1, 1))
    blocked = g.interaction_tensor_gpu_path(ens, X)
    np.testing.assert_array_equal(blocked[0], whole[0])
    np.testing.assert_array_equal(blocked[1], whole[1])
    assert blocked[2] == whole[2]
