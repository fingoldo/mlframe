"""Shared helpers of the ``test_treeshap_vs_shap_*`` files: data makers, the official-shap reference, our backends, the invariants."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_explain as spe
from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_treeshap as ts

shap = pytest.importorskip("shap")
xgb = pytest.importorskip("xgboost")
lgb = pytest.importorskip("lightgbm")

ATOL_CPU = 1e-9
ATOL_F32 = 1e-4


def gpu_available() -> bool:
    """Whether a CUDA device usable for the TreeSHAP kernels is present."""
    try:
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_gpu import gpu_treeshap_available

        return bool(gpu_treeshap_available())
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not gpu_available(), reason="no CUDA device / cupy unavailable")


def make_data(n: int = 300, f: int = 6, seed: int = 0, kind: str = "reg"):
    """Return ``(X float32, y)``; kind is ``reg``, ``bin`` or ``multi`` (3 classes)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f)).astype(np.float32)
    signal = 2 * X[:, 0] + X[:, 1] * X[:, 2 % f] - np.sin(X[:, 3 % f]) + 0.2 * rng.normal(size=n)
    if kind == "reg":
        return X, signal
    if kind == "bin":
        return X, (signal > np.median(signal)).astype(int)
    q = np.quantile(signal, [1 / 3, 2 / 3])
    return X, np.digitize(signal, q).astype(int)


def shap_ref(model, X):
    """Official ``shap`` path-dependent values: ``(phi, base)`` for a single-output model, positive class for a classifier."""
    with spe._maybe_patch_shap_xgb_base_score():
        ex = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        phi = ex.shap_values(X, check_additivity=False)
    if isinstance(phi, list):
        phi = phi[-1]
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim == 3:
        phi = phi[:, :, -1]
    base = np.ravel(ex.expected_value)[-1]
    return phi, float(base)


def xgb_ref64(model, X):
    """Official shap C++ float64 path-dependent kernel on an XGBoost model.

    ``shap.TreeExplainer(xgb_model)`` delegates to xgboost's native float32 ``pred_contribs`` (about 1e-6 off a double-precision TreeSHAP), so the exact reference feeds shap the same trees as a plain dict
    parsed independently through ``trees_to_dataframe`` (xgboost routes ``x < split``, shap ``x <= threshold``: the threshold is moved one float32 down). Returns phi (n, f)."""
    ex = _xgb_dict_explainer(model)
    return np.asarray(ex.shap_values(np.asarray(X, dtype=np.float64), check_additivity=False), dtype=np.float64)


def xgb_interactions_ref64(model, X):
    """Official shap float64 interaction values on an XGBoost model fed as a plain tree dict (see ``xgb_ref64``); (n, f, f)."""
    ex = _xgb_dict_explainer(model)
    return np.asarray(ex.shap_interaction_values(np.asarray(X, dtype=np.float64)), dtype=np.float64)


def _xgb_dict_explainer(model):
    """``shap.TreeExplainer`` over the XGBoost trees parsed into shap's plain dict format."""
    booster = model.get_booster()
    names = booster.feature_names
    df = booster.trees_to_dataframe()
    trees = []
    for _, g in df.groupby("Tree"):
        g = g.assign(i=g["ID"].str.split("-").str[1].astype(int)).sort_values("i")
        idx = {node_id: k for k, node_id in enumerate(g["ID"])}
        leaf = (g["Feature"] == "Leaf").to_numpy()

        def child(col, idx=idx, leaf=leaf, g=g):
            """Child column mapped to local indices, -1 on leaves."""
            return np.array([-1 if lf else idx[c] for lf, c in zip(leaf, g[col])])

        split = g["Split"].fillna(0).to_numpy(np.float32)
        trees.append(
            dict(
                children_left=child("Yes"),
                children_right=child("No"),
                children_default=child("Missing"),
                features=np.array([-2 if lf else (names.index(f) if names else int(f[1:])) for lf, f in zip(leaf, g["Feature"])]),
                thresholds=np.where(leaf, 0.0, np.nextafter(split, np.float32(-np.inf)).astype(np.float64)),
                values=np.where(leaf, g["Gain"], 0.0).reshape(-1, 1),
                node_sample_weight=g["Cover"].to_numpy(),
            )
        )
    return shap.TreeExplainer({"trees": trees}, feature_perturbation="tree_path_dependent")


def shap_interactions_ref(model, X):
    """Official ``shap_interaction_values`` as a float64 (n, f, f) array."""
    with spe._maybe_patch_shap_xgb_base_score():
        ex = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        out = ex.shap_interaction_values(X)
    if isinstance(out, list):
        out = out[-1]
    return np.asarray(out, dtype=np.float64)


def margin(model, X):
    """Raw model margin (log-odds / target space) of an xgboost or lightgbm sklearn model."""
    if type(model).__module__.startswith("xgboost"):
        booster = model.get_booster()
        return booster.predict(xgb.DMatrix(np.asarray(X, dtype=np.float32), feature_names=booster.feature_names), output_margin=True)
    return model.predict(X, raw_score=True)


def ours(model, X, which: str = "numba"):
    """``(phi, base)`` of our backend: ``numba`` dispatcher, ``path`` kernel, ``table`` kernel, ``gpu`` / ``gpu_path`` device kernels."""
    ens = ts.extract_ensemble(model)
    assert ens is not None
    if which == "numba":
        return ts.treeshap_phi_base_numba(ens, X)
    if which == "path":
        return ts.treeshap_phi_base_path(ens, X)
    if which == "table":
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_table import treeshap_phi_base_table

        out = treeshap_phi_base_table(ens, np.ascontiguousarray(X, dtype=np.float32))
        assert out is not None
        return out
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_gpu import treeshap_phi_base_gpu

    return treeshap_phi_base_gpu(ens, X, use_table=(which == "gpu"))


def is_xgb(model) -> bool:
    """Whether ``model`` is an xgboost sklearn model."""
    return type(model).__module__.startswith("xgboost")


def assert_matches_shap(model, X, which: str = "numba", atol: float = ATOL_CPU, ref_X=None):
    """Our values equal official shap within ``atol``, additivity and the mean|phi| rank hold.

    LightGBM: ``TreeExplainer(model)`` directly, base equals ``expected_value``. XGBoost: the shap float64 kernel via ``xgb_ref64`` (shap's direct XGBoost path is xgboost's float32 native one, see
    ``test_xgboost_matches_shap_native_float32``)."""
    Xr = X if ref_X is None else ref_X
    phi, base = ours(model, X, which)
    if is_xgb(model):
        ref = xgb_ref64(model, Xr)
    else:
        ref, ref_base = shap_ref(model, Xr)
        assert base == pytest.approx(ref_base, abs=atol)
    assert phi.shape == ref.shape
    np.testing.assert_allclose(phi, ref, rtol=0, atol=atol)
    mg = margin(model, Xr)
    # xgboost's margin is a float32 sum; lightgbm's is double precision
    add_tol = 5e-6 + 2e-6 * float(np.abs(mg).max(initial=0.0)) if is_xgb(model) else 1e-9 * (1 + float(np.abs(mg).max(initial=0.0)))
    if atol > ATOL_CPU:
        add_tol = max(add_tol, atol * 10)
    np.testing.assert_allclose(base + phi.sum(1), mg, rtol=0, atol=add_tol)
    assert_same_rank(phi, ref)
    return phi, base


def assert_same_rank(phi, ref):
    """The features order by mean|phi| alike (ties within 1e-6 are not ordered)."""
    a, b = np.abs(phi).mean(0), np.abs(ref).mean(0)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-4)
    for i in np.argsort(-b)[:-1]:
        for j in np.argsort(-b):
            if b[i] - b[j] > 1e-4:
                assert a[i] > a[j]


def fit_xgb(X, y, kind="reg", **kw):
    """Small XGBoost sklearn model; defaults are deterministic and fast."""
    params = dict(n_estimators=15, max_depth=4, tree_method="hist", verbosity=0, n_jobs=1)
    params.update(kw)
    cls = xgb.XGBRegressor if kind == "reg" else xgb.XGBClassifier
    sw = params.pop("sample_weight", None)
    return cls(**params).fit(X, y, sample_weight=sw)


def fit_lgb(X, y, kind="reg", **kw):
    """Small LightGBM sklearn model; defaults are deterministic and fast."""
    params = dict(n_estimators=15, num_leaves=15, learning_rate=0.2, min_child_samples=5, verbose=-1, n_jobs=1, random_state=0)
    params.update(kw)
    cls = lgb.LGBMRegressor if kind == "reg" else lgb.LGBMClassifier
    return cls(**params).fit(X, y)
