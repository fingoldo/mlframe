"""CatBoost cached-quantization-borders fast path for RFECV fold fits: identity vs the generic fit, and dispatch behaviour."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

catboost = pytest.importorskip("catboost")
from catboost import CatBoostClassifier, CatBoostRegressor

from mlframe.feature_selection.wrappers import RFECV
from mlframe.feature_selection.wrappers.rfecv import _cb_border_cache as cbc


def _frame(seed: int, n: int = 1200, p: int = 8, with_cat: bool = False, with_nan: bool = False):
    """Build a float32 frame with a constant column and optional categorical and NaN columns."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)).astype(np.float32), columns=[f"f{i}" for i in range(p)])
    X["const"] = 1.0
    if with_nan:
        X.iloc[::9, 2] = np.nan
    cats = []
    if with_cat:
        X["cat0"] = pd.Categorical(rng.choice(list("abcd"), n))
        cats = ["cat0"]
    y = pd.Series(((X.f0 + X.f1 * X.f2.fillna(0) + rng.normal(size=n)) > 0).astype(int))
    return X, y, cats


def _generic_fit(cls, Xs, ys, Xv, yv, cats, weight, **kw):
    """Fit a CatBoost model on the given slices with early stopping and return it."""
    m = cls(iterations=25, depth=4, verbose=0, allow_writing_files=False, random_seed=3, **kw)
    m.fit(Xs, ys, eval_set=(Xv, yv), cat_features=cats or None, sample_weight=weight, use_best_model=True, early_stopping_rounds=10)
    return m


@pytest.mark.parametrize("with_cat,with_nan,weighted", [(False, False, False), (True, True, False), (False, True, True), (True, False, True)])
def test_cached_borders_subset_fit_bit_identical_to_generic_fit(with_cat, with_nan, weighted):
    """Cached borders subset fit bit identical to generic fit."""
    X, y, cats = _frame(0, with_cat=with_cat, with_nan=with_nan)
    Xt, yt, Xv, yv = X.iloc[:1000], y.iloc[:1000], X.iloc[1000:], y.iloc[1000:]
    w = np.random.default_rng(1).uniform(0.5, 2.0, len(Xt)) if weighted else None
    rows = np.arange(len(Xt))
    src = type("Src", (), {})()
    full = list(X.columns)
    for subset in (full, [c for c in full if c not in ("f1", "f5")], ["f0", "f2", "f3", *cats]):
        params = {"cat_features": [subset.index(c) for c in cats if c in subset], "eval_set": (Xv[subset], yv), "use_best_model": True, "early_stopping_rounds": 10}
        m = CatBoostClassifier(iterations=25, depth=4, verbose=0, allow_writing_files=False, random_seed=3)
        assert cbc.fit_catboost_with_cached_borders(m, source=src, X_train=Xt[subset], y_train=yt, fit_features=subset, fit_params=params, train_rows=rows, sample_weight=w)
        ref = _generic_fit(CatBoostClassifier, Xt[subset], yt, Xv[subset], yv, [subset.index(c) for c in cats if c in subset], w)
        assert np.array_equal(m.predict_proba(Xv[subset]), ref.predict_proba(Xv[subset]))
    cache = cbc._CACHES[id(src)]
    assert cache.misses == 1 and cache.hits == 2


def test_cached_borders_regressor_identical():
    """Cached borders regressor identical."""
    X, y, _ = _frame(1)
    Xt, yt, Xv, yv = X.iloc[:1000], y.iloc[:1000].astype(float), X.iloc[1000:], y.iloc[1000:].astype(float)
    src = type("Src", (), {})()
    full, sub = list(X.columns), ["f0", "f1", "f4"]
    for cols in (full, sub):
        m = CatBoostRegressor(iterations=20, depth=4, verbose=0, allow_writing_files=False, random_seed=3)
        assert cbc.fit_catboost_with_cached_borders(m, source=src, X_train=Xt[cols], y_train=yt, fit_features=cols, fit_params={"eval_set": (Xv[cols], yv)}, train_rows=np.arange(1000))
        ref = CatBoostRegressor(iterations=20, depth=4, verbose=0, allow_writing_files=False, random_seed=3).fit(Xt[cols], yt, eval_set=(Xv[cols], yv))
        assert np.array_equal(m.predict(Xv[cols]), ref.predict(Xv[cols]))


def _rfecv(est, cats, **kw):
    """Build a small RFECV around the estimator."""
    return RFECV(estimator=est, cat_features=cats or None, cv=3, max_refits=4, verbose=0, leakage_corr_threshold=None, random_state=0, **kw)


@pytest.mark.parametrize("seed,with_cat,with_nan,weighted", [(0, False, False, False), (1, True, True, False), (2, False, True, True), (3, True, False, True)])
def test_rfecv_selection_identical_with_and_without_cached_borders(monkeypatch, seed, with_cat, with_nan, weighted):
    """Rfecv selection identical with and without cached borders."""
    X, y, cats = _frame(seed, with_cat=with_cat, with_nan=with_nan)
    w = np.random.default_rng(seed).uniform(0.5, 2.0, len(X)) if weighted else None
    out = {}
    for label, env in (("off", "0"), ("on", "1")):
        monkeypatch.setenv("MLFRAME_RFECV_CB_CACHED_BORDERS", env)
        cbc.TOTALS.update(hits=0, misses=0, fallbacks=0)
        r = _rfecv(CatBoostClassifier(iterations=15, depth=3, verbose=0, allow_writing_files=False, random_seed=seed), cats)
        r.fit(X, y, sample_weight=w) if weighted else r.fit(X, y)
        out[label] = (list(r.support_), np.asarray(r.cv_results_["cv_mean_perf"]), dict(cbc.TOTALS))
    assert out["off"][0] == out["on"][0]
    assert np.array_equal(out["off"][1], out["on"][1])
    assert out["off"][2] == {"hits": 0, "misses": 0, "fallbacks": 0}
    assert out["on"][2]["hits"] > 0 and out["on"][2]["fallbacks"] == 0


def test_fast_path_used_for_catboost_and_skipped_otherwise(monkeypatch):
    """Fast path used for catboost and skipped otherwise."""
    from sklearn.ensemble import RandomForestClassifier

    monkeypatch.setenv("MLFRAME_RFECV_CB_CACHED_BORDERS", "1")
    X, y, _ = _frame(5)
    cbc.TOTALS.update(hits=0, misses=0, fallbacks=0)
    _rfecv(RandomForestClassifier(n_estimators=10, random_state=0), []).fit(X, y)
    assert cbc.TOTALS == {"hits": 0, "misses": 0, "fallbacks": 0}
    assert not cbc.is_supported(RandomForestClassifier(), {})
    assert cbc.is_supported(CatBoostClassifier(verbose=0), {})
    assert not cbc.is_supported(CatBoostClassifier(verbose=0, task_type="GPU"), {})
    assert not cbc.is_supported(CatBoostClassifier(verbose=0), {"text_features": ["t"]})
    monkeypatch.setenv("MLFRAME_RFECV_CB_CACHED_BORDERS", "0")
    assert not cbc.is_supported(CatBoostClassifier(verbose=0), {})


def test_cache_not_shared_across_source_frames():
    """Cache not shared across source frames."""
    a, b = type("Src", (), {})(), type("Src", (), {})()
    assert cbc._get_cache(a) is not cbc._get_cache(b)
    assert cbc._get_cache(a) is cbc._get_cache(a)


def test_cb_cached_borders_param_default_true_and_roundtrips():
    """Cb cached borders param default true and roundtrips."""
    import pickle

    from sklearn.base import clone

    r = _rfecv(CatBoostClassifier(verbose=0), [])
    assert r.cb_cached_borders is True and r.get_params()["cb_cached_borders"] is True
    off = RFECV(estimator=CatBoostClassifier(verbose=0), cb_cached_borders=False)
    assert clone(off).cb_cached_borders is False
    assert pickle.loads(pickle.dumps(off)).cb_cached_borders is False


def test_cb_cached_borders_param_false_disables_fast_path_and_true_uses_it(monkeypatch):
    """Cb cached borders param false disables fast path and true uses it."""
    monkeypatch.delenv("MLFRAME_RFECV_CB_CACHED_BORDERS", raising=False)
    X, y, _ = _frame(0)
    for flag in (False, True):
        cbc.TOTALS.update(hits=0, misses=0, fallbacks=0)
        r = _rfecv(CatBoostClassifier(iterations=10, depth=3, verbose=0, allow_writing_files=False, random_seed=0), [], cb_cached_borders=flag)
        r.fit(X, y)
        assert (cbc.TOTALS["hits"] > 0) is flag


def test_cb_cached_borders_env_zero_overrides_param_true(monkeypatch):
    """Cb cached borders env zero overrides param true."""
    monkeypatch.setenv("MLFRAME_RFECV_CB_CACHED_BORDERS", "0")
    X, y, _ = _frame(0)
    cbc.TOTALS.update(hits=0, misses=0, fallbacks=0)
    _rfecv(CatBoostClassifier(iterations=10, depth=3, verbose=0, allow_writing_files=False, random_seed=0), [], cb_cached_borders=True).fit(X, y)
    assert cbc.TOTALS["hits"] == 0


def test_feature_selection_config_rfecv_kwargs_accepts_cb_cached_borders():
    """Feature selection config rfecv kwargs accepts cb cached borders."""
    from mlframe.training import FeatureSelectionConfig

    cfg = FeatureSelectionConfig(rfecv_models=["cb_rfecv"], rfecv_kwargs={"cb_cached_borders": False})
    assert cfg.rfecv_kwargs == {"cb_cached_borders": False}
    with pytest.raises(ValueError, match="unknown key"):
        FeatureSelectionConfig(rfecv_models=["cb_rfecv"], rfecv_kwargs={"cb_cached_border": False})
