"""Unit tests for the ShapProxiedFS native-importance pre-filter (``_shap_proxy_prefilter``).

Covers: method routing (``auto`` smart default by width/rows + CUDA presence, explicit pass-through,
unknown -> ValueError), each concrete ranking method keeps the planted informatives, the
original-column mapping (``working_cols`` is sorted original indices and the selector's ``support_``
stays in original-column space after a prefilter), and graceful fall-through when no importances exist.

Fast by design: tiny widths so the four methods run in well under the test budget; the heavy
speed/quality characterization lives in the benchmark + the slow biz_value test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")


# --------------------------------------------------------------------------- routing (no fit needed)
def test_resolve_explicit_methods_pass_through():
    from mlframe.feature_selection._shap_proxy_prefilter import PREFILTER_METHODS, resolve_prefilter_method

    for m in PREFILTER_METHODS:
        assert resolve_prefilter_method(m, n_features=10000, n_rows=4000) == m


def test_resolve_unknown_method_raises():
    from mlframe.feature_selection._shap_proxy_prefilter import resolve_prefilter_method

    with pytest.raises(ValueError):
        resolve_prefilter_method("not_a_method", n_features=100, n_rows=100)


def test_resolve_auto_keeps_model_for_moderate_width():
    from mlframe.feature_selection._shap_proxy_prefilter import _auto_fast_width, resolve_prefilter_method

    narrow = _auto_fast_width() - 1
    assert resolve_prefilter_method("auto", n_features=narrow, n_rows=4000) == "model"


def test_resolve_auto_switches_fast_for_very_wide_when_no_gpu(monkeypatch):
    import mlframe.feature_selection._shap_proxy_prefilter as PF

    # No CUDA -> very-wide auto must pick the cheap interaction-aware fast_model (never model).
    monkeypatch.setattr(PF, "gpu_model_available", lambda: False)
    wide = PF._auto_fast_width() + 5000
    assert PF.resolve_prefilter_method("auto", n_features=wide, n_rows=4000) == "fast_model"
    # Even with many rows, no device means no gpu_model.
    assert PF.resolve_prefilter_method("auto", n_features=wide, n_rows=10 ** 6) == "fast_model"


def test_resolve_auto_routes_gpu_when_device_and_enough_rows(monkeypatch):
    import mlframe.feature_selection._shap_proxy_prefilter as PF

    monkeypatch.setattr(PF, "gpu_model_available", lambda: True)
    wide = PF._auto_fast_width() + 5000
    big_n = PF._gpu_model_min_rows()
    assert PF.resolve_prefilter_method("auto", n_features=wide, n_rows=big_n) == "gpu_model"
    # Too few rows -> GPU transfer overhead not worth it -> fast_model even with a device.
    assert PF.resolve_prefilter_method("auto", n_features=wide, n_rows=big_n - 1) == "fast_model"


# --------------------------------------------------------------------------- ranking correctness
def _wide_xy(seed=0, width=300, n_informative=5):
    """A few strong informatives + lots of independent noise (no clustering needed here)."""
    rng = np.random.default_rng(seed)
    n = 1200
    inf = rng.normal(size=(n, n_informative))
    noise = rng.normal(size=(n, width - n_informative))
    X = pd.DataFrame(np.column_stack([inf, noise]),
                     columns=[f"inf{i}" for i in range(n_informative)]
                     + [f"noise{i}" for i in range(width - n_informative)])
    coefs = np.array([0.9, 0.8, -0.7, 0.6, 0.4])[:n_informative]
    logit = (inf * coefs).sum(axis=1)
    y = (logit + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y


@pytest.mark.parametrize("method", ["model", "univariate", "fast_model"])
def test_prefilter_keeps_informatives_and_returns_sorted_original_indices(method):
    from mlframe.feature_selection._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection._shap_proxy_prefilter import prefilter_columns

    X, y = _wide_xy(width=300, n_informative=5)
    model = make_default_estimator(classification=True, random_state=0, n_estimators=120)
    keep_k = 40
    working_cols, info = prefilter_columns(
        model, X, y.astype(np.float64), method=method, prefilter_top=keep_k,
        classification=True, n_features=X.shape[1])

    # working_cols: sorted, unique, within range, length == kept.
    assert working_cols.ndim == 1 and len(working_cols) == keep_k == info["kept"]
    assert list(working_cols) == sorted(set(int(c) for c in working_cols))
    assert working_cols.min() >= 0 and working_cols.max() < X.shape[1]
    assert info["method"] == method and info["of"] == X.shape[1]

    # All 5 informatives (original indices 0..4) survive the cut for every interaction-aware /
    # marginal-strong method on this main-effect-driven target.
    kept = set(int(c) for c in working_cols)
    assert {0, 1, 2, 3, 4} <= kept, f"{method}: lost informatives, kept head={sorted(kept)[:8]}"


def test_prefilter_no_importance_model_falls_through_to_identity():
    """A model exposing neither feature_importances_ nor coef_ -> keep all columns (identity)."""
    from mlframe.feature_selection._shap_proxy_prefilter import prefilter_columns

    class _NoImportance:
        def get_params(self, deep=False):
            return {}

        def fit(self, X, y):
            return self

    X, y = _wide_xy(width=50)
    working_cols, info = prefilter_columns(
        _NoImportance(), X, y.astype(np.float64), method="model", prefilter_top=10,
        classification=True, n_features=X.shape[1])
    np.testing.assert_array_equal(working_cols, np.arange(X.shape[1]))
    assert info["kept"] == X.shape[1] and info.get("skipped") == "no_importance"


# --------------------------------------------------------------------------- end-to-end selector wire
@pytest.mark.parametrize("method", ["model", "univariate", "fast_model"])
def test_selector_support_stays_in_original_space_under_prefilter(method):
    """The prefilter restricts the working frame; support_ / selected_features_ must still be reported
    in ORIGINAL column space (and name-based transform must match)."""
    pytest.importorskip("shap")
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _wide_xy(seed=1, width=200, n_informative=5)
    sel = ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto", prefilter_top=60,
        prefilter_method=method, cluster_features=True, cluster_corr_threshold=0.7,
        top_n=8, n_splits=3, n_revalidation_models=1, trust_guard=False,
        run_importance_ablation=False, random_state=0, verbose=False)
    sel.fit(X, pd.Series(y))

    assert sel.support_.shape == (X.shape[1],)
    assert sel.support_.dtype == bool
    assert sel.n_features_in_ == X.shape[1]
    assert len(sel.selected_features_) == int(sel.support_.sum())
    # names map back to ORIGINAL columns, and support_ agrees with selected names.
    assert set(sel.selected_features_) <= set(X.columns)
    support_named = {c for c, m in zip(X.columns, sel.support_) if m}
    assert support_named == set(sel.selected_features_)
    # prefilter actually fired and recorded the resolved method.
    pf = sel.shap_proxy_report_["prefilter"]
    assert pf["method"] == method and pf["kept"] == 60 and pf["of"] == X.shape[1]
    # name-based transform returns exactly the selected columns.
    out = sel.transform(X)
    assert list(out.columns) == list(sel.selected_features_)


def test_default_prefilter_method_is_auto():
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    assert ShapProxiedFS().prefilter_method == "auto"
