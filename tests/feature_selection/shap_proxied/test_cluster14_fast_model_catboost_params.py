"""Cluster 14 regression: fast_model reduces a catboost-style template (iterations / depth / rsm) as well as the sklearn-style one (audit 2026-11-20, 14-F4)."""

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter import _fast_model_params


class _Cb:
    """A catboost-like template: iterations, depth, rsm."""

    iterations = 1000


class _Sk:
    """An sklearn-like template: n_estimators, max_depth, colsample_bytree."""

    n_estimators = 400


def test_catboost_names_are_reduced():
    """iterations is cut to a quarter (and by the cap), depth and rsm are set (14-F4)."""
    fast = _fast_model_params(_Cb(), {"iterations", "depth", "rsm", "learning_rate"}, None)
    assert fast == {"iterations": 250, "depth": 3, "rsm": 0.5}
    capped = _fast_model_params(_Cb(), {"iterations", "depth"}, 100)
    assert capped["iterations"] == 100


def test_sklearn_names_behave_as_before():
    """n_estimators is cut to a quarter and max_depth / colsample_bytree / subsample are set (14-F4)."""
    fast = _fast_model_params(_Sk(), {"n_estimators", "max_depth", "colsample_bytree", "subsample"}, None)
    assert fast == {"n_estimators": 100, "max_depth": 3, "colsample_bytree": 0.5, "subsample": 0.7}


def test_a_template_without_budget_params_is_left_alone():
    """A template exposing none of the names gets no overrides (14-F4)."""
    assert _fast_model_params(object(), {"alpha"}, None) == {}
