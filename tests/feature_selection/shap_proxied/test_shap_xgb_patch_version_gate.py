"""Regression: the shap-XGB base_score workaround must NOT clobber the
module-global ``float`` in shap's tree explainer on shap >= 0.52.

shap 0.52 parses the XGBoost 2.x/3.x base_score array natively and uses
``float`` as a numpy dtype (``np.asarray(base_score, dtype=float)``). The
legacy workaround set ``shap.explainers._tree.float = _safe_float``, which
turned that into ``dtype=<function _safe_float>`` and raised
``TypeError: Cannot interpret ... as a data type`` for every TreeExplainer.
"""

import builtins

import pytest

shap = pytest.importorskip("shap")

from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_explain as spe


def test_patch_is_noop_on_shap_ge_052():
    """Patch is noop on shap ge 052."""
    ver = tuple(int(p) for p in shap.__version__.split(".")[:2])
    if ver < (0, 52):
        pytest.skip("workaround is expected to apply on shap < 0.52")

    from shap.explainers import _tree as _shap_tree

    # shap's _tree module does not define a module-global ``float`` -- the legacy
    # patch was what introduced one. So the invariant is: the patch must NOT
    # leave a non-builtin ``float`` attribute behind on shap >= 0.52.
    saved_flag = spe._SHAP_XGB_PATCHED
    had_attr = "float" in _shap_tree.__dict__
    saved_attr = _shap_tree.__dict__.get("float")
    try:
        spe._SHAP_XGB_PATCHED = False
        _shap_tree.__dict__.pop("float", None)
        spe._maybe_patch_shap_xgb_base_score()
        assert _shap_tree.__dict__.get("float", builtins.float) is builtins.float, (
            "patch clobbered shap._tree.float on shap >= 0.52; np.asarray(base_score, dtype=float) will break"
        )
    finally:
        if had_attr:
            _shap_tree.float = saved_attr
        else:
            _shap_tree.__dict__.pop("float", None)
        spe._SHAP_XGB_PATCHED = saved_flag


def test_xgb_treeexplainer_smoke_no_dtype_error():
    """Xgb treeexplainer smoke no dtype error."""
    xgb = pytest.importorskip("xgboost")
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = xgb.XGBClassifier(n_estimators=8, max_depth=3, verbosity=0)
    model.fit(X, y)

    spe._maybe_patch_shap_xgb_base_score()
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    phi = explainer.shap_values(X, check_additivity=False)
    assert np.asarray(phi).shape[0] == X.shape[0]
