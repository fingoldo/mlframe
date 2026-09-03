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
    had_attr = "float" in _shap_tree.__dict__
    saved_attr = _shap_tree.__dict__.get("float")
    try:
        _shap_tree.__dict__.pop("float", None)
        with spe._maybe_patch_shap_xgb_base_score():
            assert (
                _shap_tree.__dict__.get("float", builtins.float) is builtins.float
            ), "patch clobbered shap._tree.float on shap >= 0.52; np.asarray(base_score, dtype=float) will break"
    finally:
        if had_attr:
            _shap_tree.float = saved_attr
        else:
            _shap_tree.__dict__.pop("float", None)


def test_patch_restores_shap_tree_float_after_the_with_block():
    """CI regression: a plain function call left ``shap.explainers._tree.float`` monkeypatched to
    ``_safe_float`` for the rest of the process, so an unrelated LATER ``TreeExplainer`` construction on
    a DIFFERENT model (LightGBM, not XGBoost -- reproduced live in
    ``test_shap_proxy_treeshap_lightgbm.py::test_lightgbm_interaction_parity``) that happened to run in
    the same pytest session hit shap's shared ``_tree`` module's now-patched ``float`` name and crashed
    with ``AttributeError: 'TreeEnsemble' object has no attribute 'values'`` (``_safe_float`` is a plain
    function, not a type, and something inside shap's tree-loading machinery needs ``float`` usable as a
    numpy dtype). The context-manager form must restore (or remove) the name on exit so no later,
    unrelated ``TreeExplainer`` call in the same process is affected."""
    ver = tuple(int(p) for p in shap.__version__.split(".")[:2])
    if ver >= (0, 52):
        pytest.skip("the patch is a no-op on shap >= 0.52 -- nothing to restore")

    from shap.explainers import _tree as _shap_tree

    had_attr_before = "float" in _shap_tree.__dict__
    saved_attr_before = _shap_tree.__dict__.get("float")
    try:
        _shap_tree.__dict__.pop("float", None)
        with spe._maybe_patch_shap_xgb_base_score():
            assert (
                _shap_tree.__dict__.get("float") is spe._SafeFloatCallable
            ), "patch must install _SafeFloatCallable (not the plain _safe_float function) inside the with block on shap < 0.52"
        assert "float" not in _shap_tree.__dict__, "patch must remove its own float name after the with block exits (none was present before)"
    finally:
        if had_attr_before:
            _shap_tree.float = saved_attr_before
        else:
            _shap_tree.__dict__.pop("float", None)


def test_xgb_treeexplainer_smoke_no_dtype_error():
    """Xgb treeexplainer smoke no dtype error."""
    xgb = pytest.importorskip("xgboost")
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = xgb.XGBClassifier(n_estimators=8, max_depth=3, verbosity=0)
    model.fit(X, y)

    with spe._maybe_patch_shap_xgb_base_score():
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        phi = explainer.shap_values(X, check_additivity=False)
    assert np.asarray(phi).shape[0] == X.shape[0]
