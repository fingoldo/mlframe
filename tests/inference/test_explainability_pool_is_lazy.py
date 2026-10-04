"""Explaining a LightGBM model built a CatBoost Pool over the whole frame and threw it away.

`_X = Pool(X, cat_features=...)` sat above the CV loop, but `_X` is read only inside the
`catboost_native_feature_importance=True` branch -- which is not the default. Every call on the default path
paid a whole-frame copy plus quantisation for a value nothing read, on frames this project sizes in the tens of
GB. The unconditional `from catboost import EFstrType, Pool` alongside it made catboost a hard import
requirement for explaining a model that has nothing to do with CatBoost.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pandas as pd
import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "mlframe" / "inference" / "explainability.py"


def _fn() -> ast.FunctionDef:
    """The `compute_shap_on_cv` definition."""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "compute_shap_on_cv")


def _guarded_by_the_native_flag(node: ast.AST, fn: ast.FunctionDef) -> bool:
    """True when `node` sits inside an `if catboost_native_feature_importance:` block."""
    for parent in ast.walk(fn):
        if isinstance(parent, ast.If) and isinstance(parent.test, ast.Name) and parent.test.id == "catboost_native_feature_importance":
            if any(n is node for stmt in parent.body for n in ast.walk(stmt)):
                return True
    return False


def test_the_pool_is_only_built_for_the_branch_that_reads_it():
    """The whole-frame copy must not happen on the default path."""
    fn = _fn()
    pools = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "Pool"]
    assert pools, "the Pool construction vanished entirely; the native-importance branch needs it"
    unguarded = [n for n in pools if not _guarded_by_the_native_flag(n, fn)]
    assert not unguarded, f"{len(unguarded)} Pool construction(s) still run on the default path"


def test_catboost_is_not_imported_unconditionally():
    """Explaining an XGBoost model must not hard-require catboost."""
    fn = _fn()
    top_level = [n for n in fn.body if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("catboost")]
    assert not top_level, "catboost is still imported at the top of compute_shap_on_cv"


def test_the_native_branch_still_has_what_it_needs():
    """The lazy import must actually cover both names the branch uses: the native path runs end to end on a CatBoost model."""
    pytest.importorskip("shap")
    pytest.importorskip("imblearn")
    catboost = pytest.importorskip("catboost")
    from sklearn.model_selection import KFold

    from mlframe.inference.explainability import compute_shap_on_cv

    rng = np.random.default_rng(0)
    n, f = 120, 4
    X = pd.DataFrame(rng.standard_normal((n, f)), columns=[f"f{i}" for i in range(f)])
    y = pd.Series((X["f0"] + 0.3 * rng.standard_normal(n) > 0).astype(int))
    values, _base, interactions, _ibase, predictions, _expected = compute_shap_on_cv(
        X,
        y,
        catboost.CatBoostClassifier,
        {"iterations": 5, "verbose": 0, "random_seed": 0, "thread_count": 1},
        KFold(n_splits=2, shuffle=True, random_state=0),
        catboost_native_feature_importance=True,
        show_oos_metrics=False,
        plot=False,
    )
    assert values.shape == (2, n, f + 1)
    assert interactions.shape == (2, n, f + 1, f + 1)
    assert predictions.shape == (2, n)
