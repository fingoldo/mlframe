"""Anti-mask-via-guard meta-tests for RFECV defaults.

Lesson from F38 (importance_getter='auto' crashed silently on LinearRegression
for years): the prior test_wrappers.py had a per-estimator workaround
``importance_getter='coef_' if name=='LinearRegression' else 'feature_importances_'``
that masked the dispatch bug instead of reporting it. Default-args path was
NEVER tested. Without these tests every common-path regression hides in plain
sight.

This file enforces that EVERY supported estimator (sklearn linear, sklearn
trees, optionally CatBoost / XGBoost / LightGBM) works with RFECV using ONLY
default arguments, on a small synthetic problem. No per-estimator overrides
allowed - if a test needs an override to pass, the override IS the bug.

Add a new estimator: append it to ``DEFAULT_CLF_ESTIMATORS`` /
``DEFAULT_REG_ESTIMATORS`` and watch the meta-test gate it.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso,
    SGDClassifier,
)

from mlframe.feature_selection.wrappers import RFECV


# ----------------------------------------------------------------------------
# Estimator registries. Anything that's a routinely-used sklearn estimator
# should be in these lists; anything that fails is a bug, not a "configure
# importance_getter manually" workaround opportunity.
# ----------------------------------------------------------------------------
DEFAULT_CLF_ESTIMATORS = [
    ("LogisticRegression",   lambda: LogisticRegression(max_iter=400, random_state=0)),
    ("RandomForest",         lambda: RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=1)),
    ("SGDClassifier",        lambda: SGDClassifier(max_iter=200, random_state=0, loss="log_loss")),
]

DEFAULT_REG_ESTIMATORS = [
    ("LinearRegression",     lambda: LinearRegression()),
    ("Ridge",                lambda: Ridge(random_state=0)),
    ("Lasso",                lambda: Lasso(alpha=0.1, max_iter=1000)),
    ("RandomForestReg",      lambda: RandomForestRegressor(n_estimators=20, random_state=0, n_jobs=1)),
]


@pytest.fixture(scope="module")
def small_clf_data():
    X, y = make_classification(
        n_samples=200, n_features=12, n_informative=4,
        n_redundant=0, n_classes=2, n_clusters_per_class=1,
        random_state=0, shuffle=False,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(12)]), y


@pytest.fixture(scope="module")
def small_reg_data():
    X, y = make_regression(
        n_samples=200, n_features=12, n_informative=4,
        noise=0.5, random_state=0, shuffle=False,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(12)]), y


# ----------------------------------------------------------------------------
# Anti-mask test 1: every classifier works with DEFAULT RFECV args
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("name, factory", DEFAULT_CLF_ESTIMATORS, ids=[n for n, _ in DEFAULT_CLF_ESTIMATORS])
def test_default_args_classifier(small_clf_data, name, factory):
    """Construct RFECV with ONLY (estimator=...) and small overrides for
    runtime (cv, max_refits, verbose). NO importance_getter override - that's
    the dispatch path F38 patched. NO scoring override. Default plotting.
    Default optimum search. Default voting. If any of these fail in
    combination, this test fails."""
    X, y = small_clf_data
    rfecv = RFECV(
        estimator=factory(),
        cv=3,            # short for CI
        max_refits=4,    # short for CI
        verbose=0,
    )
    rfecv.fit(X, y)
    assert rfecv.n_features_ >= 1, f"{name} produced empty support_ on default args"
    # transform must round-trip
    out = rfecv.transform(X)
    assert out.shape[1] == rfecv.n_features_
    assert out.shape[0] == X.shape[0]


@pytest.mark.parametrize("name, factory", DEFAULT_REG_ESTIMATORS, ids=[n for n, _ in DEFAULT_REG_ESTIMATORS])
def test_default_args_regressor(small_reg_data, name, factory):
    """Same as above for regressors. Critically includes LinearRegression /
    Ridge / Lasso - the F38 victims. With the prior dispatch they all hit
    AttributeError on getattr(model, 'feature_importances_')."""
    X, y = small_reg_data
    rfecv = RFECV(
        estimator=factory(),
        cv=3,
        max_refits=4,
        verbose=0,
    )
    rfecv.fit(X, y)
    assert rfecv.n_features_ >= 1, f"{name} produced empty support_ on default args"
    out = rfecv.transform(X)
    assert out.shape[1] == rfecv.n_features_
    assert out.shape[0] == X.shape[0]


# ----------------------------------------------------------------------------
# Anti-mask test 2: existing test files should not contain per-estimator
# importance_getter workarounds. If they do, that's a hint that the dispatch
# is broken AND the test author chose to mask rather than report.
# ----------------------------------------------------------------------------
def test_no_per_estimator_importance_getter_workaround_in_main_tests():
    """Scan tests/feature_selection/test_wrappers.py for the pattern
    ``importance_getter = 'coef_' if name == 'LinearRegression' else ...``
    or any conditional that picks importance_getter based on the estimator
    name. The right fix is to call RFECV with default importance_getter=None
    and let the dispatch resolve at fit time (F38). If you genuinely need
    a callable getter for SHAP / permutation, pass a Callable - that's a
    different code path that the auto dispatch correctly does not touch.
    """
    test_path = Path(__file__).parent / "test_wrappers.py"
    if not test_path.exists():
        pytest.skip("test_wrappers.py not present in this layout")
    text = test_path.read_text(encoding="utf-8", errors="replace")

    # Pattern 1: ternary that picks importance_getter based on estimator name string
    pattern_ternary = re.compile(
        r"importance_getter\s*=\s*['\"]\w+_['\"]\s+if\s+\w+\s*==\s*['\"]\w+['\"]\s+else\s+['\"]\w+_['\"]"
    )
    matches_ternary = pattern_ternary.findall(text)

    # Pattern 2: explicit dict keyed by estimator name -> getter string
    pattern_dict = re.compile(
        r"\{['\"]\w+['\"]\s*:\s*['\"]coef_['\"]\s*,\s*['\"]\w+['\"]\s*:\s*['\"]feature_importances_['\"]"
    )
    matches_dict = pattern_dict.findall(text)

    if matches_ternary or matches_dict:
        offenders = matches_ternary + matches_dict
        pytest.fail(
            "Found per-estimator importance_getter workarounds in test_wrappers.py:\n"
            + "\n".join(f"  - {m}" for m in offenders[:5])
            + "\n"
            "The default importance_getter=None ('auto') dispatch should "
            "handle every estimator type. If a test needs an override, the "
            "override IS the bug - update the dispatch in get_feature_importances "
            "instead of paper-mill´ing it in the test."
        )


# ----------------------------------------------------------------------------
# Anti-mask test 3: RFECV's __init__ defaults must produce a fittable model.
# Detects accidental "this default never worked" regressions.
# ----------------------------------------------------------------------------
def test_init_defaults_are_fittable(small_clf_data):
    """RFECV(estimator=LR()) with no other overrides must fit on standard
    pandas+numpy input. The prior optimizer_plotting=None default ran
    plt.show() and blocked pytest forever - the default was effectively
    'works only in an interactive session'. Other defaults should also
    work in a fresh-import + cold-fit setting."""
    X, y = small_clf_data
    # Class-stratified subsample to 80 rows: a head-slice [:80] of
    # shuffle=False data has near-zero class diversity in the first half
    # and trips LR's two-class-required check. We want a small problem
    # that still has both classes.
    rng = np.random.default_rng(0)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    take = 40
    sel = np.concatenate([
        rng.choice(pos_idx, size=min(take, len(pos_idx)), replace=False),
        rng.choice(neg_idx, size=min(take, len(neg_idx)), replace=False),
    ])
    rng.shuffle(sel)
    X_small = X.iloc[sel].reset_index(drop=True)
    y_small = y[sel]

    rfecv = RFECV(estimator=LogisticRegression(max_iter=200, random_state=0))
    rfecv.fit(X_small, y_small)
    assert rfecv.n_features_ >= 1


# ----------------------------------------------------------------------------
# Anti-mask test 4: every OptimumSearch enum value either WORKS or raises
# NotImplementedError. None should silently produce a wrong result.
# ----------------------------------------------------------------------------
def test_every_optimum_search_value_either_works_or_errors(small_clf_data):
    """Pre-fix, calling RFECV with OptimumSearch.ScipyLocal hit
    UnboundLocalError 30 lines deep. F1 fix made it raise
    NotImplementedError at the first iteration. This test enumerates ALL
    enum members and asserts each one either:
      - completes a fit (the supported set), or
      - raises NotImplementedError (the explicitly unsupported set).
    Anything else - silent succeed-with-garbage, generic Python errors,
    crash mid-fold - is a regression."""
    from mlframe.feature_selection.wrappers import OptimumSearch

    X, y = small_clf_data
    # Stratified subsample (same reason as test_init_defaults_are_fittable)
    rng = np.random.default_rng(0)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    take = 30
    sel = np.concatenate([
        rng.choice(pos_idx, size=min(take, len(pos_idx)), replace=False),
        rng.choice(neg_idx, size=min(take, len(neg_idx)), replace=False),
    ])
    rng.shuffle(sel)
    X_small = X.iloc[sel].reset_index(drop=True)
    y_small = y[sel]

    for method in OptimumSearch:
        try:
            rfecv = RFECV(
                estimator=LogisticRegression(max_iter=200, random_state=0),
                cv=2,
                max_refits=2,
                top_predictors_search_method=method,
                verbose=0,
            )
            rfecv.fit(X_small, y_small)
            # If it fits, n_features_ must be reasonable.
            assert rfecv.n_features_ >= 1, f"{method} produced empty support_"
        except NotImplementedError as exc:
            # Acceptable - the dispatch explicitly rejected this method.
            assert method.value in str(exc), (
                f"NotImplementedError for {method} should name the method, got: {exc}"
            )
        except Exception as exc:  # pragma: no cover
            pytest.fail(
                f"OptimumSearch.{method.name} raised unexpected {type(exc).__name__}: {exc}\n"
                f"Either implement the dispatch in get_next_features_subset OR raise "
                f"NotImplementedError with the enum value at construction time."
            )
