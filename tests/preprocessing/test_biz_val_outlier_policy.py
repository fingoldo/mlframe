"""biz_value test for ``preprocessing.apply_outlier_policy`` / ``is_tree_based_model``.

The win: when the true label depends on the MAGNITUDE of an outlier value (a realistic fraud/finance
scenario -- only sufficiently extreme values are positive), naive quantile-capping ahead of a tree model
collapses that magnitude distinction and destroys most of the signal, while the tree-aware policy (leave raw
values untouched, add only an outlier-score flag) preserves it -- letting a LightGBM model split directly on
the raw magnitude.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mlframe.preprocessing.outlier_policy import apply_outlier_policy, is_tree_based_model


def test_biz_val_tree_aware_outlier_policy_beats_naive_capping():
    """Tree aware outlier policy beats naive capping."""
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    n = 5000
    x = rng.normal(0, 1, n)
    outlier_mask = rng.random(n) < 0.08
    x[outlier_mask] = rng.uniform(3, 30, int(outlier_mask.sum()))
    y = (x > 10).astype(int)
    y = np.where((x < 10) & (rng.random(n) < 0.02), 1, y)

    df = pd.DataFrame({"x": x})
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=0)

    model_probe = lgb.LGBMClassifier(n_estimators=100, verbosity=-1)
    assert is_tree_based_model(model_probe) is True
    assert is_tree_based_model(object()) is False

    capped_train = apply_outlier_policy(X_train, object(), cap_quantiles=(0.10, 0.90))
    capped_test = apply_outlier_policy(X_test, object(), cap_quantiles=(0.10, 0.90))
    model_capped = lgb.LGBMClassifier(n_estimators=100, verbosity=-1).fit(capped_train, y_train)
    auc_capped = roc_auc_score(y_test, model_capped.predict_proba(capped_test)[:, 1])

    flagged_train = apply_outlier_policy(X_train, model_probe)
    flagged_test = apply_outlier_policy(X_test, model_probe)
    assert "outlier_score" in flagged_train.columns
    assert np.array_equal(flagged_train["x"].to_numpy(), X_train["x"].to_numpy())  # raw values untouched
    model_flagged = lgb.LGBMClassifier(n_estimators=100, verbosity=-1).fit(flagged_train, y_train)
    auc_flagged = roc_auc_score(y_test, model_flagged.predict_proba(flagged_test)[:, 1])

    assert auc_flagged > auc_capped + 0.3, (
        f"tree-aware (uncapped + flag) policy should substantially beat naive capping when magnitude carries signal: "
        f"tree_aware={auc_flagged:.4f} capped={auc_capped:.4f}"
    )


def test_is_tree_based_model_detects_common_families():
    """Is tree based model detects common families."""
    class FakeLGBMClassifier:
        """Groups tests covering FakeLGBMClassifier."""
        pass

    class FakeXGBRegressor:
        """Groups tests covering FakeXGBRegressor."""
        pass

    class FakeLinearRegression:
        """Groups tests covering FakeLinearRegression."""
        pass

    assert is_tree_based_model(FakeLGBMClassifier()) is True
    assert is_tree_based_model(FakeXGBRegressor()) is True
    assert is_tree_based_model(FakeLinearRegression()) is False


def test_unwrap_pipeline_default_off_matches_prior_exact_behavior():
    """Default (unwrap_pipeline=False) must be bit-identical to the pre-extension behavior: a tree model
    hidden inside a Pipeline/meta-estimator wrapper is NOT detected unless the caller opts in."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline(steps=[("scaler", StandardScaler()), ("clf", RandomForestClassifier())])
    assert is_tree_based_model(pipe) is False
    assert is_tree_based_model(pipe, unwrap_pipeline=False) is False

    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(0, 1, 200)})
    out_default = apply_outlier_policy(X, pipe)
    out_explicit_off = apply_outlier_policy(X, pipe, unwrap_pipeline=False)
    assert "outlier_score" not in out_default.columns  # routed to capping, same as before this extension
    pd.testing.assert_frame_equal(out_default, out_explicit_off)


def test_is_tree_based_model_unwrap_pipeline_detects_wrapped_tree_estimators():
    """Coverage test: unwrap_pipeline=True must correctly classify several real (not fake) sklearn/xgboost/
    lightgbm estimators, including several hidden behind common meta-estimator wrappers, and must NOT
    misclassify a wrapped non-tree estimator as tree-based."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import BaggingRegressor, RandomForestClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cases = [
        (RandomForestClassifier(), True, "bare sklearn tree estimator"),
        (lgb.LGBMClassifier(), True, "bare lightgbm estimator"),
        (xgb.XGBRegressor(), True, "bare xgboost estimator"),
        (LinearRegression(), False, "bare linear estimator"),
        (Pipeline(steps=[("scaler", StandardScaler()), ("clf", lgb.LGBMClassifier())]), True, "lgbm inside Pipeline"),
        (Pipeline(steps=[("scaler", StandardScaler()), ("clf", LogisticRegression())]), False, "logreg inside Pipeline"),
        (CalibratedClassifierCV(estimator=RandomForestClassifier()), True, "random forest inside CalibratedClassifierCV"),
        (BaggingRegressor(estimator=xgb.XGBRegressor()), True, "xgboost inside BaggingRegressor"),
        (BaggingRegressor(estimator=LinearRegression()), False, "linear regression inside BaggingRegressor"),
    ]

    for model, expected, label in cases:
        assert is_tree_based_model(model, unwrap_pipeline=True) is expected, f"{label}: expected is_tree_based_model(..., unwrap_pipeline=True)={expected}"
        # without the opt-in, wrapped estimators are never detected (only bare ones are, since the wrapper
        # itself carries no tree-family marker in its own MRO)
        is_bare = "inside" not in label
        assert is_tree_based_model(model, unwrap_pipeline=False) is (expected if is_bare else False)


def test_apply_outlier_policy_unwrap_pipeline_routes_wrapped_tree_model_correctly():
    """biz_value-style coverage test: without unwrap_pipeline, a tree model hidden in a Pipeline is
    misrouted to the capping policy and loses outlier-magnitude signal (same failure mode as the original
    biz_value test, but triggered by a wrapper instead of a bare wrong-family model); unwrap_pipeline=True
    fixes the routing and recovers the signal."""
    import lightgbm as lgb
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    rng = np.random.default_rng(2)
    n = 5000
    x = rng.normal(0, 1, n)
    outlier_mask = rng.random(n) < 0.08
    x[outlier_mask] = rng.uniform(3, 30, int(outlier_mask.sum()))
    y = (x > 10).astype(int)
    y = np.where((x < 10) & (rng.random(n) < 0.02), 1, y)

    df = pd.DataFrame({"x": x})
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=0)

    wrapped_model = Pipeline(steps=[("noop", FunctionTransformer()), ("clf", lgb.LGBMClassifier(n_estimators=100, verbosity=-1))])
    assert is_tree_based_model(wrapped_model) is False  # confirms the misrouting exists pre-fix

    misrouted_train = apply_outlier_policy(X_train, wrapped_model, cap_quantiles=(0.10, 0.90))
    misrouted_test = apply_outlier_policy(X_test, wrapped_model, cap_quantiles=(0.10, 0.90))
    model_misrouted = lgb.LGBMClassifier(n_estimators=100, verbosity=-1).fit(misrouted_train, y_train)
    auc_misrouted = roc_auc_score(y_test, model_misrouted.predict_proba(misrouted_test)[:, 1])

    fixed_train = apply_outlier_policy(X_train, wrapped_model, cap_quantiles=(0.10, 0.90), unwrap_pipeline=True)
    fixed_test = apply_outlier_policy(X_test, wrapped_model, cap_quantiles=(0.10, 0.90), unwrap_pipeline=True)
    assert "outlier_score" in fixed_train.columns
    model_fixed = lgb.LGBMClassifier(n_estimators=100, verbosity=-1).fit(fixed_train, y_train)
    auc_fixed = roc_auc_score(y_test, model_fixed.predict_proba(fixed_test)[:, 1])

    assert auc_fixed > auc_misrouted + 0.14, (
        f"unwrap_pipeline=True should recover the tree-aware-policy win lost to misrouting: fixed={auc_fixed:.4f} misrouted={auc_misrouted:.4f}"
    )
