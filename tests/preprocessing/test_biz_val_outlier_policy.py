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
    class FakeLGBMClassifier:
        pass

    class FakeXGBRegressor:
        pass

    class FakeLinearRegression:
        pass

    assert is_tree_based_model(FakeLGBMClassifier()) is True
    assert is_tree_based_model(FakeXGBRegressor()) is True
    assert is_tree_based_model(FakeLinearRegression()) is False
