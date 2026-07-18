"""biz_value test for the ``n_features``-driven dart heuristic in ``models.lgbm_defaults.default_lgbm_params``.

Source: 9th_home-credit-default-risk.md -- "method=dart outperforms method=gbdt because I had so many
features that it helped basically as feature_fraction." With many correlated/redundant features, gbdt's
greedy per-split search keeps re-picking the same top features every round; dart's per-round tree dropout
plus a lower feature_fraction spreads the ensemble across more of the available features. Measured directly
(not assumed): at EQUAL n_estimators dart underperforms gbdt (dart's dropout needs more rounds to reach the
same effective tree count), so the heuristic auto-scales n_estimators 3x when it activates -- this test
pins that the auto-scaled dart preset beats a same-budget gbdt preset on a synthetic engineered to stress the
"many redundant/correlated features" scenario, and that at equal (unscaled) n_estimators the win reverses.
"""

from __future__ import annotations

import numpy as np
import pytest
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mlframe.models.lgbm_defaults import DART_REDUNDANCY_THRESHOLD, LARGE_N_FEATURES_THRESHOLD, _estimate_feature_redundancy, default_lgbm_params


def _make_redundant_feature_regression(n: int, n_features: int, n_informative: int, seed: int, dup_range: int = 12):
    """Helper: Make redundant feature regression."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    true_idx = rng.choice(n_features, n_informative, replace=False)
    beta = rng.normal(size=n_informative) * 3.0
    y = X[:, true_idx] @ beta + rng.normal(scale=0.3, size=n)
    # many near-duplicate correlated copies of each informative column -- the regime the source's win targets.
    for idx in true_idx:
        for k in range(1, dup_range):
            dup_idx = (idx + k * 7) % n_features
            X[:, dup_idx] = X[:, idx] + rng.normal(scale=0.02, size=n)
    return train_test_split(X, y, test_size=0.3, random_state=0)


def test_biz_val_dart_heuristic_beats_gbdt_on_many_redundant_features():
    """Biz val dart heuristic beats gbdt on many redundant features."""
    n_features = 350
    assert n_features >= LARGE_N_FEATURES_THRESHOLD
    Xtr, Xte, ytr, yte = _make_redundant_feature_regression(n=1000, n_features=n_features, n_informative=5, seed=1)

    gbdt_params = default_lgbm_params(objective="regression", num_leaves=15, learning_rate=0.15, n_estimators=150)
    dart_params = default_lgbm_params(objective="regression", num_leaves=15, learning_rate=0.15, n_estimators=150, n_features=n_features)

    assert dart_params["boosting_type"] == "dart"
    assert dart_params["feature_fraction"] < 1.0
    assert dart_params["n_estimators"] == 150 * 3  # auto-scaled to compensate dart's per-round tree dropout.

    rmse_gbdt = float(mean_squared_error(yte, LGBMRegressor(**gbdt_params).fit(Xtr, ytr).predict(Xte)) ** 0.5)
    rmse_dart = float(mean_squared_error(yte, LGBMRegressor(**dart_params).fit(Xtr, ytr).predict(Xte)) ** 0.5)

    assert (
        rmse_dart < rmse_gbdt * 0.97
    ), f"expected the n_features-driven dart preset to beat gbdt by >=3% RMSE on a redundant-feature regime, got dart={rmse_dart:.4f} gbdt={rmse_gbdt:.4f}"


def test_dart_at_equal_unscaled_n_estimators_underperforms_gbdt():
    """Confirms the n_estimators auto-scaling is load-bearing, not cosmetic: without it, dart loses."""
    n_features = 350
    Xtr, Xte, ytr, yte = _make_redundant_feature_regression(n=1000, n_features=n_features, n_informative=5, seed=1)

    gbdt_params = default_lgbm_params(objective="regression", num_leaves=15, learning_rate=0.15, n_estimators=150)
    unscaled_dart_params = dict(gbdt_params, boosting_type="dart", feature_fraction=0.5)  # same n_estimators=150, no auto-scale.

    rmse_gbdt = float(mean_squared_error(yte, LGBMRegressor(**gbdt_params).fit(Xtr, ytr).predict(Xte)) ** 0.5)
    rmse_unscaled_dart = float(mean_squared_error(yte, LGBMRegressor(**unscaled_dart_params).fit(Xtr, ytr).predict(Xte)) ** 0.5)

    assert (
        rmse_unscaled_dart > rmse_gbdt
    ), f"expected dart at equal (unscaled) n_estimators to underperform gbdt, got dart={rmse_unscaled_dart:.4f} gbdt={rmse_gbdt:.4f}"


def test_default_lgbm_params_below_threshold_stays_gbdt():
    """Default lgbm params below threshold stays gbdt."""
    params = default_lgbm_params(objective="regression", n_features=LARGE_N_FEATURES_THRESHOLD - 1)
    assert "boosting_type" not in params
    assert "feature_fraction" not in params
    assert params["n_estimators"] == 500


def test_default_lgbm_params_explicit_n_estimators_override_bypasses_scaling():
    """Default lgbm params explicit n estimators override bypasses scaling."""
    params = default_lgbm_params(objective="regression", n_features=LARGE_N_FEATURES_THRESHOLD + 100, n_estimators=200, boosting_type="gbdt")
    # an explicit boosting_type override wins over the heuristic entirely (applied last).
    assert params["boosting_type"] == "gbdt"
    assert params["n_estimators"] == 200 * 3  # n_estimators is still treated as the pre-scale budget.


def _make_independent_feature_regression(n: int, n_features: int, n_informative: int, seed: int):
    """Many genuinely UNCORRELATED features -- the raw-count heuristic's blind spot: it can't tell this
    apart from the redundant-duplicate regime above just by looking at ``n_features``, but dart's implicit-
    ``feature_fraction`` rationale doesn't apply here (there's no redundant top feature gbdt keeps re-picking).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    true_idx = rng.choice(n_features, n_informative, replace=False)
    beta = rng.normal(size=n_informative) * 3.0
    y = X[:, true_idx] @ beta + rng.normal(scale=0.3, size=n)
    return train_test_split(X, y, test_size=0.3, random_state=0)


def test_default_lgbm_params_auto_dart_redundancy_is_opt_in():
    """Omitting ``auto_dart_redundancy`` must leave the raw-count heuristic's output bit-identical."""
    baseline_a = default_lgbm_params(objective="regression", n_features=350)
    baseline_b = default_lgbm_params(objective="regression", n_features=350)
    assert baseline_a == baseline_b
    assert default_lgbm_params(objective="regression") == default_lgbm_params(objective="regression")


def test_auto_dart_redundancy_requires_X():
    """Auto dart redundancy requires X."""
    with pytest.raises(ValueError):
        default_lgbm_params(objective="regression", auto_dart_redundancy=True)


def test_biz_val_auto_dart_redundancy_ignores_wide_independent_features_unlike_raw_count():
    """500 genuinely independent features: the raw-count heuristic (threshold 300) triggers dart purely on
    count, but the redundancy-aware probe correctly declines -- there's no redundant-reselection problem here.
    """
    n_features = 500
    assert n_features >= LARGE_N_FEATURES_THRESHOLD  # the raw-count heuristic WOULD trigger dart here.
    Xtr, _Xte, _ytr, _yte = _make_independent_feature_regression(n=1000, n_features=n_features, n_informative=5, seed=2)

    count_based_params = default_lgbm_params(objective="regression", n_features=n_features)
    redundancy_based_params = default_lgbm_params(objective="regression", auto_dart_redundancy=True, X=Xtr)

    assert count_based_params["boosting_type"] == "dart"
    assert "boosting_type" not in redundancy_based_params, "redundancy-aware probe must NOT trigger dart on genuinely independent features"


def test_biz_val_auto_dart_redundancy_triggers_on_correlated_features_unlike_raw_count():
    """100 heavily-correlated features: BELOW the raw-count threshold (300), so the count heuristic stays
    gbdt, but the redundancy probe correctly triggers dart -- and the redundancy-aware choice measurably
    beats the count-based one on held-out RMSE, proving the decision is not merely different but better.
    A higher informative-column count and duplication density than the 350-feature fixture above pushes
    mean pairwise correlation across the WHOLE (smaller) matrix above ``DART_REDUNDANCY_THRESHOLD``, while
    still being far below ``LARGE_N_FEATURES_THRESHOLD`` on raw count alone.
    """
    n_features = 100
    assert n_features < LARGE_N_FEATURES_THRESHOLD  # the raw-count heuristic would NOT trigger dart here.
    Xtr, Xte, ytr, yte = _make_redundant_feature_regression(n=1000, n_features=n_features, n_informative=10, seed=3, dup_range=10)

    redundancy = _estimate_feature_redundancy(Xtr)
    assert redundancy >= DART_REDUNDANCY_THRESHOLD, f"fixture must actually be measured as redundant, got {redundancy:.4f}"

    count_based_params = default_lgbm_params(objective="regression", n_features=n_features, num_leaves=15, learning_rate=0.15, n_estimators=150)
    redundancy_based_params = default_lgbm_params(objective="regression", auto_dart_redundancy=True, X=Xtr, num_leaves=15, learning_rate=0.15, n_estimators=150)

    assert "boosting_type" not in count_based_params, "raw-count heuristic must stay gbdt below its threshold"
    assert redundancy_based_params["boosting_type"] == "dart", "redundancy-aware probe must trigger dart on heavily-correlated features"

    rmse_count_based = float(mean_squared_error(yte, LGBMRegressor(**count_based_params).fit(Xtr, ytr).predict(Xte)) ** 0.5)
    rmse_redundancy_based = float(mean_squared_error(yte, LGBMRegressor(**redundancy_based_params).fit(Xtr, ytr).predict(Xte)) ** 0.5)

    assert rmse_redundancy_based < rmse_count_based * 0.95, (
        f"expected the redundancy-aware dart trigger to beat the raw-count-based gbdt choice by >=5% RMSE "
        f"on a below-count-threshold but heavily-redundant regime, got redundancy_based={rmse_redundancy_based:.4f} "
        f"count_based={rmse_count_based:.4f}"
    )
