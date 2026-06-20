"""Tests for the 2026-05-28 grouped pydantic configs + ``importance_getter='boruta'``.

Verifies:
  - SearchConfig / FIConfig / RobustnessConfig override matching flat kwargs.
  - Default-field config does NOT silently clobber explicit flat values.
  - Pydantic field validators reject invalid values.
  - importance_getter='boruta' returns shadow-relative importances.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier

from mlframe.feature_selection.wrappers import (
    RFECV, SearchConfig, FIConfig, RobustnessConfig,
)
from mlframe.feature_selection.wrappers._helpers import get_feature_importances


# ----------------------------------------------------------------------- Config wiring


class TestSearchConfig:
    def test_default_doesnt_clobber_flat(self):
        # Flat max_refits=100 must survive when only a default SearchConfig() is passed.
        r = RFECV(estimator=Ridge(), max_refits=100, search_config=SearchConfig())
        assert r.max_refits == 100

    def test_explicit_field_overrides_flat(self):
        r = RFECV(estimator=Ridge(), max_refits=100,
                  search_config=SearchConfig(max_refits=42))
        assert r.max_refits == 42

    def test_convergence_tol_via_config(self):
        r = RFECV(estimator=Ridge(),
                  search_config=SearchConfig(convergence_tol=1e-3, convergence_tol_window=7))
        assert r.convergence_tol == pytest.approx(1e-3)
        assert r.convergence_tol_window == 7

    def test_invalid_optimizer_target_rejected_by_pydantic(self):
        with pytest.raises(ValueError, match="optimizer_target"):
            SearchConfig(optimizer_target="bogus")

    def test_invalid_dichotomic_epsilon_rejected(self):
        with pytest.raises(ValueError):
            SearchConfig(dichotomic_epsilon=2.0)  # > 1.0 not allowed


class TestFIConfig:
    def test_fi_missing_policy_override(self):
        r = RFECV(estimator=Ridge(), fi_config=FIConfig(fi_missing_policy="median"))
        assert r.fi_missing_policy == "median"

    def test_decay_rate(self):
        r = RFECV(estimator=Ridge(), fi_config=FIConfig(fi_decay_rate=0.07))
        assert r.fi_decay_rate == pytest.approx(0.07)

    def test_invalid_fi_missing_policy_rejected(self):
        with pytest.raises(ValueError, match="fi_missing_policy"):
            FIConfig(fi_missing_policy="bogus")

    def test_invalid_rule_rejected(self):
        with pytest.raises(ValueError, match="n_features_selection_rule"):
            FIConfig(n_features_selection_rule="bogus")

    def test_invalid_coef_scale_source_rejected(self):
        with pytest.raises(ValueError, match="coef_scale_source"):
            FIConfig(coef_scale_source="bogus")


class TestRobustnessConfig:
    def test_leakage_action_override(self):
        r = RFECV(estimator=Ridge(),
                  robustness_config=RobustnessConfig(leakage_action="raise"))
        assert r.leakage_action == "raise"

    def test_must_exclude_strict_via_config(self):
        r = RFECV(estimator=Ridge(),
                  robustness_config=RobustnessConfig(must_exclude_strict=False))
        assert r.must_exclude_strict is False

    def test_invalid_leakage_action_rejected(self):
        with pytest.raises(ValueError, match="leakage_action"):
            RobustnessConfig(leakage_action="bogus")

    def test_prescreen_through_config(self):
        r = RFECV(estimator=Ridge(),
                  robustness_config=RobustnessConfig(prescreen="univariate_ht"))
        assert r.prescreen == "univariate_ht"


class TestEndToEndConfigDriven:
    def test_fit_with_all_three_configs(self):
        X, y = make_classification(n_samples=200, n_features=10, n_informative=4, random_state=0)
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        sel = RFECV(
            estimator=LogisticRegression(max_iter=200),
            search_config=SearchConfig(max_refits=4, init_design_size=3),
            fi_config=FIConfig(fi_missing_policy="worst",
                               n_features_selection_rule="one_se_max"),
            robustness_config=RobustnessConfig(leakage_corr_threshold=None),
        )
        sel.fit(Xdf, y)
        assert sel.n_features_ >= 1


# ----------------------------------------------------------------------- Boruta importance_getter


class TestBorutaImportanceGetter:
    def test_boruta_returns_shadow_relative_scores(self):
        X, y = make_classification(n_samples=200, n_features=10, n_informative=4, random_state=0)
        model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
        fi = get_feature_importances(
            model, current_features=list(range(10)),
            importance_getter="boruta", data=X, target=y,
        )
        assert len(fi) == 10
        # At least some informative features beat the shadow threshold.
        n_positive = sum(1 for v in fi.values() if v > 0)
        assert n_positive >= 2, f"Expected >= 2 features to beat shadow; got {n_positive}"

    def test_boruta_requires_data_and_target(self):
        from sklearn.ensemble import RandomForestClassifier as RFC
        m = RFC(n_estimators=5, random_state=0).fit(np.random.normal(size=(50, 4)), [0, 1] * 25)
        with pytest.raises(ValueError, match="data"):
            get_feature_importances(m, list(range(4)), "boruta", data=None, target=[0] * 50)

    def test_drop_column_importance_ground_truth(self):
        """drop-column importance is the oracle baseline; expensive but
        unambiguous. Verify it routes correctly for small p."""
        from sklearn.linear_model import Ridge
        X, y = make_regression(n_samples=200, n_features=5, n_informative=3, random_state=0, noise=0.1)
        model = Ridge().fit(X, y)
        fi = get_feature_importances(
            model, current_features=list(range(5)),
            importance_getter="drop_column", data=X, target=y,
        )
        assert len(fi) == 5
        # Informative columns (0..2) should produce LARGER score drops than noise (3, 4).
        assert max(fi[0], fi[1], fi[2]) > min(fi[3], fi[4])

    def test_drop_column_requires_data_target(self):
        from sklearn.linear_model import Ridge
        m = Ridge().fit(np.random.default_rng(0).normal(size=(50, 4)), np.arange(50) * 0.1)
        with pytest.raises(ValueError, match="data"):
            get_feature_importances(m, list(range(4)), "drop_column", data=None, target=np.arange(50))

    def test_rfecv_with_boruta_importance_getter(self):
        X, y = make_classification(n_samples=200, n_features=8, n_informative=4, random_state=0)
        sel = RFECV(
            estimator=RandomForestClassifier(n_estimators=10, random_state=0),
            importance_getter="boruta",
            cv=3, max_refits=3, random_state=0,
        )
        sel.fit(X, y)
        assert sel.n_features_ >= 1
