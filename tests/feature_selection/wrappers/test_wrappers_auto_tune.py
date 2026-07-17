"""Tests for ``RFECV(auto_tune=True)`` (TODO A / Wave 6 prelim, 2026-05-28).

Verifies the DataFingerprint extractor + the rule-based suggester
+ the wiring into RFECV.fit. The rule body itself is a stop-gap until
the synthetic-bench-trained classifier lands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import Ridge

from mlframe.feature_selection.wrappers import RFECV
from mlframe.feature_selection.wrappers._auto_tune import (
    DataFingerprint,
    suggest_configs,
    explain_suggestion,
)


# ----------------------------------------------------------------------- DataFingerprint


class TestDataFingerprint:
    def test_binary_detected(self):
        X, y = make_classification(n_samples=100, n_features=5, n_informative=3, random_state=0)
        fp = DataFingerprint.from_xy(X, y)
        assert fp.target_type == "binary"
        assert fp.n_rows == 100
        assert fp.n_features == 5

    def test_multiclass_detected(self):
        X, y = make_classification(n_samples=200, n_features=6, n_informative=4, n_classes=4, n_clusters_per_class=1, random_state=0)
        fp = DataFingerprint.from_xy(X, y)
        assert fp.target_type == "multiclass"

    def test_regression_detected(self):
        X, y = make_regression(n_samples=200, n_features=5, n_informative=3, random_state=0)
        fp = DataFingerprint.from_xy(X, y)
        assert fp.target_type == "regression"

    def test_p_n_ratio(self):
        X, y = make_classification(n_samples=50, n_features=200, n_informative=10, random_state=0)
        fp = DataFingerprint.from_xy(X, y)
        assert fp.p_n_ratio == pytest.approx(4.0)

    def test_continuous_floats_not_flagged_as_high_card(self):
        X, y = make_regression(n_samples=200, n_features=10, random_state=0)
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        fp = DataFingerprint.from_xy(Xdf, y)
        assert fp.frac_high_card == pytest.approx(0.0)

    def test_int_id_flagged_as_high_card(self):
        rng = np.random.default_rng(0)
        Xdf = pd.DataFrame(
            {
                "real": rng.normal(size=200),
                "hash_id": rng.integers(0, 1_000_000, size=200, dtype=np.int64),
            }
        )
        y = (Xdf["real"] > 0).astype(int).values
        fp = DataFingerprint.from_xy(Xdf, y)
        assert fp.frac_high_card == pytest.approx(0.5)

    def test_imbalanced_binary_imbalance(self):
        # 95/5 imbalance.
        y = np.array([0] * 190 + [1] * 10)
        X = np.random.default_rng(0).normal(size=(200, 4))
        fp = DataFingerprint.from_xy(X, y)
        assert fp.target_imbalance < 0.1


# ----------------------------------------------------------------------- suggest_configs


class TestSuggestConfigs:
    def test_high_p_n_triggers_prescreen(self):
        X, y = make_classification(n_samples=100, n_features=300, n_informative=10, random_state=0)
        fp = DataFingerprint.from_xy(X, y)
        _sc, _fic, _rc = suggest_configs(fp)
        assert "univariate_ht" in (_rc.prescreen or "")

    def test_tiny_p_init_design_2(self):
        X, y = make_classification(n_samples=200, n_features=6, n_informative=3, random_state=0)
        fp = DataFingerprint.from_xy(X, y)
        _sc, _, _ = suggest_configs(fp)
        assert _sc.init_design_size == 2

    def test_flat_curve_one_se_max(self):
        # Synthetic with no real signal -> max_corr ~ 0 -> flat curve rule.
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 8))
        y = rng.normal(size=200)
        fp = DataFingerprint.from_xy(X, y)
        # max_corr should be small.
        if fp.max_abs_corr_to_y < 0.3:
            _, _fic, _ = suggest_configs(fp)
            assert _fic.n_features_selection_rule == "one_se_max"

    def test_explain_suggestion_returns_str(self):
        X, y = make_classification(n_samples=100, n_features=5, random_state=0)
        fp = DataFingerprint.from_xy(X, y)
        s = explain_suggestion(fp)
        assert isinstance(s, str)
        assert "DataFingerprint" in s


# ----------------------------------------------------------------------- E2E with auto_tune=True


class TestRFECVAutoTune:
    def test_auto_tune_smoke_regression(self):
        X, y = make_regression(n_samples=200, n_features=10, n_informative=4, noise=0.5, random_state=0)
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        sel = RFECV(estimator=Ridge(), auto_tune=True, cv=3, max_refits=4, random_state=0)
        sel.fit(Xdf, y)
        assert hasattr(sel, "auto_tune_decision_")
        assert "fingerprint" in sel.auto_tune_decision_
        assert "explanation" in sel.auto_tune_decision_
        assert "applied" in sel.auto_tune_decision_
        assert sel.n_features_ >= 1

    def test_auto_tune_respects_explicit_user_kwarg(self):
        # User explicitly sets convergence_tol; auto-tune may also want to set it
        # for flat curves but must NOT override the explicit value.
        X, y = make_regression(n_samples=200, n_features=10, random_state=0, noise=100)
        sel = RFECV(
            estimator=Ridge(),
            auto_tune=True,
            cv=3,
            max_refits=4,
            random_state=0,
            convergence_tol=1e-6,  # user explicit, very tight tol
        )
        sel.fit(X, y)
        assert sel.convergence_tol == pytest.approx(1e-6)

    def test_auto_tune_off_by_default(self):
        sel = RFECV(estimator=Ridge())
        assert sel.auto_tune is False
