"""Wave 5 (2026-05-28) literature-extension regression tests for RFECV.

Covers:
  - L5 : ``stability_vs_n_curve_`` returns per-N stability across CV folds + ``n_stability_elbow_``.
  - L3 : ``knockoff_importance(w_statistic=...)`` routes to TreeSHAP / coef / gain.
  - L6 : multi-output y rejected with helpful error.
  - L1 / L2 : ``importance_getter='boruta_shap' / 'powershap'`` raise informative ImportError when libs missing.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from mlframe.feature_selection.wrappers import RFECV
from mlframe.feature_selection.wrappers._helpers import (
    knockoff_importance,
    make_gaussian_knockoffs,
    get_feature_importances,
)


# ----------------------------------------------------------------------- L5


class TestStabilityCurve:
    def test_curve_and_elbow_smoke(self):
        X, y = make_classification(n_samples=300, n_features=10, n_informative=5, random_state=0)
        rfecv = RFECV(estimator=LogisticRegression(max_iter=300), cv=3, max_refits=4, random_state=0)
        rfecv.fit(X, y)
        curve = rfecv.stability_vs_n_curve_()
        assert isinstance(curve, dict)
        for n, s in curve.items():
            assert 0.0 <= s <= 1.0
        elbow = rfecv.n_stability_elbow_()
        assert elbow >= 0

    def test_curve_dice_metric(self):
        X, y = make_classification(n_samples=300, n_features=10, n_informative=5, random_state=0)
        rfecv = RFECV(estimator=LogisticRegression(max_iter=300), cv=3, max_refits=4, random_state=0)
        rfecv.fit(X, y)
        curve_j = rfecv.stability_vs_n_curve_(metric="jaccard")
        curve_d = rfecv.stability_vs_n_curve_(metric="dice")
        # Both should have same keys (per-N).
        assert set(curve_j.keys()) == set(curve_d.keys())

    def test_invalid_metric_raises(self):
        X, y = make_classification(n_samples=100, n_features=5, n_informative=3, random_state=0)
        rfecv = RFECV(estimator=LogisticRegression(max_iter=200), cv=3, max_refits=2, random_state=0)
        rfecv.fit(X, y)
        with pytest.raises(ValueError, match="metric must be"):
            rfecv.stability_vs_n_curve_(metric="bogus")


# ----------------------------------------------------------------------- L3


class TestKnockoffWStatistic:
    def test_w_gain_routes_to_feature_importances(self):
        X, y = make_classification(n_samples=200, n_features=8, n_informative=4, random_state=0)
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
        W = knockoff_importance(
            model_factory=lambda: RandomForestClassifier(n_estimators=20, random_state=0),
            X=Xdf,
            y=y,
            random_state=0,
            w_statistic="gain",
        )
        assert isinstance(W, dict)
        assert len(W) == 8

    def test_w_coef_routes_to_coef(self):
        X, y = make_classification(n_samples=200, n_features=8, n_informative=4, random_state=0)
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
        W = knockoff_importance(
            model_factory=lambda: LogisticRegression(max_iter=300),
            X=Xdf,
            y=y,
            random_state=0,
            w_statistic="coef",
        )
        assert len(W) == 8

    def test_w_auto_picks_shap_for_tree(self):
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed; w_statistic='auto' for tree falls back at runtime")
        X, y = make_classification(n_samples=200, n_features=6, n_informative=3, random_state=0)
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
        W = knockoff_importance(
            model_factory=lambda: RandomForestClassifier(n_estimators=20, random_state=0),
            X=Xdf,
            y=y,
            random_state=0,
            w_statistic="auto",
        )
        assert len(W) == 6


# ----------------------------------------------------------------------- L6


class TestMultiOutputGuard:
    def test_2d_y_default_union_selects_per_target(self):
        # The friendly default (multioutput_strategy='union') fits one single-target RFECV per output column and
        # OR-aggregates support_ instead of crashing on a 2D y. Pins the default so the friendly behaviour can't regress.
        X, y = make_regression(n_samples=100, n_features=6, n_targets=3, random_state=0)
        rfecv = RFECV(estimator=Ridge(), cv=3, max_refits=2)
        rfecv.fit(X, y)
        assert getattr(rfecv, "multioutput_strategy_", None) == "union"
        assert rfecv.support_.shape[0] == 6 and rfecv.support_.dtype == bool
        assert len(getattr(rfecv, "multioutput_supports_", {})) == 3

    def test_2d_y_optout_raises_informative(self):
        # Opting OUT (multioutput_strategy=None) restores the historical clear NotImplementedError at fit entry, BEFORE
        # the deep sklearn error, with an actionable message: the offending y shape AND the per-target-loop + union(OR)
        # recipe. Pins both so a future refactor cannot regress to the opaque sklearn error.
        X, y = make_regression(n_samples=80, n_features=5, n_targets=2, random_state=1)
        rfecv = RFECV(estimator=Ridge(), cv=2, max_refits=2, multioutput_strategy=None)
        with pytest.raises(NotImplementedError) as ei:
            rfecv.fit(X, y)
        msg = str(ei.value)
        assert "(80, 2)" in msg, f"message must report the offending y shape; got: {msg}"
        assert "per target" in msg and "support_" in msg, f"message must give per-target/union guidance; got: {msg}"

    def test_2d_y_one_constant_column_does_not_abort_others(self):
        # F6: a single degenerate output column (all-constant target -> single-class crash in the sub-fit) must NOT
        # abort the other valid columns. The failed column is skipped + recorded; aggregation proceeds over the rest.
        X, y_good = make_classification(n_samples=200, n_features=6, n_informative=4, random_state=3)
        y = np.column_stack([y_good, np.zeros(X.shape[0], dtype=int)])  # 2nd column all-constant -> single class
        rfecv = RFECV(estimator=LogisticRegression(max_iter=300), cv=3, max_refits=2, multioutput_strategy="union")
        rfecv.fit(X, y)  # pre-fix: the constant column's single-class crash propagated and aborted ALL columns
        assert rfecv.support_.dtype == bool and rfecv.support_.shape[0] == 6
        assert bool(rfecv.support_.any()), "the valid output column should still drive a non-empty selection"
        assert "y0" in rfecv.multioutput_supports_, "the valid column must be fitted and aggregated"
        assert "y1" in rfecv.multioutput_skipped_, "the constant column must be recorded as skipped, not abort the fit"


class TestObjectDtypeNaNScan:
    def test_object_dtype_ndarray_nan_is_imputed_not_passed_to_core(self):
        # F5: an object-dtype ndarray carrying embedded float('nan') was NOT scanned (kind 'O' fell through), so the
        # NaN reached the linear core and crashed. The policy now coerces object arrays and median-imputes them.
        X_num, y = make_classification(n_samples=120, n_features=5, n_informative=3, random_state=7)
        X = X_num.astype(object)
        X[0, 0] = float("nan")
        X[5, 2] = float("nan")
        rfecv = RFECV(estimator=LogisticRegression(max_iter=300), cv=3, max_refits=2)
        rfecv.fit(X, y)  # pre-fix: object-dtype NaN unscanned -> "Input X contains NaN" from validate_data
        assert rfecv.support_.dtype == bool and rfecv.support_.shape[0] == 5

    def test_object_dtype_ndarray_nan_raise_policy_detects(self):
        # The 'raise' policy must also SEE object-dtype NaN; pre-fix it silently passed through (scan skipped kind 'O').
        X_num, y = make_classification(n_samples=80, n_features=4, n_informative=2, random_state=8)
        X = X_num.astype(object)
        X[3, 1] = float("nan")
        rfecv = RFECV(estimator=LogisticRegression(max_iter=300), cv=2, max_refits=2, nan_in_X_policy="raise")
        with pytest.raises(ValueError, match="nan_in_X_policy='raise'"):
            rfecv.fit(X, y)


# ----------------------------------------------------------------------- L1 / L2


class TestBorutaShapPowerSHAPRouting:
    def test_boruta_shap_without_lib_raises_import(self):
        try:
            import BorutaShap  # noqa: F401

            try:
                from arfs.feature_selection import GrootCV  # noqa: F401

                pytest.skip("BorutaShap or arfs is installed; opt-out of the missing-import test")
            except ImportError:
                pytest.skip("BorutaShap is installed; opt-out of the missing-import test")
        except ImportError:
            pass
        X, y = make_classification(n_samples=80, n_features=4, n_informative=2, random_state=0)
        with pytest.raises(ImportError, match="BorutaShap"):
            get_feature_importances(
                model=RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y),
                current_features=list(range(4)),
                data=X,
                target=y,
                importance_getter="boruta_shap",
            )

    def test_prescreen_callable_restricts_universe(self):
        # L7: user-supplied callable prescreen filters the candidate set before MBH loop.
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=0)
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])

        # Prescreen keeps first 4 features only.
        def my_prescreen(X, y):
            return [f"f{i}" for i in range(4)]

        rfecv = RFECV(
            estimator=LogisticRegression(max_iter=300),
            cv=3,
            max_refits=3,
            random_state=0,
            prescreen=my_prescreen,
        )
        rfecv.fit(Xdf, y)
        # Selected features MUST be a subset of {f0..f3}.
        selected = set(rfecv.get_feature_names_out())
        assert selected.issubset({f"f{i}" for i in range(4)})

    def test_prescreen_unknown_string_warns_and_noops(self, caplog):
        X, y = make_classification(n_samples=100, n_features=6, n_informative=3, random_state=0)
        rfecv = RFECV(
            estimator=LogisticRegression(max_iter=300),
            cv=3,
            max_refits=2,
            random_state=0,
            prescreen="not_a_real_prescreen",
            verbose=1,
        )
        with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.wrappers.rfecv"):
            rfecv.fit(X, y)
        assert any("Unknown prescreen" in r.getMessage() for r in caplog.records)

    def test_mrmr_prescreen_restricts_universe(self):
        """``prescreen='mrmr'`` runs the existing MRMR filter as a pre-pass and
        restricts the MBH search universe to the top-K it returns. Closes
        the long-standing TODO #1 (mRMR pre-screening for p >> n problems)."""
        X, y = make_classification(n_samples=300, n_features=30, n_informative=8, random_state=0)
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(30)])
        sel = RFECV(
            estimator=LogisticRegression(max_iter=300),
            cv=3,
            max_refits=3,
            random_state=0,
            prescreen="mrmr",
            prescreen_top_k=10,
        )
        sel.fit(Xdf, y)
        # Selected features must come from the prescreen's top-K subset.
        selected = set(sel.get_feature_names_out())
        assert len(selected) <= 10

    def test_prescreen_callable_failure_falls_back(self, caplog):
        X, y = make_classification(n_samples=100, n_features=6, n_informative=3, random_state=0)
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])

        def bad_prescreen(X, y):
            raise RuntimeError("boom")

        rfecv = RFECV(
            estimator=LogisticRegression(max_iter=300),
            cv=3,
            max_refits=2,
            random_state=0,
            prescreen=bad_prescreen,
            verbose=1,
        )
        with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.wrappers.rfecv"):
            rfecv.fit(Xdf, y)
        # Fit must complete despite prescreen failure.
        assert rfecv.n_features_ >= 1
        assert any("Prescreen callable failed" in r.getMessage() for r in caplog.records)

    def test_shap_oof_alias_routes_to_shap(self):
        # 'shap_oof' is an alias of 'shap' that makes the semantic explicit.
        # If shap is not installed, both raise ImportError with the same module.
        try:
            import shap  # noqa: F401

            pytest.skip("shap is installed; this test verifies the missing-import path")
        except ImportError:
            pass
        X, y = make_classification(n_samples=80, n_features=4, n_informative=2, random_state=0)
        with pytest.raises(ImportError, match="shap"):
            get_feature_importances(
                model=RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y),
                current_features=list(range(4)),
                data=X,
                target=y,
                importance_getter="shap_oof",
            )

    def test_powershap_without_lib_raises_import(self):
        try:
            import powershap  # noqa: F401

            pytest.skip("powershap is installed; opt-out of the missing-import test")
        except ImportError:
            pass
        X, y = make_classification(n_samples=80, n_features=4, n_informative=2, random_state=0)
        with pytest.raises(ImportError, match="powershap"):
            get_feature_importances(
                model=RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y),
                current_features=list(range(4)),
                data=X,
                target=y,
                importance_getter="powershap",
            )
