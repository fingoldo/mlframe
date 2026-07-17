"""Edge-case audit P0 fixes (RFECV + MRMR).

Each test corresponds to one finding from the audit report. P0 = silent
miscompute / confusion that production pipelines could hit. Fixes were
landed in the same commit; tests below pin the new behaviour.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression

from mlframe.feature_selection.wrappers import RFECV
from mlframe.feature_selection.filters.mrmr import MRMR


# ----------------------------------------------------------------------------
# A1: single-class y -> ValueError at fit entry (was: opaque sklearn error)
# ----------------------------------------------------------------------------
class TestA1_SingleClassY:
    def test_all_zero_y_raises_value_error(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((100, 5)))
        y = np.zeros(100, dtype=int)
        with pytest.raises(ValueError, match="unique class"):
            RFECV(estimator=LogisticRegression(max_iter=100)).fit(X, y)

    def test_all_one_y_raises(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((100, 5)))
        with pytest.raises(ValueError, match="unique class"):
            RFECV(estimator=LogisticRegression(max_iter=100)).fit(X, np.ones(100, int))


# ----------------------------------------------------------------------------
# A2/A3: NaN / Inf in y -> ValueError (was: silent miscompute)
# ----------------------------------------------------------------------------
class TestA2_NaNInfInY:
    def test_nan_in_y_raises(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((90, 5)))
        y = np.array([1.0, 0.0, np.nan] * 30)
        with pytest.raises(ValueError, match="NaN"):
            RFECV(estimator=LogisticRegression(max_iter=100)).fit(X, y)

    def test_inf_in_y_raises(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((100, 4)))
        y = np.array([0.5, 1.0, 2.0, np.inf] * 25)
        with pytest.raises(ValueError, match="inf"):
            RFECV(estimator=LinearRegression()).fit(X, y)


# ----------------------------------------------------------------------------
# A5: minority class < n_splits -> ValueError (was: half-fitted state)
# ----------------------------------------------------------------------------
class TestA5_MinorityClassValidation:
    def test_extreme_imbalance_raises(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((300, 5)))
        # Only 2 positives - cv=3 cannot stratify.
        y = np.zeros(300, int)
        y[:2] = 1
        with pytest.raises(ValueError, match="Minority class"):
            RFECV(estimator=LogisticRegression(max_iter=100), cv=3).fit(X, y)


# ----------------------------------------------------------------------------
# C18: leakage_action='exclude' auto-drops corr=1.0 leak columns
# ----------------------------------------------------------------------------
class TestC18_LeakageAction:
    def _make_problem_with_leak(self):
        rng = np.random.default_rng(0)
        n = 200
        X = pd.DataFrame(rng.standard_normal((n, 4)), columns=list("abcd"))
        y = (X["a"] > 0).astype(int).values
        # Near-perfect leak (corr ~ 0.999)
        X["leak"] = y.astype(float) + rng.standard_normal(n) * 0.01
        return X, y

    def test_exclude_drops_leak_column(self):
        X, y = self._make_problem_with_leak()
        rfecv = RFECV(
            estimator=LogisticRegression(max_iter=200, random_state=0),
            cv=3,
            max_refits=2,
            verbose=0,
            leakage_corr_threshold=0.9,
            leakage_action="exclude",
        )
        rfecv.fit(X, y)
        assert "leak" not in list(rfecv.get_feature_names_out())

    def test_raise_aborts_fit(self):
        X, y = self._make_problem_with_leak()
        with pytest.raises(ValueError, match="leakage"):
            RFECV(
                estimator=LogisticRegression(max_iter=200),
                cv=3,
                max_refits=2,
                verbose=0,
                leakage_corr_threshold=0.9,
                leakage_action="raise",
            ).fit(X, y)

    def test_warn_is_legacy_default(self, caplog):
        X, y = self._make_problem_with_leak()
        with caplog.at_level(logging.WARNING):
            rfecv = RFECV(
                estimator=LogisticRegression(max_iter=200),
                cv=3,
                max_refits=2,
                verbose=0,
                leakage_corr_threshold=0.9,
                # leakage_action default is 'warn'
            )
            rfecv.fit(X, y)
        # Leak column may still be in support_ (legacy behaviour) but warning fired.
        assert any("leakage" in r.getMessage().lower() for r in caplog.records)


# ----------------------------------------------------------------------------
# F30: must_include + must_exclude intersection -> clear ValueError
# ----------------------------------------------------------------------------
class TestF30_MustIncludeExcludeIntersection:
    def test_intersection_raises_with_both_param_names(self):
        X = pd.DataFrame(np.random.default_rng(0).standard_normal((100, 5)), columns=list("abcde"))
        y = (X["a"] > 0).astype(int).values
        with pytest.raises(ValueError, match="must_include and must_exclude"):
            RFECV(
                estimator=LogisticRegression(max_iter=100),
                must_include=["a", "b"],
                must_exclude=["a", "c"],
            ).fit(X, y)


# ----------------------------------------------------------------------------
# G34: NotFittedError on transform-before-fit (sklearn convention)
# ----------------------------------------------------------------------------
class TestG34_NotFittedError:
    def test_rfecv_transform_unfitted_raises(self):
        rfecv = RFECV(estimator=LogisticRegression(max_iter=100))
        with pytest.raises(NotFittedError):
            rfecv.transform(pd.DataFrame(np.random.default_rng(0).standard_normal((10, 5))))

    def test_mrmr_transform_unfitted_raises(self):
        mrmr = MRMR()
        with pytest.raises(NotFittedError):
            mrmr.transform(pd.DataFrame(np.random.default_rng(0).standard_normal((10, 5))))


# ----------------------------------------------------------------------------
# G35: MRMR.transform raises on column drift (symmetric with RFECV)
# ----------------------------------------------------------------------------
class TestG35_MRMRTransformColumnDrift:
    def test_drift_raises_runtime_error(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((200, 6)), columns=[f"f{i}" for i in range(6)])
        y = (X["f0"] > 0).astype(int).values
        mrmr = MRMR()
        mrmr.fit(X, y)
        # Drop a column the selector relies on.
        X_drift = X.drop(columns=mrmr.feature_names_in_[:1])
        with pytest.raises(RuntimeError, match="missing from input X"):
            mrmr.transform(X_drift)
