"""Wave 4 (2026-05-28) edge-cases / robustness regression tests for RFECV.

Covers E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, C4, C8, C9, C12.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import make_regression
from sklearn.linear_model import LogisticRegression, Ridge

from mlframe.feature_selection.wrappers import RFECV
from mlframe.feature_selection.wrappers._helpers import make_gaussian_knockoffs

# ----------------------------------------------------------------------- E5


class TestHighCardinalityIntDetector:
    """Groups tests covering TestHighCardinalityIntDetector."""
    def test_hicard_int_column_warns(self, caplog):
        """Hicard int column warns."""
        rng = np.random.default_rng(0)
        n = 200
        X = pd.DataFrame(
            {
                "real": rng.normal(size=n),
                "hash_id": rng.integers(0, 1_000_000, size=n).astype(np.int64),
            }
        )
        y = (X["real"] > 0).astype(int).values
        rfecv = RFECV(
            estimator=LogisticRegression(max_iter=200),
            cv=3,
            max_refits=2,
            verbose=1,
            random_state=0,
        )
        with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.wrappers.rfecv"):
            rfecv.fit(X, y)
        assert any(
            "cardinality" in rec.getMessage() and "ID" in rec.getMessage() for rec in caplog.records
        ), f"Expected high-card warning; got: {[r.getMessage() for r in caplog.records[-5:]]}"


# ----------------------------------------------------------------------- E6


class TestNearConstantFilter:
    """Groups tests covering TestNearConstantFilter."""
    def test_near_constant_float_dropped(self):
        """Near constant float dropped."""
        rng = np.random.default_rng(0)
        n = 200
        X = pd.DataFrame(
            {
                "real": rng.normal(size=n),
                "near_const": 1.0 + rng.normal(scale=1e-16, size=n),  # var ~ 1e-32 < 1e-12
            }
        )
        y = (X["real"] > 0).astype(int).values
        rfecv = RFECV(
            estimator=LogisticRegression(max_iter=200),
            cv=3,
            max_refits=2,
            verbose=0,
            random_state=0,
        )
        rfecv.fit(X, y)
        assert "near_const" not in rfecv.feature_names_in_


# ----------------------------------------------------------------------- E9


class TestEstimatorsTypeFamilyAssert:
    """Groups tests covering TestEstimatorsTypeFamilyAssert."""
    def test_mix_classifier_regressor_rejected(self):
        """Mix classifier regressor rejected."""
        with pytest.raises(ValueError, match="classifier and regressor"):
            RFECV(estimators=[LogisticRegression(), Ridge()])

    def test_pure_classifier_list_ok(self):
        """Pure classifier list ok."""
        RFECV(estimators=[LogisticRegression(), LogisticRegression(C=0.1)])

    def test_pure_regressor_list_ok(self):
        """Pure regressor list ok."""
        RFECV(estimators=[Ridge(), Ridge(alpha=0.5)])


# ----------------------------------------------------------------------- E10


class TestKnockoffNumericOnly:
    """Groups tests covering TestKnockoffNumericOnly."""
    def test_non_numeric_column_raises(self):
        """Non numeric column raises."""
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0], "cat": ["a", "b", "c"]})
        with pytest.raises(ValueError, match="non-numeric"):
            make_gaussian_knockoffs(X, random_state=0)


# ----------------------------------------------------------------------- E12


class TestStabilitySelectionFloor:
    """Groups tests covering TestStabilitySelectionFloor."""
    def test_n_below_20_raises(self):
        """N below 20 raises."""
        X = np.random.default_rng(0).normal(size=(15, 4))
        y = np.array([0, 1] * 7 + [0])
        rfecv = RFECV(
            estimator=LogisticRegression(max_iter=200),
            stability_selection=True,
            stability_n_bootstrap=10,
        )
        with pytest.raises(ValueError, match="n_samples >= 20"):
            rfecv.fit(X, y)


# ----------------------------------------------------------------------- E13


class TestNaTInDatetimeIndex:
    """Groups tests covering TestNaTInDatetimeIndex."""
    def test_nat_disables_temporal_autodetect(self, caplog):
        # Direct call into _resolve_cv_and_val_cv: NaT in DatetimeIndex must
        # warn and disable the temporal auto-detect.
        """Nat disables temporal autodetect."""
        from mlframe.feature_selection.wrappers.rfecv._cv_setup import _resolve_cv_and_val_cv

        rng = np.random.default_rng(0)
        n = 60
        dates = pd.to_datetime(["2024-01-01"] * n) + pd.to_timedelta(range(n), unit="D")
        dates = list(dates)
        dates[30] = pd.NaT
        X = pd.DataFrame(rng.normal(size=(n, 3)), index=pd.DatetimeIndex(dates))
        y = (X.iloc[:, 0] > 0).astype(int).values
        with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.wrappers.rfecv"):
            cv, _, _ = _resolve_cv_and_val_cv(
                cv=3,
                X=X,
                y=y,
                groups=None,
                estimator=LogisticRegression(),
                cv_shuffle=False,
                random_state=0,
                fit_params={},
                early_stopping_val_nsplits=None,
                early_stopping_rounds=None,
                _polars_time_series_hint=False,
                verbose=1,
            )
        from sklearn.model_selection import TimeSeriesSplit

        assert not isinstance(cv, TimeSeriesSplit)
        assert any("NaT" in rec.getMessage() for rec in caplog.records)


# ----------------------------------------------------------------------- E14


class TestOMPNumThreadsNotSet:
    """E14 reverted post-bench: _pin_threads_to_one must NOT clobber the global
    OMP_NUM_THREADS env var (leak across tests breaks LightGBM split-finder).
    """

    def test_pin_threads_does_not_set_omp(self, monkeypatch):
        """Pin threads does not set omp."""
        from mlframe.feature_selection.wrappers._helpers import _pin_threads_to_one

        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        _pin_threads_to_one(LogisticRegression())
        assert os.environ.get("OMP_NUM_THREADS") is None


# ----------------------------------------------------------------------- E15


class TestMustExcludeTypoRaises:
    """Groups tests covering TestMustExcludeTypoRaises."""
    def test_default_raises_on_typo(self):
        """Default raises on typo."""
        X = pd.DataFrame(np.random.default_rng(0).normal(size=(100, 4)), columns=list("abcd"))
        y = (X["a"] > 0).astype(int).values
        rfecv = RFECV(
            estimator=LogisticRegression(max_iter=200),
            cv=3,
            max_refits=2,
            must_exclude=["typo_name"],
        )
        with pytest.raises(ValueError, match="must_exclude"):
            rfecv.fit(X, y)

    def test_strict_false_silently_ignores(self):
        """Strict false silently ignores."""
        X = pd.DataFrame(np.random.default_rng(0).normal(size=(100, 4)), columns=list("abcd"))
        y = (X["a"] > 0).astype(int).values
        rfecv = RFECV(
            estimator=LogisticRegression(max_iter=200),
            cv=3,
            max_refits=2,
            must_exclude=["typo_name"],
            must_exclude_strict=False,
        )
        rfecv.fit(X, y)
        assert rfecv.n_features_ >= 1


# ----------------------------------------------------------------------- C12


class TestOneSeDirection:
    """Groups tests covering TestOneSeDirection."""
    def test_direction_max_returns_largest_in_band(self):
        """Direction max returns largest in band."""
        X, y = make_regression(n_samples=200, n_features=10, n_informative=4, random_state=0)
        rfecv = RFECV(estimator=Ridge(), cv=3, max_refits=6, random_state=0)
        rfecv.fit(X, y)
        n_min = rfecv.n_features_one_se_(direction="min")
        n_max = rfecv.n_features_one_se_(direction="max")
        assert n_max >= n_min

    def test_invalid_direction_raises(self):
        """Invalid direction raises."""
        rfecv = RFECV(estimator=Ridge())
        with pytest.raises(ValueError, match="direction"):
            rfecv.n_features_one_se_(direction="bogus")
