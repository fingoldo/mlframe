"""Tests for Phase 8 RFECV improvements.

Covered:
- __sklearn_tags__ delegates to inner estimator's classifier/regressor flag
- cv auto-detect: TimeSeriesSplit on monotonic DatetimeIndex
- cv_results_df_ property returns a tabular DataFrame view
- swap_top_k: opt-in truncated SFFS final-pass swap
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit, KFold, StratifiedKFold

from mlframe.feature_selection.wrappers import RFECV


# ----------------------------------------------------------------------------
# __sklearn_tags__
# ----------------------------------------------------------------------------
class TestSklearnTags:
    def test_classifier_estimator_marks_classifier(self):
        rfecv = RFECV(estimator=LogisticRegression(max_iter=200))
        tags = rfecv.__sklearn_tags__()
        assert tags.estimator_type == "classifier", (
            f"RFECV around a classifier should report estimator_type='classifier'; got {tags.estimator_type}"
        )

    def test_regressor_estimator_marks_regressor(self):
        rfecv = RFECV(estimator=Ridge())
        tags = rfecv.__sklearn_tags__()
        assert tags.estimator_type == "regressor", (
            f"RFECV around a regressor should report estimator_type='regressor'; got {tags.estimator_type}"
        )

    def test_multi_estimator_uses_first(self):
        """When ``estimators=[clf1, clf2]`` is set, tags follow the first."""
        rfecv = RFECV(
            estimator=None,
            estimators=[LogisticRegression(max_iter=200), LogisticRegression(max_iter=200)],
        )
        tags = rfecv.__sklearn_tags__()
        assert tags.estimator_type == "classifier"

    def test_no_estimator_returns_default_tags(self):
        """Bare RFECV() (no estimator yet) should still return a Tags dataclass."""
        rfecv = RFECV()
        tags = rfecv.__sklearn_tags__()
        # Type check only - default should be the parent's tags (no crash).
        assert tags is not None
        assert hasattr(tags, "estimator_type")


# ----------------------------------------------------------------------------
# cv auto-detect: TimeSeriesSplit on DatetimeIndex
# ----------------------------------------------------------------------------
class TestCvAutoDetect:
    def test_datetime_index_triggers_timeseries_split(self, caplog):
        """X with monotonic DatetimeIndex should auto-select TimeSeriesSplit.
        Uses a regression target so class-imbalance on the first temporal
        fold isn't an issue (TSS gives an early fold that may have zero
        instances of one class on shuffle=False classification data)."""
        import logging
        n = 200
        X, y = make_regression(
            n_samples=n, n_features=5, n_informative=2,
            random_state=0, shuffle=False, noise=1.0,
        )
        Xdf = pd.DataFrame(
            X,
            columns=[f"f{i}" for i in range(5)],
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        rfecv = RFECV(
            estimator=LinearRegression(),
            cv=3, max_refits=3, verbose=1, random_state=0,
        )
        with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.wrappers._rfecv"):
            rfecv.fit(Xdf, y)
        assert any(
            "DatetimeIndex" in rec.getMessage() for rec in caplog.records
        ), f"DatetimeIndex auto-detect log not seen; got: {[r.getMessage() for r in caplog.records]}"

    def test_non_monotonic_datetime_index_does_not_trigger_tss(self, caplog):
        """If the DatetimeIndex is shuffled (not monotonic), KFold should be used."""
        import logging
        n = 200
        X, y = make_regression(
            n_samples=n, n_features=5, n_informative=2,
            random_state=0, shuffle=False, noise=1.0,
        )
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        rng = np.random.default_rng(0)
        Xdf = pd.DataFrame(
            X,
            columns=[f"f{i}" for i in range(5)],
            index=idx[rng.permutation(n)],
        )
        rfecv = RFECV(
            estimator=LinearRegression(),
            cv=3, max_refits=3, verbose=1, random_state=0,
        )
        with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.wrappers._rfecv"):
            rfecv.fit(Xdf, y)
        assert not any(
            "DatetimeIndex" in rec.getMessage() for rec in caplog.records
        ), "Non-monotonic DatetimeIndex should NOT trigger TSS auto-detect"

    def test_explicit_cv_overrides_auto_detect(self):
        """Passing cv=KFold(...) explicitly disables the TSS auto-detect."""
        n = 200
        X, y = make_regression(
            n_samples=n, n_features=5, n_informative=2,
            random_state=0, shuffle=False, noise=1.0,
        )
        Xdf = pd.DataFrame(
            X,
            columns=[f"f{i}" for i in range(5)],
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        explicit_cv = KFold(n_splits=3, shuffle=False)
        rfecv = RFECV(
            estimator=LinearRegression(),
            cv=explicit_cv, max_refits=3, verbose=0, random_state=0,
        )
        rfecv.fit(Xdf, y)
        assert rfecv.n_features_ >= 1


# ----------------------------------------------------------------------------
# cv_results_df_ DataFrame property
# ----------------------------------------------------------------------------
class TestCvResultsDataFrame:
    def test_property_returns_dataframe(self):
        X, y = make_classification(
            n_samples=100, n_features=5, n_informative=2,
            n_redundant=0, random_state=0, shuffle=False, class_sep=2.0,
        )
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        rfecv = RFECV(
            estimator=LogisticRegression(max_iter=200, random_state=0),
            cv=3, max_refits=4, verbose=0, random_state=0,
        ).fit(Xdf, y)
        df = rfecv.cv_results_df_
        assert isinstance(df, pd.DataFrame)
        assert set(["nfeatures", "cv_mean_perf", "cv_std_perf"]).issubset(df.columns)

    def test_columns_match_dict_keys(self):
        X, y = make_classification(
            n_samples=100, n_features=5, n_informative=2,
            n_redundant=0, random_state=0, shuffle=False, class_sep=2.0,
        )
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        rfecv = RFECV(
            estimator=LogisticRegression(max_iter=200, random_state=0),
            cv=3, max_refits=4, verbose=0, random_state=0,
        ).fit(Xdf, y)
        df = rfecv.cv_results_df_
        # Each column must round-trip with the dict version
        for col in df.columns:
            assert df[col].tolist() == list(rfecv.cv_results_[col]), col

    def test_property_raises_before_fit(self):
        rfecv = RFECV(estimator=LogisticRegression())
        with pytest.raises(ValueError, match="cv_results_df_"):
            _ = rfecv.cv_results_df_

    def test_dict_access_still_works(self):
        """Backward-compat: cv_results_['nfeatures'] keeps returning the list."""
        X, y = make_classification(
            n_samples=100, n_features=5, n_informative=2,
            n_redundant=0, random_state=0, shuffle=False, class_sep=2.0,
        )
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        rfecv = RFECV(
            estimator=LogisticRegression(max_iter=200, random_state=0),
            cv=3, max_refits=4, verbose=0, random_state=0,
        ).fit(Xdf, y)
        nfs = rfecv.cv_results_["nfeatures"]
        assert isinstance(nfs, list)
        assert all(isinstance(n, (int, np.integer)) for n in nfs)


# ----------------------------------------------------------------------------
# swap_top_k: truncated SFFS final-pass swap
# ----------------------------------------------------------------------------
class TestSffsSwap:
    def test_default_disabled(self):
        """swap_top_k=0 (default) must NOT call _sffs_swap_pass."""
        X, y = make_classification(
            n_samples=120, n_features=8, n_informative=3,
            n_redundant=0, random_state=0, shuffle=False, class_sep=2.0,
        )
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
        rfecv = RFECV(
            estimator=LogisticRegression(max_iter=200, random_state=0),
            cv=3, max_refits=4, verbose=0, random_state=0,
            # swap_top_k default = 0
        ).fit(Xdf, y)
        # Smoke: completed without error
        assert rfecv.n_features_ >= 1

    def test_swap_top_k_triggers_pass_without_crash(self, caplog):
        """swap_top_k > 0 must run the swap pass and complete.
        Uses a synthetic where the model actually beats the dummy
        baseline so the early-stop path doesn't bypass the swap pass.
        """
        import logging
        X, y = make_classification(
            n_samples=400, n_features=8, n_informative=5,
            n_redundant=0, random_state=0, shuffle=False, class_sep=3.0,
        )
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
        rfecv = RFECV(
            estimator=LogisticRegression(max_iter=300, random_state=0),
            cv=3, max_refits=6, verbose=1, random_state=0,
            swap_top_k=3,
        )
        with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.wrappers._rfecv"):
            rfecv.fit(Xdf, y)
        msgs = [rec.getMessage() for rec in caplog.records]
        assert any("SFFS swap pass" in m for m in msgs), (
            f"SFFS swap-pass summary log not seen; got: {msgs[-10:]}"
        )
        assert rfecv.n_features_ >= 1

    def test_swap_score_monotone_better_or_equal(self):
        """The swap pass should never make the best CV score worse than
        what the main loop found - swaps are accepted only on strict
        improvement."""
        X, y = make_classification(
            n_samples=400, n_features=8, n_informative=5,
            n_redundant=0, random_state=0, shuffle=False, class_sep=3.0,
        )
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
        baseline = RFECV(
            estimator=LogisticRegression(max_iter=300, random_state=0),
            cv=3, max_refits=6, verbose=0, random_state=0,
        ).fit(Xdf, y)
        swapped = RFECV(
            estimator=LogisticRegression(max_iter=300, random_state=0),
            cv=3, max_refits=6, verbose=0, random_state=0,
            swap_top_k=3,
        ).fit(Xdf, y)
        baseline_max = max(baseline.cv_results_["cv_mean_perf"])
        swapped_max = max(swapped.cv_results_["cv_mean_perf"])
        assert swapped_max >= baseline_max - 1e-9, (
            f"swap_top_k=3 produced a worse best score than disabled: "
            f"baseline_max={baseline_max:.4f}, swapped_max={swapped_max:.4f}"
        )

    def test_swap_top_k_with_regression(self):
        """Smoke: regression target works through the swap pass."""
        X, y = make_regression(
            n_samples=300, n_features=6, n_informative=3,
            random_state=0, shuffle=False, noise=1.0,
        )
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
        rfecv = RFECV(
            estimator=LinearRegression(),
            cv=3, max_refits=4, verbose=0, random_state=0,
            swap_top_k=2,
        ).fit(Xdf, y)
        assert rfecv.n_features_ >= 1
