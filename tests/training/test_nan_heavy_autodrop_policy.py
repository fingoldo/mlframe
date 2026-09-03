"""A NaN-heavy column must not be dropped when every model in the run reads NaN natively.

Two defects surfaced by a production log:

- the rule label was the hardcoded string ``nan_heavy (>=50% missing)`` and kept claiming 50% long after the
  threshold constant moved to 0.99, so an operator reading the log concluded the change had never been applied;
- the drop fired at all on a CatBoost-only run. A gradient booster routes a missing value down a learned default
  branch, so an essentially-empty column still carries "was this field present" into the split. The missingness of
  a structural field (a deliverables blurb that only some postings have) is signal, and dropping it discards it.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from mlframe.training.core._main_train_suite_target_distribution import (
    _all_models_handle_nan,
    _maybe_auto_drop_after_feature_analyzer,
)
from mlframe.training.targets._target_distribution_analyzer import _NAN_FRACTION_THRESHOLD


def _frames():
    """Three aligned split frames with one NaN-heavy column and one low-variance column."""
    def make(n):
        """A frame with one all-null column and one constant column, at whatever row count the case needs."""
        return pd.DataFrame({"keep": range(n), "mostly_nan": [None] * n, "flat": [1] * n})
    return make(50), make(10), make(10)


def _report(nan_cols=("mostly_nan",), flat_cols=("flat",)):
    """A stand-in analyzer report: drop candidates plus the per-column warnings that name the rule."""
    warnings = {c: [f"nan_fraction=1.00 >= {_NAN_FRACTION_THRESHOLD}"] for c in nan_cols}
    warnings.update({c: ["low_variance=0.0"] for c in flat_cols})
    return SimpleNamespace(drop_candidates=list(nan_cols) + list(flat_cols), feature_warnings=warnings)


def _behavior(enabled: bool = True):
    """Behaviour config with the auto-drop knob at ``enabled``."""
    return SimpleNamespace(auto_drop_distribution_analyzer_candidates=enabled, auto_drop_near_duplicate_threshold=2.0)


class TestNanNativeDetection:
    """Which model sets count as reading NaN natively."""

    @pytest.mark.parametrize("models", [["cb"], ["lgb", "xgb"], ["cb", "hgb"], ["CB"]])
    def test_gradient_boosters_are_nan_native(self, models):
        """The whole GBDT family routes missing values internally; case is irrelevant."""
        assert _all_models_handle_nan(models) is True

    @pytest.mark.parametrize("models", [["cb", "linear"], ["mlp"], ["knn", "lgb"]])
    def test_a_single_non_native_model_disables_it(self, models):
        """One learner needing a finite matrix is enough to keep the drop on."""
        assert _all_models_handle_nan(models) is False

    @pytest.mark.parametrize("models", [None, [], ["something_new"]])
    def test_unknown_or_empty_is_conservative(self, models):
        """Not knowing what will be fitted must not silently disable a protective drop."""
        assert _all_models_handle_nan(models) is False


class TestDropPolicy:
    """What actually leaves the frames."""

    def test_nan_heavy_column_survives_a_catboost_only_run(self, caplog):
        """The defect: a CatBoost-only run dropped a column whose emptiness the model handles by design."""
        train, val, test = _frames()
        # The module logs under the SUITE logger, not its own module name, so the mini-HPT phase reads as one stream.
        with caplog.at_level(logging.INFO, logger="mlframe.training.core._main_train_suite"):
            train, val, test, dropped = _maybe_auto_drop_after_feature_analyzer(
                fd_report=_report(),
                train_df=train,
                val_df=val,
                test_df=test,
                behavior_config=_behavior(),
                metadata={},
                verbose=True,
                mlframe_models=["cb"],
            )
        assert "mostly_nan" in train.columns, "a NaN-native run must keep the NaN-heavy column"
        assert "mostly_nan" not in dropped
        assert any("consumes NaN natively" in r.getMessage() for r in caplog.records)

    def test_other_drop_reasons_still_apply(self):
        """Only the NaN rule is about NaN; low variance is still worth dropping for any model."""
        train, val, test = _frames()
        train, _val, _test, dropped = _maybe_auto_drop_after_feature_analyzer(
            fd_report=_report(),
            train_df=train,
            val_df=val,
            test_df=test,
            behavior_config=_behavior(),
            metadata={},
            verbose=False,
            mlframe_models=["cb"],
        )
        assert "flat" in dropped and "flat" not in train.columns

    def test_a_non_native_model_still_drops_the_nan_heavy_column(self):
        """With a linear model in the run the column has to go -- it cannot be fitted on NaN."""
        train, val, test = _frames()
        train, _val, _test, dropped = _maybe_auto_drop_after_feature_analyzer(
            fd_report=_report(),
            train_df=train,
            val_df=val,
            test_df=test,
            behavior_config=_behavior(),
            metadata={},
            verbose=False,
            mlframe_models=["cb", "linear"],
        )
        assert "mostly_nan" in dropped and "mostly_nan" not in train.columns

    def test_unknown_model_list_keeps_the_old_behaviour(self):
        """No model list = no licence to change what gets dropped."""
        train, val, test = _frames()
        _train, _val, _test, dropped = _maybe_auto_drop_after_feature_analyzer(
            fd_report=_report(),
            train_df=train,
            val_df=val,
            test_df=test,
            behavior_config=_behavior(),
            metadata={},
            verbose=False,
            mlframe_models=None,
        )
        assert "mostly_nan" in dropped


class TestTheLabelStatesTheRealThreshold:
    """The log line an operator reads has to match the constant the code applies."""

    def test_label_is_derived_not_hardcoded(self, caplog):
        """It said 50% while the threshold was 0.99; the number now comes from the constant itself."""
        train, val, test = _frames()
        # The module logs under the SUITE logger, not its own module name, so the mini-HPT phase reads as one stream.
        with caplog.at_level(logging.INFO, logger="mlframe.training.core._main_train_suite"):
            _maybe_auto_drop_after_feature_analyzer(
                fd_report=_report(),
                train_df=train,
                val_df=val,
                test_df=test,
                behavior_config=_behavior(),
                metadata={},
                verbose=True,
                mlframe_models=["linear"],  # keeps the nan rule active so the label is emitted
            )
        text = " ".join(r.getMessage() for r in caplog.records)
        assert f">={_NAN_FRACTION_THRESHOLD:.0%} missing" in text
        assert ">=50% missing" not in text or _NAN_FRACTION_THRESHOLD == 0.5
