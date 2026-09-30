"""How the training suite configures RFECV's CV: a bare fold count by default, time-ordered folds on a chronological suite."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from mlframe.feature_selection.wrappers.rfecv import RFECV
from mlframe.feature_selection.wrappers.rfecv._timestamp_ordered_split import TimestampOrderedSplit
from mlframe.training._model_configs import ModelHyperparamsConfig
from mlframe.training._preprocessing_configs import TrainingSplitConfig
from mlframe.training.core._rfecv_temporal_cv import apply_temporal_cv_to_rfecv, rfecv_cv_is_temporal


def test_training_configs_pass_rfecv_a_fold_count_not_a_plain_kfold():
    """The default ``cv_n_splits`` became an unstratified, unshuffled KFold instance, which RFECV cannot upgrade to
    StratifiedKFold, a group-aware or a temporal splitter; it must stay an int for RFECV to resolve."""
    from mlframe.training.trainer import get_training_configs

    configs = get_training_configs(rfecv_kwargs={"cv_n_splits": 4}, has_time=False, enabled_models=["cb"])
    assert configs.COMMON_RFECV_PARAMS["cv"] == 4
    assert not isinstance(configs.COMMON_RFECV_PARAMS["cv"], KFold)


def test_training_configs_keep_time_series_split_for_has_time():
    """Training configs keep time series split for has time."""
    from mlframe.training.trainer import get_training_configs

    configs = get_training_configs(rfecv_kwargs={"cv_n_splits": 4}, has_time=True, enabled_models=["cb"])
    assert isinstance(configs.COMMON_RFECV_PARAMS["cv"], TimeSeriesSplit)
    assert configs.COMMON_RFECV_PARAMS["cv"].n_splits == 4


def test_suite_int_cv_resolves_to_stratified_kfold_for_a_classifier(caplog):
    """With the int the suite now hands over, a classifier RFECV gets StratifiedKFold, and the early-stopping val_cv
    is built from it without the bogus 'does not accept n_splits' warning."""
    from mlframe.feature_selection.wrappers.rfecv._cv_setup import _resolve_cv_and_val_cv

    X = pd.DataFrame({"a": np.arange(40.0)})
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.wrappers.rfecv"):
        cv, val_cv, _ = _resolve_cv_and_val_cv(
            cv=4, X=X, y=np.arange(40) % 2, groups=None, estimator=LogisticRegression(), cv_shuffle=True, random_state=0,
            fit_params={}, early_stopping_val_nsplits=10, early_stopping_rounds=None, _polars_time_series_hint=False, verbose=0,
        )
    assert not caplog.records, [r.getMessage() for r in caplog.records]
    assert isinstance(cv, StratifiedKFold) and cv.shuffle
    assert isinstance(val_cv, StratifiedKFold) and val_cv.n_splits == 10 and val_cv.random_state == cv.random_state


@pytest.mark.parametrize(
    "split_kwargs, hp_kwargs, has_ts, expected",
    [
        (dict(shuffle_val=True, shuffle_test=False), {}, True, True),
        (dict(shuffle_val=True, shuffle_test=False), {}, False, False),
        (dict(shuffle_val=True, shuffle_test=True, val_sequential_fraction=0.0), {}, True, False),
        (dict(shuffle_val=True, shuffle_test=True, val_sequential_fraction=0.5), {}, True, True),
        (dict(shuffle_val=True, shuffle_test=True, val_sequential_fraction=0.0, cv_strategy="timeseries"), {}, True, True),
        (dict(shuffle_test=False), dict(has_time=False), True, False),
        (dict(shuffle_test=True, shuffle_val=True, val_sequential_fraction=0.0), dict(has_time=True), True, True),
        (dict(shuffle_test=False), dict(rfecv_kwargs={"cv_shuffle": True}), True, False),
        (dict(shuffle_test=False), dict(rfecv_kwargs={"cv": 5}), True, False),
    ],
    ids=["user_report", "no_timestamps", "fully_shuffled_holdouts", "half_sequential_val", "cv_strategy_timeseries",
         "explicit_has_time_false", "explicit_has_time_true", "explicit_cv_shuffle", "explicit_cv"],
)
def test_rfecv_cv_is_temporal_decision(split_kwargs, hp_kwargs, has_ts, expected):
    """Rfecv cv is temporal decision."""
    temporal, reason = rfecv_cv_is_temporal(
        np.arange(10) if has_ts else None, TrainingSplitConfig(**split_kwargs), ModelHyperparamsConfig(**hp_kwargs),
    )
    assert temporal is expected, reason


def _suite_rfecv(cv):
    """RFECV around logistic regression with the given cv and a one-minute runtime cap."""
    return RFECV(estimator=LogisticRegression(), cv=cv, cv_shuffle=True, max_runtime_mins=1.0)


def test_apply_temporal_cv_replaces_suite_cv_with_train_row_timestamps():
    """The reported run: ts_field on the extractor, shuffle_val=True / shuffle_test=False, cb_rfecv. RFECV used to get a
    KFold across time; it must get forward-chained folds over the train rows' own timestamps."""
    n = 100
    rng = np.random.default_rng(0)
    timestamps = pd.Series(pd.date_range("2025-01-01", periods=n, freq="D")[rng.permutation(n)])
    train_idx = np.sort(rng.choice(n, size=70, replace=False))
    params = {"cb_rfecv": _suite_rfecv(4), "user_rfecv": _suite_rfecv(KFold(3)), "tss_rfecv": _suite_rfecv(TimeSeriesSplit(n_splits=5, gap=1))}
    replaced = apply_temporal_cv_to_rfecv(
        params, timestamps=timestamps, train_idx=train_idx, split_config=TrainingSplitConfig(shuffle_val=True, shuffle_test=False),
        hyperparams_config=ModelHyperparamsConfig(), verbose=0,
    )
    assert replaced
    cv = params["cb_rfecv"].cv
    assert isinstance(cv, TimestampOrderedSplit) and cv.n_splits == 4 and params["cb_rfecv"].cv_shuffle is False
    assert isinstance(params["user_rfecv"].cv, KFold)
    assert isinstance(params["tss_rfecv"].cv, TimestampOrderedSplit) and params["tss_rfecv"].cv.gap == 1
    train_ts = timestamps.iloc[train_idx].to_numpy()
    for tr, te in cv.split(np.zeros((train_idx.size, 1))):
        assert train_ts[tr].max() < train_ts[te].min()


def test_apply_temporal_cv_leaves_iid_suite_alone():
    """Apply temporal cv leaves iid suite alone."""
    params = {"cb_rfecv": _suite_rfecv(4)}
    replaced = apply_temporal_cv_to_rfecv(
        params, timestamps=None, train_idx=None, split_config=TrainingSplitConfig(), hyperparams_config=ModelHyperparamsConfig(),
    )
    assert not replaced and params["cb_rfecv"].cv == 4 and params["cb_rfecv"].cv_shuffle is True


def test_rfecv_fit_with_timestamp_ordered_cv_on_unsorted_rows(caplog):
    """A full RFECV fit over rows not sorted by time: resolves to the timestamp-ordered splitter, builds its early-stopping
    val_cv without warnings, and every fold it scores trains on earlier rows than it tests on."""
    n = 240
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = (X["a"] + 0.3 * rng.normal(size=n) > 0).astype(int)
    ts = rng.permutation(n)
    rfecv = RFECV(
        estimator=LogisticRegression(), cv=TimestampOrderedSplit(n_splits=3, timestamps=ts), cv_shuffle=False,
        early_stopping_val_nsplits=4, max_runtime_mins=1.0, verbose=0,
    )
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.wrappers.rfecv"):
        rfecv.fit(X, y)
    assert isinstance(rfecv.cv_, TimestampOrderedSplit)
    assert not [r for r in caplog.records if "n_splits" in r.getMessage() or "TimestampOrderedSplit" in r.getMessage()]
    assert "a" in list(rfecv.get_feature_names_out())
    for tr, te in rfecv.cv_.split(X):
        assert ts[tr].max() < ts[te].min()
