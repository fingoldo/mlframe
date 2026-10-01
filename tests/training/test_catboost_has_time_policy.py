"""CatBoost ``has_time`` follows the unified CV policy only when the train rows are chronological (see core/_catboost_has_time.py)."""
import logging

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("catboost")
from catboost import CatBoostClassifier, CatBoostRegressor

from mlframe.feature_selection.cv_policy import CVPolicy
from mlframe.training._model_configs import ModelHyperparamsConfig
from mlframe.training.core._catboost_has_time import apply_catboost_has_time, decide_catboost_has_time, timestamps_are_chronological


def _ts(n=200, shuffled=False):
    """Hourly timestamps, optionally shuffled."""
    ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="h"))
    return ts.sample(frac=1.0, random_state=0).reset_index(drop=True) if shuffled else ts


def _policy(ts, kind="temporal"):
    """CV policy of the given kind, carrying the timestamps only when temporal."""
    return CVPolicy(kind, "test", ts if kind == "temporal" else None)


def _params():
    """Model params for a CatBoost entry with has_time off and a non-CatBoost entry."""
    return {"cb": {"model": CatBoostClassifier(iterations=3, has_time=False, verbose=0, allow_writing_files=False)}, "lgb": {"model": object()}}


def test_sorted_train_and_temporal_policy_enables_has_time():
    """Sorted train and temporal policy enables has time."""
    mp = _params()
    assert apply_catboost_has_time(mp, _policy(_ts()), ModelHyperparamsConfig()) is True
    assert mp["cb"]["model"].get_params()["has_time"] is True


def test_unsorted_train_leaves_has_time_off_and_logs_once(caplog):
    """Unsorted train leaves has time off and logs once."""
    mp = _params()
    with caplog.at_level(logging.INFO, logger="mlframe.training.core._catboost_has_time"):
        assert apply_catboost_has_time(mp, _policy(_ts(shuffled=True)), ModelHyperparamsConfig()) is False
    assert mp["cb"]["model"].get_params()["has_time"] is False
    msgs = [r.getMessage() for r in caplog.records if "has_time stays off" in r.getMessage()]
    assert len(msgs) == 1 and "not in chronological order" in msgs[0]


def test_explicit_false_wins_over_temporal_policy():
    """Explicit false wins over temporal policy."""
    mp = _params()
    cfg = ModelHyperparamsConfig(has_time=False)
    assert apply_catboost_has_time(mp, _policy(_ts()), cfg) is False
    assert mp["cb"]["model"].get_params()["has_time"] is False


def test_explicit_cb_kwargs_has_time_is_not_overridden():
    """Explicit cb kwargs has time is not overridden."""
    cfg = ModelHyperparamsConfig(cb_kwargs={"has_time": False})
    assert decide_catboost_has_time(_policy(_ts()), cfg)[0] is False


@pytest.mark.parametrize("policy", [None, CVPolicy("iid", "x"), CVPolicy("temporal", "x", None)])
def test_no_policy_or_no_timestamps_is_unchanged(policy):
    """No policy or no timestamps is unchanged."""
    mp = _params()
    assert apply_catboost_has_time(mp, policy, ModelHyperparamsConfig()) is False
    assert mp["cb"]["model"].get_params()["has_time"] is False


def test_missing_timestamps_are_not_chronological():
    """Missing timestamps are not chronological."""
    ts = _ts()
    ts.iloc[5] = pd.NaT
    assert not timestamps_are_chronological(ts)
    assert timestamps_are_chronological(np.array([1.0, 1.0, 2.0]))
    assert not timestamps_are_chronological(np.array([1.0, np.nan, 2.0]))
    assert timestamps_are_chronological(_ts().dt.tz_localize("UTC"))


def test_regressor_wrapper_param_is_reached():
    """Regressor wrapper param is reached."""
    mp = {"cb": {"model": CatBoostRegressor(iterations=3, verbose=0, allow_writing_files=False)}}
    assert apply_catboost_has_time(mp, _policy(_ts()), ModelHyperparamsConfig()) is True
    assert mp["cb"]["model"].get_params()["has_time"] is True


def test_real_fit_with_has_time_reaches_model():
    """Real fit with has time reaches model."""
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"a": rng.normal(size=n), "c": rng.integers(0, 5, n).astype(str)})
    y = (X["a"] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    mp = {"cb": {"model": CatBoostClassifier(iterations=10, verbose=0, allow_writing_files=False, thread_count=1)}}
    assert apply_catboost_has_time(mp, _policy(_ts(n)), ModelHyperparamsConfig()) is True
    model = mp["cb"]["model"]
    model.fit(X, y, cat_features=["c"])
    assert model.get_params()["has_time"] is True
    assert model.predict_proba(X).shape == (n, 2)


def test_catboost_wrapped_by_metamodel_gets_has_time():
    """Catboost wrapped by metamodel gets has time."""
    from sklearn.calibration import CalibratedClassifierCV

    wrapped = CalibratedClassifierCV(CatBoostClassifier(iterations=3, verbose=0, allow_writing_files=False))
    mp = {"cb": {"model": wrapped}}
    assert apply_catboost_has_time(mp, _policy(_ts()), ModelHyperparamsConfig()) is True
    assert wrapped.get_params(deep=True)["estimator__has_time"] is True
