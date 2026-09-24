"""AutoMLConfig.time_limit reaches both AutoML backends; an explicit per-library key wins."""

import pandas as pd

import mlframe.training.automl as automl
from mlframe.training.configs import AutoMLConfig


def _capture(monkeypatch):
    seen = {}
    monkeypatch.setattr(automl, "train_autogluon_model", lambda **kw: seen.setdefault("ag", kw) and None)
    monkeypatch.setattr(automl, "train_lama_model", lambda **kw: seen.setdefault("lama", kw) and None)
    return seen


def _run(config):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "target": [0, 1, 0, 1]})
    automl.train_automl_models_suite(train_df=df, test_df=df, target_name="target", config=config)


def test_time_limit_is_forwarded(monkeypatch):
    seen = _capture(monkeypatch)
    _run(AutoMLConfig(use_autogluon=True, use_lama=True, time_limit=60))
    assert seen["ag"]["fit_params"] == {"time_limit": 60}
    assert seen["lama"]["init_params"] == {"timeout": 60}


def test_explicit_library_key_wins_and_unset_budget_changes_nothing(monkeypatch):
    seen = _capture(monkeypatch)
    _run(AutoMLConfig(use_autogluon=True, use_lama=True, time_limit=60, autogluon_fit_params={"time_limit": 5, "presets": "x"}))
    assert seen["ag"]["fit_params"] == {"time_limit": 5, "presets": "x"}
    seen.clear()
    _run(AutoMLConfig(use_autogluon=True, use_lama=True))
    assert seen["ag"]["fit_params"] is None and seen["lama"]["init_params"] is None
