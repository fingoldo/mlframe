"""The unigram rescue put ``text_processing`` into ``fit()`` kwargs, where CatBoost does not accept it.

A production run hit the rescue path and died on ``TypeError: CatBoostClassifier.fit() got an unexpected keyword
argument 'text_processing'`` -- the recovery meant to save the fit killed the whole 21-minute suite instead, and
the TypeError propagated all the way out of ``train_mlframe_models_suite``. ``text_processing`` is a CatBoost
PARAMETER; the scaled-occurrence path in the same module already sets it through ``set_params``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from mlframe.training._training_loop import _train_model_with_fallback

DICT_EMPTY = "catboost/private/libs/feature_estimator/text_feature_estimators.cpp:89: Dictionary size is 0"


class _FakeCatBoost:
    """Raises the empty-dictionary error once, then records how the retry was configured.

    ``fit`` rejects unknown keywords exactly as CatBoost's does, which is the whole point: a rescue that pushes
    a parameter through fit kwargs has to fail here the same way it failed in production.
    """

    def __init__(self):
        self.params: dict[str, Any] = {}
        self.fit_calls: list[dict] = []

    def get_params(self):
        """Current estimator parameters."""
        return dict(self.params)

    def set_params(self, **kwargs):
        """Record parameters the caller sets, sklearn-style."""
        self.params.update(kwargs)
        return self

    # Mirrors the real ``CatBoostClassifier.fit`` signature closely enough to reject what it rejects.
    # ``text_processing`` is deliberately absent: it is a constructor parameter there, which is the
    # whole defect this file pins.
    _FIT_KEYWORDS = frozenset({
        "text_features", "cat_features", "embedding_features", "eval_set", "sample_weight", "baseline",
        "use_best_model", "verbose", "verbose_eval", "logging_level", "silent", "plot", "plot_file",
        "metric_period", "early_stopping_rounds", "save_snapshot", "snapshot_file", "snapshot_interval",
        "init_model", "callbacks", "column_description", "graph", "log_cout", "log_cerr",
    })

    def fit(self, X, y, **kwargs):
        """Accept only the keywords CatBoost's fit really takes; raise the dictionary error on the first call."""
        _unknown = set(kwargs) - self._FIT_KEYWORDS
        if _unknown:
            raise TypeError(f"CatBoostClassifier.fit() got an unexpected keyword argument '{sorted(_unknown)[0]}'")
        text_features = kwargs.get("text_features")
        cat_features = kwargs.get("cat_features")
        self.fit_calls.append({"text_features": text_features, "cat_features": cat_features})
        if len(self.fit_calls) == 1:
            raise RuntimeError(DICT_EMPTY)
        return self


@pytest.fixture
def single_token_frame():
    """Text columns carrying exactly one token per row -- the shape that empties CatBoost's bigram dictionary."""
    n = 64
    rng = np.random.default_rng(0)
    return (
        pd.DataFrame({
            "num": rng.standard_normal(n),
            "skills_text": [f"tok{i % 7}" for i in range(n)],
            "lang_text": [f"lang{i % 5}" for i in range(n)],
        }),
        pd.Series((rng.random(n) < 0.5).astype(int)),
    )


def _run(model, X, y):
    """Drive the fallback with text features declared, as the CatBoost strategy does."""
    return _train_model_with_fallback(
        model=model, model_obj=model, model_type_name="CatBoostClassifier",
        train_df=X, train_target=y, fit_params={"text_features": ["skills_text", "lang_text"]}, verbose=False,
    )


class TestTheRescueSurvivesTheRetry:
    """The defect was that the retry could not even be attempted."""

    def test_the_retry_does_not_raise_a_typeerror(self, single_token_frame, monkeypatch):
        """One assertion for the production failure: the rescue must not take the suite down."""
        monkeypatch.setattr("mlframe.training.cb._cb_text_probe.unigram_rescues_text_features", lambda *a, **k: True)
        X, y = single_token_frame
        model = _FakeCatBoost()
        _run(model, X, y)
        assert len(model.fit_calls) == 2, "the rescue never retried the fit"

    def test_text_processing_goes_to_params_not_fit(self, single_token_frame, monkeypatch):
        """Where the value has to land, stated directly."""
        monkeypatch.setattr("mlframe.training.cb._cb_text_probe.unigram_rescues_text_features", lambda *a, **k: True)
        X, y = single_token_frame
        model = _FakeCatBoost()
        _run(model, X, y)
        assert "text_processing" in model.get_params()

    def test_the_rescue_keeps_every_text_column(self, single_token_frame, monkeypatch):
        """Its whole purpose: no column the caller promoted gets dropped."""
        monkeypatch.setattr("mlframe.training.cb._cb_text_probe.unigram_rescues_text_features", lambda *a, **k: True)
        X, y = single_token_frame
        model = _FakeCatBoost()
        _run(model, X, y)
        # Set comparison: the contract is that no promoted column is lost, not the order they are listed in.
        assert set(model.fit_calls[1]["text_features"]) == {"skills_text", "lang_text"}

    def test_the_unigram_dictionary_is_what_gets_set(self, single_token_frame, monkeypatch):
        """A bigram dictionary would reproduce the very error being recovered from."""
        monkeypatch.setattr("mlframe.training.cb._cb_text_probe.unigram_rescues_text_features", lambda *a, **k: True)
        X, y = single_token_frame
        model = _FakeCatBoost()
        _run(model, X, y)
        _tp = model.get_params()["text_processing"]
        assert "1" in str(_tp) or "Unigram" in str(_tp) or "gram_order" in str(_tp)


class TestWhenTheRescueCannotApply:
    """A model that rejects the parameter must fall through, not crash."""

    def test_a_set_params_failure_falls_back_to_probing(self, single_token_frame, monkeypatch):
        """The per-column probe is the documented second option; an exception here must not escape."""
        monkeypatch.setattr("mlframe.training.cb._cb_text_probe.unigram_rescues_text_features", lambda *a, **k: True)
        monkeypatch.setattr("mlframe.training.cb._cb_text_probe.unusable_text_features", lambda *a, **k: {"skills_text": "probe says unusable"})
        X, y = single_token_frame
        model = _FakeCatBoost()

        def _reject(**kwargs):
            """Stands in for a CatBoost build that does not accept this parameter."""
            raise ValueError("this build does not accept text_processing")

        monkeypatch.setattr(model, "set_params", _reject)
        _run(model, X, y)
        assert len(model.fit_calls) == 2
        assert model.fit_calls[1]["text_features"] == ["lang_text"], "the unusable column should have been dropped"

    def test_a_dropped_column_is_rerouted_to_cat_features(self, single_token_frame, monkeypatch):
        """Left out entirely, CatBoost tries to cast its strings to float and raises."""
        monkeypatch.setattr("mlframe.training.cb._cb_text_probe.unigram_rescues_text_features", lambda *a, **k: False)
        monkeypatch.setattr("mlframe.training.cb._cb_text_probe.unusable_text_features", lambda *a, **k: {"skills_text": "probe says unusable"})
        X, y = single_token_frame
        model = _FakeCatBoost()
        _run(model, X, y)
        assert "skills_text" in (model.fit_calls[1]["cat_features"] or [])
