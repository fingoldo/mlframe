"""A cached model behind a feature selector must be reused when the RAW input is unchanged.

The stale-cache check compared the saved model's feature names -- the selector's OUTPUT -- with the current frame's columns -- the selector's
INPUT -- so it reported a mismatch on every rerun ("saved=18, current=104") and discarded the dump, which for an RFECV pipeline is hours of work.
The comparable quantity is the input column list the fitted pre-pipeline recorded.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

from mlframe.training import FeatureSelectionConfig, OutputConfig, PreprocessingConfig
from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.train_eval import _validate_cached_model_schema

from .shared import SimpleFeaturesAndTargetsExtractor
from .test_suite_coverage_gaps import _LEAN_REPORTING_CONFIG, _make_baseline_pandas

pytest.importorskip("catboost")


def _run(df, data_dir):
    """Run."""
    pipeline = Pipeline([("sel", SelectKBest(f_classif, k=2).set_output(transform="pandas"))])
    return train_mlframe_models_suite(
        df=df, target_name="tgt", model_name="mdl", features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(regression=False),
        mlframe_models=["cb"], hyperparams_config={"iterations": 5, "cb_kwargs": {"task_type": "CPU", "verbose": 0}},
        preprocessing_config=PreprocessingConfig(drop_columns=[]), use_ordinary_models=False, use_mlframe_ensembles=False,
        feature_selection_config=FeatureSelectionConfig(custom_pre_pipelines={"kbest": pipeline}), verbose=1,
        output_config=OutputConfig(data_dir=str(data_dir), models_dir="models", save_charts=False), reporting_config=_LEAN_REPORTING_CONFIG,
    )


def _scores(models):
    """Scores."""
    return {e.model_name: {s: {k: v for k, v in (e.metrics.get(s) or {}).items() if isinstance(v, (int, float))} for s in ("val", "test")}
            for per_target in models.values() for entries in per_target.values() for e in entries}


def test_selector_model_is_loaded_not_retrained_on_rerun(tmp_path, monkeypatch, caplog):
    """Second run on the same data dir logs 'Loaded.', does not refit the selector, and reproduces the metrics."""
    fits = []
    original_fit = SelectKBest.fit

    def spy(self, *args, **kwargs):
        """Spy."""
        fits.append(1)
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(SelectKBest, "fit", spy)
    df = _make_baseline_pandas(n=400, seed=0, with_cat=False, regression=False)
    first, _ = _run(df, tmp_path)
    fits_after_first = len(fits)
    assert fits_after_first >= 1

    caplog.clear()
    with caplog.at_level(logging.INFO):
        second, _ = _run(df, tmp_path)
    messages = [r.getMessage() for r in caplog.records]
    assert any(m == "Loaded." for m in messages), messages
    assert not any("Invalidating stale cached model" in m for m in messages), [m for m in messages if "Invalidating" in m]
    assert len(fits) == fits_after_first, "the selector was refit although the cached model was reused"
    assert _scores(first) == _scores(second)


def test_changed_input_columns_still_invalidate_selector_model(tmp_path, monkeypatch, caplog):
    """A different RAW column set is a different input: the old dump is not reused and the selector refits."""
    fits = []
    original_fit = SelectKBest.fit
    monkeypatch.setattr(SelectKBest, "fit", lambda self, *a, **k: (fits.append(1), original_fit(self, *a, **k))[1])
    df = _make_baseline_pandas(n=400, seed=0, with_cat=False, regression=False)
    _run(df, tmp_path)
    fits_after_first = len(fits)

    caplog.clear()
    with caplog.at_level(logging.INFO):
        _run(df.assign(num_extra=np.arange(len(df), dtype="float32")), tmp_path)
    messages = [r.getMessage() for r in caplog.records]
    # The dump's file name carries a schema hash, so a changed column set never even finds the old dump; the validator below is the second line.
    assert not any(m == "Loaded." for m in messages), messages
    assert len(fits) > fits_after_first


class _Named:
    """Named."""
    def __init__(self, names):
        self.feature_names_in_ = np.asarray(names, dtype=object)


class _OutModel:
    """OutModel."""
    def __init__(self, names):
        self.feature_names_ = list(names)


def test_validator_compares_pipeline_input_not_model_output():
    """Validator compares pipeline input not model output."""
    raw = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
    loaded = SimpleNamespace(model=_OutModel(["b"]), pre_pipeline=Pipeline([("sel", _Named(["a", "b", "c"]))]))
    assert _validate_cached_model_schema(loaded, raw) is None
    assert "pre-pipeline input" in _validate_cached_model_schema(loaded, raw.assign(d=1.0))


def test_validator_keeps_conservative_invalidation_when_pipeline_records_no_input_names():
    """Validator keeps conservative invalidation when pipeline records no input names."""
    raw = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
    loaded = SimpleNamespace(model=_OutModel(["b"]), pre_pipeline=Pipeline([("sel", SimpleNamespace())]))
    reason = _validate_cached_model_schema(loaded, raw)
    assert reason and "records no input column names" in reason


def test_validator_without_pipeline_keeps_model_name_comparison():
    """Validator without pipeline keeps model name comparison."""
    raw = pd.DataFrame({"a": [1.0], "b": [2.0]})
    loaded = SimpleNamespace(model=_OutModel(["a", "b"]), pre_pipeline=None)
    assert _validate_cached_model_schema(loaded, raw) is None
    assert "feature-name mismatch" in _validate_cached_model_schema(loaded, raw.assign(c=1.0))
