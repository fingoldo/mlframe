"""Cluster 37 regression: a missing preprocessing config raises RuntimeError before any pipeline component is built (audit 2026-11-20, finding 37-F43).

The check used to be an ``assert``, which ``python -O`` strips, so a None config reached the pipeline. It is now an explicit raise that runs
before the components are built.
"""

from types import SimpleNamespace

import pytest

from mlframe.training.core import _main_train_suite as suite


def _state(preprocessing_config):
    """A minimal suite state that reaches the preprocessing guard and nothing else."""
    return SimpleNamespace(
        verbose=False,
        split_config=None,
        preprocessing_config=preprocessing_config,
        cat_features=[],
        common_params_dict={},
        rfecv_models=[],
        mrmr_kwargs={},
    )


def test_missing_preprocessing_config_raises_runtime_error_naming_it(monkeypatch):
    """With no preprocessing config the stage raises RuntimeError that names the missing config (37-F43)."""
    with pytest.raises(RuntimeError, match="preprocessing_config must be resolved"):
        suite._suite_stage_defaults_and_conversion(SimpleNamespace(), _state(None))


def test_guard_fires_before_any_pipeline_component_is_built(monkeypatch):
    """The guard runs before the pipeline components are built, so nothing is constructed from a missing config (37-F43)."""
    built = []
    monkeypatch.setattr(suite, "_get_pipeline_components", lambda *args, **kwargs: built.append(args) or (None, None, None))
    with pytest.raises(RuntimeError):
        suite._suite_stage_defaults_and_conversion(SimpleNamespace(), _state(None))
    assert built == []
