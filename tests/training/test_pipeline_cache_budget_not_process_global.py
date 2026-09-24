"""The pipeline cache's RAM-budget fraction belongs to the cache instance, not to os.environ."""

import os


from mlframe.training.strategies.pipeline_cache import PipelineCache, _resolve_pipeline_cache_budget


def test_a_second_cache_gets_its_own_fraction(monkeypatch):
    """Publishing the fraction into the environment let the first suite's budget stick for the whole process."""
    monkeypatch.delenv("MLFRAME_PIPELINE_CACHE_RAM_FRACTION", raising=False)
    monkeypatch.delenv("MLFRAME_PIPELINE_CACHE_BYTES_LIMIT", raising=False)
    small = PipelineCache(verbose=False, ram_budget_fraction=0.02)
    large = PipelineCache(verbose=False, ram_budget_fraction=0.5)
    assert small._bytes_limit <= large._bytes_limit
    assert "MLFRAME_PIPELINE_CACHE_RAM_FRACTION" not in os.environ, "the fraction must not be published process-globally"


def test_the_operator_env_still_wins_over_the_configured_fraction(monkeypatch):
    """An operator-set RAM fraction in the environment overrides the fraction the suite was configured with."""
    monkeypatch.delenv("MLFRAME_PIPELINE_CACHE_BYTES_LIMIT", raising=False)
    monkeypatch.setenv("MLFRAME_PIPELINE_CACHE_RAM_FRACTION", "0.02")
    assert _resolve_pipeline_cache_budget(0.5) == _resolve_pipeline_cache_budget(0.02)


def test_an_absolute_env_limit_still_wins_over_everything(monkeypatch):
    """An absolute byte limit in the environment beats every fraction, configured or not."""
    monkeypatch.setenv("MLFRAME_PIPELINE_CACHE_BYTES_LIMIT", "123456789")
    assert _resolve_pipeline_cache_budget(0.5) == 123456789
    assert PipelineCache(verbose=False, ram_budget_fraction=0.5)._bytes_limit == 123456789


def test_suite_setup_leaves_the_environment_alone(monkeypatch):
    """Running suite setup with a configured fraction must not publish it, so the next suite is free to configure its own."""
    import mlframe.training.core._phase_config_setup as pcs
    from mlframe.training.configs import TrainingBehaviorConfig

    monkeypatch.delenv("MLFRAME_PIPELINE_CACHE_RAM_FRACTION", raising=False)
    monkeypatch.delenv("MLFRAME_PIPELINE_CACHE_BYTES_LIMIT", raising=False)
    kwargs = dict(
        preprocessing_config=None, pipeline_config=None, feature_types_config=None, split_config=None,
        hyperparams_config=None, behavior_config=TrainingBehaviorConfig(pipeline_cache_ram_budget_fraction=0.2),
        reporting_config=None, output_config=None, outlier_detection_config=None, feature_selection_config=None,
        confidence_analysis_config=None, baseline_diagnostics_config=None, dummy_baselines_config=None,
        quantile_regression_config=None, composite_target_discovery_config=None, feature_handling_config=None,
        model_name="m", target_name="t", mlframe_models=None, verbose=0,
    )
    ctx = pcs.setup_configuration(**kwargs)
    assert "MLFRAME_PIPELINE_CACHE_RAM_FRACTION" not in os.environ
    assert ctx.behavior_config.pipeline_cache_ram_budget_fraction == 0.2, "the configured value is what the cache is handed"
