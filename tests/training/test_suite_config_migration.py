"""Verify the 2026-04-27 train_mlframe_models_suite signature migration.

The user explicitly waived back-compat. These tests assert the breaking
changes hold:
- `init_common_params` kwarg deleted from the suite signature
- 9 orphan top-level kwargs (data_dir/models_dir/save_charts/outlier_detector/
  od_val_set/use_mrmr_fs/mrmr_kwargs/rfecv_models/custom_pre_pipelines)
  deleted; each migrated to a typed config
- 6 model-selection kwargs intentionally KEPT at the top level
"""

import inspect

import pytest

from mlframe.training import (
    ConfidenceAnalysisConfig,
    FeatureImportanceConfig,
    FeatureSelectionConfig,
    OutlierDetectionConfig,
    OutputConfig,
    ReportingConfig,
    train_mlframe_models_suite,
)


@pytest.fixture(scope="session")
def suite_params():
    # Session-scoped: inspecting a function signature is pure introspection,
    # idempotent across the whole session.
    return set(inspect.signature(train_mlframe_models_suite).parameters)


class TestRemovedKwargs:
    """Each previously-top-level kwarg must be gone from the suite signature."""

    @pytest.mark.parametrize(
        "kwarg",
        [
            "init_common_params",
            "data_dir",
            "models_dir",
            "save_charts",
            "outlier_detector",
            "od_val_set",
            "use_mrmr_fs",
            "mrmr_kwargs",
            "rfecv_models",
            "custom_pre_pipelines",
        ],
    )
    def test_kwarg_removed(self, suite_params, kwarg):
        assert kwarg not in suite_params, f"`{kwarg}` should have been removed from train_mlframe_models_suite in the 2026-04-27 refactor"

    def test_typeerror_when_calling_with_removed_kwarg(self):
        # Pass a removed kwarg via **kwargs (so the file doesn't parse-time-fail
        # on a literal that the migration grep would flag) to confirm Python
        # raises TypeError before any work runs.
        legacy_kwargs = {"init_common_params": {"show_perf_chart": False}}
        with pytest.raises(TypeError) as exc_info:
            train_mlframe_models_suite(
                df=None,
                target_name="x",
                model_name="x",
                features_and_targets_extractor=None,
                **legacy_kwargs,
            )
        # TypeError text mentions the unexpected kwarg.
        assert "init_common_params" in str(exc_info.value)


class TestNewTypedConfigsArePresent:
    @pytest.mark.parametrize(
        "kwarg",
        [
            "reporting_config",
            "output_config",
            "outlier_detection_config",
            "feature_selection_config",
            "confidence_analysis_config",
        ],
    )
    def test_kwarg_present(self, suite_params, kwarg):
        assert kwarg in suite_params, f"`{kwarg}` should be a first-class kwarg of train_mlframe_models_suite"


class TestModelSelectionKwargsRetained:
    """Model-selection kwargs intentionally stay at the top level (not wrapped in a config)."""

    @pytest.mark.parametrize(
        "kwarg",
        [
            "mlframe_models",
            "use_ordinary_models",
            "use_mlframe_ensembles",
            "recurrent_models",
            "recurrent_config",
            "sequences",
        ],
    )
    def test_model_selection_kwarg_kept(self, suite_params, kwarg):
        assert kwarg in suite_params, f'`{kwarg}` is intentionally a top-level kwarg - it answers "what does this suite do" and was NOT migrated into a config'


class TestConfigInstantiationDoesNotRaise:
    """Construct each new config with sentinel values - smoke check."""

    def test_reporting_config(self):
        cfg = ReportingConfig(
            show_prob_histogram=False,
            title_metrics_template="ICE BR",
            feature_importance_config=FeatureImportanceConfig(num_factors=5),
        )
        assert cfg.title_metrics_tokens == ("ICE", "BR")
        assert cfg.feature_importance_config.num_factors == 5

    def test_output_config(self, tmp_path):
        cfg = OutputConfig(data_dir=str(tmp_path), models_dir="models", save_charts=False)
        assert cfg.save_charts is False

    def test_outlier_detection_config(self):
        sentinel = object()
        cfg = OutlierDetectionConfig(detector=sentinel, apply_to_val=False)
        assert cfg.detector is sentinel
        assert cfg.apply_to_val is False

    def test_feature_selection_config_with_custom_pipelines(self):
        sentinel = object()
        cfg = FeatureSelectionConfig(
            use_mrmr_fs=True,
            rfecv_models=["cb"],
            custom_pre_pipelines={"my_pca": sentinel},
        )
        assert cfg.use_mrmr_fs is True
        assert cfg.rfecv_models == ["cb"]
        assert cfg.custom_pre_pipelines == {"my_pca": sentinel}

    def test_confidence_analysis_config(self):
        # Wired as first-class kwarg 2026-04-27 - was severed before that.
        cfg = ConfidenceAnalysisConfig(
            include=True,
            use_shap=False,
            max_features=8,
            cmap="viridis",
            alpha=0.5,
            ylabel="custom",
            title="custom title",
        )
        assert cfg.include is True
        assert cfg.use_shap is False
        assert cfg.max_features == 8
        assert cfg.cmap == "viridis"
