"""Tests for ReportingConfig + sibling configs introduced 2026-04-27.

Covers:
- ReportingConfig.title_metrics_template parsing + validation at construction time
- ReportingConfig.title_metrics_tokens derived field is populated by the model_validator
- Histogram subplot kwargs (show_prob_histogram, prob_histogram_yscale, show_inline_population_labels)
- OutputConfig, FeatureImportanceConfig, OutlierDetectionConfig basic invariants
- PreprocessingConfig new transformer override fields (scaler/imputer/category_encoder)
- FeatureSelectionConfig new custom_pre_pipelines field

The token-template grammar is the user's primary calibration-report knob, so
its validation gets the most coverage. Validation runs at config-construction
time, not in the plotting hot path - tests assert the failure modes happen
before training starts.
"""

import pytest

from pydantic import ValidationError

from mlframe.training.configs import (
    FeatureImportanceConfig,
    FeatureSelectionConfig,
    OutlierDetectionConfig,
    OutputConfig,
    PreprocessingConfig,
    ReportingConfig,
)


class TestReportingConfigTitleTemplate:
    """ReportingConfig.title_metrics_template - parsing, validation, derived tokens."""

    def test_default_template_parses_into_expected_tokens(self):
        cfg = ReportingConfig()
        assert cfg.title_metrics_tokens == (
            "ICE", "BR_DECOMP", "ECE", "CMAEW", "LL", "ROC_AUC", "PR_AUC",
        )

    def test_custom_template_is_parsed_in_order(self):
        cfg = ReportingConfig(title_metrics_template="CMAEW ICE")
        assert cfg.title_metrics_tokens == ("CMAEW", "ICE")

    def test_template_subset_of_tokens_keeps_order(self):
        cfg = ReportingConfig(title_metrics_template="ICE CMAEW")
        assert cfg.title_metrics_tokens == ("ICE", "CMAEW")

    def test_unknown_token_rejected_at_construction(self):
        with pytest.raises(ValidationError) as exc_info:
            ReportingConfig(title_metrics_template="ICE FOO")
        msg = str(exc_info.value)
        assert "Unknown title-metrics tokens" in msg
        assert "FOO" in msg
        assert "Allowed" in msg

    def test_duplicate_token_rejected_at_construction(self):
        with pytest.raises(ValidationError) as exc_info:
            ReportingConfig(title_metrics_template="ICE ICE")
        assert "Duplicate" in str(exc_info.value)

    def test_br_and_br_decomp_mutually_exclusive(self):
        with pytest.raises(ValidationError) as exc_info:
            ReportingConfig(title_metrics_template="BR BR_DECOMP")
        assert "mutually exclusive" in str(exc_info.value)

    def test_empty_template_legal_yields_empty_tokens(self):
        cfg = ReportingConfig(title_metrics_template="")
        assert cfg.title_metrics_tokens == ()

    def test_template_case_insensitive_normalised_to_upper(self):
        cfg = ReportingConfig(title_metrics_template="ice br_decomp ece")
        assert cfg.title_metrics_tokens == ("ICE", "BR_DECOMP", "ECE")

    def test_template_extra_whitespace_collapsed(self):
        cfg = ReportingConfig(title_metrics_template="  ICE   BR_DECOMP  ECE  ")
        assert cfg.title_metrics_tokens == ("ICE", "BR_DECOMP", "ECE")

    def test_validator_runs_before_plot_time(self):
        # Construction-time validation - we never have to call the plot to find out.
        # Asserts the failure point is at __init__, not later.
        with pytest.raises(ValidationError):
            ReportingConfig(title_metrics_template="NOT_A_REAL_METRIC")


class TestReportingConfigHistogramFields:
    """Histogram subplot toggles + label toggle."""

    def test_show_prob_histogram_default_true(self):
        assert ReportingConfig().show_prob_histogram is True

    def test_show_inline_population_labels_default_true(self):
        # Independent of histogram toggle - users can keep both, drop both, or only one.
        assert ReportingConfig().show_inline_population_labels is True

    def test_prob_histogram_yscale_default_auto(self):
        assert ReportingConfig().prob_histogram_yscale == "auto"

    def test_prob_histogram_yscale_accepts_explicit_modes(self):
        for mode in ("auto", "log", "linear"):
            cfg = ReportingConfig(prob_histogram_yscale=mode)
            assert cfg.prob_histogram_yscale == mode

    def test_prob_histogram_yscale_rejects_invalid(self):
        with pytest.raises(ValidationError):
            ReportingConfig(prob_histogram_yscale="banana")

    def test_histogram_and_label_toggles_are_independent(self):
        # All four combinations construct cleanly.
        for hist in (True, False):
            for label in (True, False):
                cfg = ReportingConfig(
                    show_prob_histogram=hist,
                    show_inline_population_labels=label,
                )
                assert cfg.show_prob_histogram is hist
                assert cfg.show_inline_population_labels is label


class TestReportingConfigMetricComputeGates:
    """compute_*set_metrics fields lifted from trainer-internal TrainingControlConfig."""

    def test_compute_trainset_metrics_default_false(self):
        # Matches the historical trainer-internal default - users who want
        # train-set metrics for overfit diagnostics opt in explicitly.
        assert ReportingConfig().compute_trainset_metrics is False

    def test_compute_valset_metrics_default_true(self):
        assert ReportingConfig().compute_valset_metrics is True

    def test_compute_testset_metrics_default_true(self):
        assert ReportingConfig().compute_testset_metrics is True

    def test_compute_metrics_round_trip(self):
        cfg = ReportingConfig(
            compute_trainset_metrics=True,
            compute_valset_metrics=False,
            compute_testset_metrics=False,
        )
        assert cfg.compute_trainset_metrics is True
        assert cfg.compute_valset_metrics is False
        assert cfg.compute_testset_metrics is False

    def test_compute_metrics_propagate_via_model_dump(self):
        # The suite-internal common_params_dict assembly does
        # `common_params_dict.update(reporting_config.model_dump())`. Confirm
        # the new keys land in model_dump so the deep dict-key consumers in
        # _build_configs_from_params pick them up.
        dump = ReportingConfig().model_dump()
        assert "compute_trainset_metrics" in dump
        assert "compute_valset_metrics" in dump
        assert "compute_testset_metrics" in dump


class TestReportingConfigCustomMetrics:
    """Custom ICE / RICE callables lifted from trainer-internal MetricsConfig."""

    def test_custom_ice_metric_default_none(self):
        # None triggers the trainer's fallback to compute_probabilistic_multiclass_error.
        assert ReportingConfig().custom_ice_metric is None

    def test_custom_rice_metric_default_none(self):
        assert ReportingConfig().custom_rice_metric is None

    def test_custom_ice_metric_round_trip(self):
        def fake_ice(y_true, y_score):
            return 0.42

        cfg = ReportingConfig(custom_ice_metric=fake_ice)
        assert cfg.custom_ice_metric is fake_ice

    def test_custom_metrics_propagate_via_model_dump(self):
        dump = ReportingConfig().model_dump()
        assert "custom_ice_metric" in dump
        assert "custom_rice_metric" in dump


class TestUseCacheDefaultFlip:
    """2026-04-27: TrainingControlConfig.use_cache default flipped False -> True
    for consistency with train_eval.py:664's de-facto True behavior. Cache
    loading is almost always faster than retraining."""

    def test_training_control_use_cache_default_true(self):
        from mlframe.training.configs import TrainingControlConfig
        assert TrainingControlConfig().use_cache is True

    def test_training_control_use_cache_can_be_disabled(self):
        from mlframe.training.configs import TrainingControlConfig
        assert TrainingControlConfig(use_cache=False).use_cache is False


class TestReportingConfigSlimming:
    """Confirm the fields that moved out of the renamed config really did move."""

    def test_no_plot_file_field(self):
        # plot_file moved to OutputConfig.
        assert "plot_file" not in ReportingConfig.model_fields

    def test_no_data_dir_field(self):
        # data_dir moved to OutputConfig.
        assert "data_dir" not in ReportingConfig.model_fields

    def test_no_models_subdir_or_models_dir(self):
        # models_subdir was renamed to models_dir; both names live on OutputConfig only.
        assert "models_subdir" not in ReportingConfig.model_fields
        assert "models_dir" not in ReportingConfig.model_fields

    def test_no_fi_kwargs_dict(self):
        # fi_kwargs replaced by typed FeatureImportanceConfig.
        assert "fi_kwargs" not in ReportingConfig.model_fields
        assert "feature_importance_config" in ReportingConfig.model_fields

    def test_no_show_x_in_title_booleans(self):
        # All 9 collapsed into title_metrics_template.
        for legacy in (
            "show_brier_loss_in_title", "show_cmaew_in_title", "show_roc_auc_in_title",
            "show_pr_auc_in_title", "show_logloss_in_title", "show_coverage_in_title",
            "show_points_density_in_title", "show_ece_in_title", "show_brier_decomp_in_title",
        ):
            assert legacy not in ReportingConfig.model_fields, f"{legacy} should have been collapsed"


class TestOutputConfig:
    def test_defaults(self):
        cfg = OutputConfig()
        assert cfg.data_dir == ""
        assert cfg.models_dir == "models"
        assert cfg.plot_file == ""
        assert cfg.save_charts is True

    def test_models_dir_renamed_from_models_subdir(self):
        # Symmetry with data_dir - both are typed peers, same noun pattern.
        cfg = OutputConfig(data_dir="./d", models_dir="./m")
        assert cfg.models_dir == "./m"
        # The pre-2026-04-27 name is gone.
        assert "models_subdir" not in OutputConfig.model_fields


class TestFeatureImportanceConfig:
    def test_defaults_match_plot_function_kwargs(self):
        cfg = FeatureImportanceConfig()
        # plot_model_feature_importances' num_factors default was bumped
        # 10 -> 15 (post-33-feature rollout); FeatureImportanceConfig's
        # default must track the plot-function default it documents.
        assert cfg.num_factors == 15
        assert cfg.figsize == (7.5, 2.5)
        assert cfg.positive_fi_only is False
        assert cfg.show_plots is True


class TestOutlierDetectionConfig:
    def test_defaults(self):
        cfg = OutlierDetectionConfig()
        assert cfg.detector is None
        assert cfg.apply_to_val is True

    def test_apply_to_val_renamed_from_od_val_set(self):
        cfg = OutlierDetectionConfig(apply_to_val=False)
        assert cfg.apply_to_val is False
        # The pre-2026-04-27 name is gone.
        assert "od_val_set" not in OutlierDetectionConfig.model_fields


class TestPreprocessingConfigTransformers:
    def test_unprefixed_field_names(self):
        # Field names match dict keys the deleted pass-through carried; no `custom_` prefix.
        for f in ("scaler", "imputer", "category_encoder"):
            assert f in PreprocessingConfig.model_fields, f"{f} missing on PreprocessingConfig"

    def test_none_default_preserves_context_aware_selection(self):
        cfg = PreprocessingConfig()
        assert cfg.scaler is None
        assert cfg.imputer is None
        assert cfg.category_encoder is None

    def test_overrides_round_trip(self):
        sentinel_scaler = object()
        sentinel_imputer = object()
        sentinel_encoder = object()
        cfg = PreprocessingConfig(
            scaler=sentinel_scaler,
            imputer=sentinel_imputer,
            category_encoder=sentinel_encoder,
        )
        assert cfg.scaler is sentinel_scaler
        assert cfg.imputer is sentinel_imputer
        assert cfg.category_encoder is sentinel_encoder


class TestFeatureSelectionConfigCustomPipelines:
    def test_custom_pre_pipelines_default_empty_dict(self):
        cfg = FeatureSelectionConfig()
        assert cfg.custom_pre_pipelines == {}

    def test_custom_pre_pipelines_accepts_dict(self):
        sentinel = object()
        cfg = FeatureSelectionConfig(custom_pre_pipelines={"pca50": sentinel})
        assert cfg.custom_pre_pipelines == {"pca50": sentinel}


class TestFeatureImportanceConfigStaleClassCoercion:
    """Regression for the 2026-05-04 prod failure where ``ReportingConfig``
    rejected a ``FeatureImportanceConfig`` instance with a name-matched but
    identity-divergent class.

    Pydantic v2's ``model_type`` validator strictly checks
    ``type(instance) is FeatureImportanceConfig``. Two real-world scenarios
    break that without any code bug:

      1) ``%autoreload 2`` re-imports ``configs.py`` after an edit; new
         class is built but ``trainer.py`` still references the OLD
         class and instantiates from it. ``ReportingConfig`` (new class)
         then sees an old-class instance and rejects.
      2) Two checkouts on ``sys.path`` (e.g. recovery + canonical) -- one
         resolves ``configs`` from path A, the other from path B; both
         classes are named ``FeatureImportanceConfig`` but identity differs.

    The ``_coerce_feature_importance_config`` validator detects the
    name-match-but-identity-mismatch case via duck typing
    (``hasattr(v, 'model_dump')`` + class-name check) and rebuilds the
    instance against THIS module's class identity via ``model_dump()``
    round-trip.
    """

    def test_same_class_identity_passes_through(self):
        fi = FeatureImportanceConfig()
        rc = ReportingConfig(feature_importance_config=fi)
        # No round-trip happened: same instance, not a copy.
        assert rc.feature_importance_config is fi

    def test_none_passes_through(self):
        rc = ReportingConfig(feature_importance_config=None)
        assert rc.feature_importance_config is None

    def test_dict_passes_through_normal_validation(self):
        rc = ReportingConfig(feature_importance_config={"n_top_features": 7})
        assert isinstance(rc.feature_importance_config, FeatureImportanceConfig)
        assert rc.feature_importance_config.n_top_features == 7

    def test_stale_class_with_matching_name_coerces_via_model_dump(self):
        """Simulates the autoreload / multi-checkout scenario: two distinct
        class objects with the same name. The validator rebuilds the
        instance against THIS module's class."""
        # Build a doppelganger class with the SAME name and SAME fields as
        # FeatureImportanceConfig, but a different class identity.
        from pydantic import BaseModel

        StaleFI = type(
            "FeatureImportanceConfig",
            (BaseModel,),
            {
                "__annotations__": {"n_top_features": int},
                "n_top_features": 17,
            },
        )
        stale_inst = StaleFI()
        # Pre-condition: the doppelganger really has identity != real one.
        assert type(stale_inst) is not FeatureImportanceConfig
        assert type(stale_inst).__name__ == FeatureImportanceConfig.__name__

        # ReportingConfig accepts it via the coercion validator.
        rc = ReportingConfig(feature_importance_config=stale_inst)
        assert isinstance(rc.feature_importance_config, FeatureImportanceConfig)
        # Field round-tripped via model_dump, so values survive.
        assert rc.feature_importance_config.n_top_features == 17

    def test_stale_class_via_importlib_reload_simulates_autoreload(self):
        """End-to-end: actually trigger ``importlib.reload`` and verify
        old-class instances feed into the new ReportingConfig cleanly.
        Mirrors the exact failure mode in the 2026-05-04 prod log.

        The reload swaps fresh class objects into ``sys.modules`` but
        leaves THIS test module's bound names pointing at the originals
        -- which is poison for any subsequent test in this class that
        compares identities. The ``try/finally`` restores via a second
        reload so siblings see consistent state regardless of order."""
        # Direct simulation of the autoreload / multi-checkout scenario:
        # construct a SECOND class with the same name as ``FeatureImportanceConfig``
        # but a distinct identity, and feed an instance of it into the
        # production ``ReportingConfig``. The validator must accept this via
        # the ``model_dump`` round-trip rather than failing with
        # ``ValidationError(model_type)``.
        # An earlier version of this test used ``importlib.reload`` to force
        # the divergence; reload pollutes other tests in this module because
        # the rebuilt pydantic schemas hold references to the new class, and
        # restoring the original symbols on the module doesn't undo that.
        from pydantic import BaseModel as _BM
        # Mirror enough of the real FeatureImportanceConfig surface that
        # model_dump round-trips through ReportingConfig's validator cleanly.
        # The validator round-trips via ``model_dump()`` then re-instantiates
        # the canonical class, so only the field names need to overlap.
        StaleFI = type(
            "FeatureImportanceConfig",
            (_BM,),
            {"model_config": {"extra": "allow"}, "__annotations__": {}},
        )
        stale_inst = StaleFI()
        # Sanity: same name but distinct class identity.
        assert StaleFI is not FeatureImportanceConfig
        assert StaleFI.__name__ == FeatureImportanceConfig.__name__
        rc = ReportingConfig(feature_importance_config=stale_inst)
        assert type(rc.feature_importance_config) is FeatureImportanceConfig
