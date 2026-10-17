"""Full public docstring of ``train_mlframe_models_suite`` (kept out of the function so the function body stays within the length gate)."""
from __future__ import annotations

SUITE_DOC = """
    Train a suite of ML models on a dataset.

    Happy path (extractor inferred from the target column):

        models, metadata = train_mlframe_models_suite(df, target_name="y")

    Args:
        df: DataFrame or path to parquet file
        target_name: Name of the target to predict
        model_name: Base name for the models (defaults to ``"model"``)
        features_and_targets_extractor: FeaturesAndTargetsExtractor instance for computing targets.
            Optional: when None, a ``SimpleFeaturesAndTargetsExtractor`` is built from ``target_name``,
            with the task type (regression vs classification) inferred from the target column's
            dtype/cardinality. Pass an explicit extractor to override.
        mlframe_models: List of model types to train (cb, lgb, xgb, mlp, hgb, linear, ridge, etc.)
        recurrent_models: List of recurrent model types to train (lstm, gru, rnn, transformer).
            These models handle sequential data and support variable-length sequences.
        recurrent_config: RecurrentConfig object for recurrent model hyperparameters.
            If None, uses default configuration.
        sequences: Pre-extracted sequences as list of (seq_len, n_features) arrays.
            If None and extractor has sequence_columns configured, sequences will be
            extracted automatically using extractor.get_sequences().
        use_ordinary_models: Whether to train regular models
        use_mlframe_ensembles: Whether to create ensembles
        target_type: Explicit ``TargetTypes`` override. None auto-detects from the extractor;
            LEARNING_TO_RANK routes to the ranker suite. See ``TargetTypes``.
        ranking_config: LTR-only dispatch knobs (objectives, eval cutoffs, rank fusion). See ``LearningToRankConfig``.
        preprocessing_config: Preprocessing configuration. Holds custom transformer overrides
            (``scaler``, ``imputer``, ``category_encoder``).
        split_config: Train/val/test split configuration
        pipeline_config: Pipeline configuration
        preprocessing_extensions: Optional FE-extension transforms (PySR, polynomial, etc.). See ``PreprocessingExtensionsConfig``.
        auxiliary_events_df: Optional separate reference table (e.g. an entity x item interaction
            log distinct from the row-per-sample training frame) consumed by
            ``preprocessing_extensions`` steps that need it -- ``latent_interaction_svd`` and
            ``nearest_past_join``. None (default) is a genuine no-op for both.
        feature_types_config: Numeric/categorical/text type-detection overrides. See ``FeatureTypesConfig``.
        linear_model_config: Linear-model family hyperparameters. See ``LinearModelConfig``.
        hyperparams_config: Model hyperparameters (iterations, learning rate, per-model kwargs).
        behavior_config: Training behavior flags (GPU preference, calibration, fairness).
        training_config: Aggregate ``TrainingConfig`` carrier; supplies ``linear_model_config`` /
            ``behavior_config`` defaults from its ``.linear_config`` / ``.behavior`` sub-configs
            when those explicit kwargs are left as None (an explicit kwarg always wins).
        multilabel_dispatch_config: Multilabel-only strategy (wrapper/chain/native). See ``MultilabelDispatchConfig``.
        reporting_config: Calibration / training-report look. Holds figure size, chart toggles,
            title-metrics template, histogram subplot toggles, inline population labels, FI plot config.
        output_config: Filesystem destinations - ``data_dir``, ``models_dir``, ``plot_file``, ``save_charts``.
        outlier_detection_config: Outlier-detector + ``apply_to_val``.
        feature_selection_config: Holds ``use_mrmr_fs``, ``mrmr_kwargs``, ``rfecv_models``,
            ``rfecv_kwargs``, ``custom_pre_pipelines``.
        confidence_analysis_config: SHAP-based confidence-of-correct-prediction analysis. See ``ConfidenceAnalysisConfig``.
        baseline_diagnostics_config: Baseline ablation / quick-model diagnostics. See ``BaselineDiagnosticsConfig``.
        dummy_baselines_config: Per-target dummy-baseline computation. See ``DummyBaselinesConfig``.
        quantile_regression_config: Quantile-regression alphas / crossing-fix / coverage. See ``QuantileRegressionConfig``.
        conformal_config: Conformal prediction intervals (regression) / sets (classification) plus
            achieved coverage into ``metadata["conformal"]``; default ON. See ``ConformalConfig``.
        regression_calibration_config: point recalibration, OFF by default; its apply_confidence_shrinkage is ON by default and always applies.
        composite_target_discovery_config: Composite-target (diff/ratio/linres) discovery. ``MLFRAME_DISABLE_COMPOSITE=1`` forces off. See ``CompositeTargetDiscoveryConfig``.
        feature_handling_config: Feature-handling / caching config bundle (advanced). See the feature_handling package.
        precomputed: Opt-in ``TrainMlframeSuitePrecomputed`` bundle for repeated-suite-on-same-train
            benchmarking. When supplied, each non-None field skips the matching in-suite compute step:
            ``trainset_features_stats`` (skips the stats pass), ``dummy_baselines`` (skips the per-target
            dummy-baseline pass by short-circuiting via ``dummy_baselines_config.enabled=False`` and
            pre-seeding ``metadata["dummy_baselines"]``), ``composite_target_specs`` (skips the
            composite discovery phase and pre-seeds ``metadata["composite_target_specs"]``). Build the
            bundle via ``mlframe.training.helpers.precompute_all`` or by hand from a prior run's
            metadata. None (default) preserves legacy behaviour: every step computes inline.
        enable_target_distribution_analyzer: When True (default), run the mini-HPT target-distribution analyzer
            (heavy-tail / multi-modal / strong-AR / imbalance detection) and merge gap-fill hyperparameter recommendations.
        verbose: Verbosity level (0=silent, 1=info, 2=debug). Mapped to a root logging level at entry
            (1->INFO, 2->DEBUG); per-site gates still distinguish only silent vs non-silent.

    Side effects (process-wide, irreversible):
        Running this suite installs idempotent monkeypatches on installed third-party libraries that persist
        for the process lifetime and affect EVERY consumer of those libraries in the same process (not just
        mlframe): a patched ``lightgbm.sklearn.LGBMModel.feature_names_in_`` property, logging wrappers on
        ``catboost.Pool`` / ``xgboost.DMatrix`` / ``lightgbm.Dataset`` constructors, and a
        ``joblib.externals.loky`` physical-core override. There is no un-patch path. Bare
        ``import mlframe.training`` does NOT patch anything (the patches are lazy, applied on first suite call).
        Where mlframe controls the call site, prefer the ``make_pool`` / ``make_dmatrix`` / ``make_lgb_dataset``
        factories in ``_model_factories`` over the global constructor wrapping.

    Returns:
        A ``SuiteResult`` — a NamedTuple that is fully back-compatible with the historical
        ``(models_dict, metadata_dict)`` 2-tuple (unpacks + indexes identically) and adds
        ``.models`` / ``.metadata`` / ``get_model(name, task=None)`` / ``best_model(task=None)``.

    Example:
        ```python
        models, metadata = train_mlframe_models_suite(
            df="data.parquet",
            target_name="target",
            model_name="experiment_1",
            features_and_targets_extractor=my_ft_extractor,
            mlframe_models=["linear", "ridge", "cb", "lgb"],
            preprocessing_config=PreprocessingConfig(fillna_value=0.0),
            split_config=TrainingSplitConfig(test_size=0.1, val_size=0.1),
            reporting_config=ReportingConfig(
                title_metrics_template="ICE BR ECE CMAEW",
            ),
            output_config=OutputConfig(data_dir="./artifacts", save_charts=True),
        )
        ```
"""
