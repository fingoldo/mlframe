"""
Core training functions for mlframe.

Contains the refactored train_mlframe_models_suite function.
"""

# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------


import os
import psutil
import joblib
import pandas as pd
import polars as pl
from typing import Union, Optional, List, Dict, Any, Tuple
from collections import defaultdict
from os.path import join
from pyutilz.system import clean_ram, tqdmu
from pyutilz.strings import slugify
from sklearn.pipeline import Pipeline
import category_encoders as ce

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from pyutilz.system import ensure_dir_exists
from pyutilz.system import ensure_dir_exists

from .configs import (
    PreprocessingConfig,
    TrainingSplitConfig,
    PolarsPipelineConfig,
    TrainingConfig,
)
from .preprocessing import (
    load_and_prepare_dataframe,
    preprocess_dataframe,
    save_split_artifacts,
    create_split_dataframes,
)
from .pipeline import fit_and_transform_pipeline
from mlframe.feature_selection.filters import MRMR
from .utils import log_ram_usage, log_phase, drop_columns_from_dataframe
from .models import is_linear_model, train_linear_model, LINEAR_MODEL_TYPES
from ..training_old import process_model, select_target, score_ensemble, make_train_test_split, FeaturesAndTargetsExtractor, TargetTypes


def _ensure_config(config, config_class, kwargs):
    """
    Convert dict/None to Pydantic config object.

    Args:
        config: Config object, dict, or None
        config_class: Pydantic config class to instantiate
        kwargs: Keyword arguments to extract config fields from

    Returns:
        Pydantic config object
    """
    if isinstance(config, dict):
        return config_class(**config)
    elif config is None:
        # Extract only fields that belong to this config class
        return config_class(**{k: v for k, v in kwargs.items() if k in config_class.model_fields})
    return config


def train_mlframe_models_suite(
    df: Union[pl.DataFrame, pd.DataFrame, str],
    target_name: str,
    model_name: str,
    features_and_targets_extractor: FeaturesAndTargetsExtractor,
    # Model selection
    mlframe_models: Optional[List[str]] = None,
    use_ordinary_models: bool = True,
    use_mlframe_ensembles: bool = True,
    # Configurations (can be dicts or Pydantic objects)
    preprocessing_config: Optional[Union[PreprocessingConfig, Dict]] = None,
    split_config: Optional[Union[TrainingSplitConfig, Dict]] = None,
    pipeline_config: Optional[Union[PolarsPipelineConfig, Dict]] = None,
    # Feature selection
    use_mrmr_fs: bool = False,
    mrmr_kwargs: Optional[Dict] = None,
    rfecv_models: Optional[List[str]] = None,
    # Override parameters (for backward compatibility)
    config_params: Optional[Dict] = None,
    control_params: Optional[Dict] = None,
    config_params_override: Optional[Dict] = None,
    control_params_override: Optional[Dict] = None,
    init_common_params: Optional[Dict] = None,
    # Paths
    data_dir: str = "",
    models_dir: str = "models",
    # Misc
    verbose: int = 1,
    # Backward compatibility parameters
    **kwargs,
) -> Tuple[Dict, Dict]:
    """
    Train a suite of ML models on a dataset.

    This is the refactored main training function with cleaner, more modular code.

    Args:
        df: DataFrame or path to parquet file
        target_name: Name of the target to predict
        model_name: Base name for the models
        features_and_targets_extractor: FeaturesAndTargetsExtractor instance for computing targets

        mlframe_models: List of model types to train (cb, lgb, xgb, mlp, hgb, linear, ridge, etc.)
        use_ordinary_models: Whether to train regular models
        use_mlframe_ensembles: Whether to create ensembles

        preprocessing_config: Preprocessing configuration
        split_config: Train/val/test split configuration
        pipeline_config: Pipeline configuration

        use_mrmr_fs: Whether to use MRMR feature selection
        mrmr_kwargs: MRMR parameters
        rfecv_models: Models to use for RFECV

        config_params_override: Override for model config parameters
        control_params_override: Override for control parameters
        init_common_params: Common initialization parameters

        data_dir: Directory for saving artifacts
        models_dir: Directory for saving models

        verbose: Verbosity level

    Returns:
        Tuple of (models_dict, metadata_dict)

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
        )
        ```
    """

    # ==================================================================================
    # 1. CONFIGURATION SETUP
    # ==================================================================================

    if verbose:
        log_phase(f"Starting mlframe training suite: {model_name}")

    # Convert dict configs to Pydantic if needed
    preprocessing_config = _ensure_config(preprocessing_config, PreprocessingConfig, kwargs)
    split_config = _ensure_config(split_config, TrainingSplitConfig, kwargs)
    pipeline_config = _ensure_config(pipeline_config, PolarsPipelineConfig, kwargs)

    # Default models
    if mlframe_models is None:
        mlframe_models = ["cb", "lgb", "xgb", "mlp", "linear"]

    # Metadata for tracking
    metadata = {
        "model_name": model_name,
        "target_name": target_name,
        "mlframe_models": mlframe_models,
        "configs": {
            "preprocessing": preprocessing_config,
            "pipeline": pipeline_config,
            "split": split_config,
        },
    }

    # ==================================================================================
    # 2. DATA LOADING & PREPROCESSING
    # ==================================================================================

    if verbose:
        log_phase("PHASE 1: Data Loading & Preprocessing")

    # Load and prepare dataframe
    df = load_and_prepare_dataframe(df, preprocessing_config, verbose=verbose)

    # Apply features_and_targets_extractor to extract targets
    if verbose:
        logger.info("Create additional features & extracting targets...")

    df, target_by_type, group_ids_raw, group_ids, timestamps, artifacts, additional_columns_to_drop = features_and_targets_extractor.transform(df)

    clean_ram()
    if verbose:
        log_ram_usage()

    # Drop columns AFTER features_and_targets_extractor (columns might be needed by features_and_targets_extractor or created by it)
    df = drop_columns_from_dataframe(
        df,
        additional_columns_to_drop=additional_columns_to_drop,
        config_drop_columns=preprocessing_config.drop_columns,
        verbose=verbose,
    )

    # Preprocess dataframe (handle nulls, infinities, constants, dtypes)
    df = preprocess_dataframe(df, preprocessing_config, verbose=verbose)

    # ==================================================================================
    # 3. TRAIN/VAL/TEST SPLITTING
    # ==================================================================================

    if verbose:
        log_phase("PHASE 2: Train/Val/Test Splitting")

    if verbose:
        logger.info(f"Making train_val_test split...")
    train_idx, val_idx, test_idx, train_details, val_details, test_details = make_train_test_split(
        df=df,
        timestamps=timestamps,
        **split_config.model_dump(),
    )
    if verbose:
        log_ram_usage()

    # Save artifacts
    if data_dir:
        save_split_artifacts(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            timestamps=timestamps,
            group_ids_raw=group_ids_raw,
            artifacts=artifacts,
            data_dir=data_dir,
            models_dir=models_dir,
            target_name=target_name,
            model_name=model_name,
        )

    metadata.update(
        {
            "train_details": train_details,
            "val_details": val_details,
            "test_details": test_details,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
        }
    )

    # Create split dataframes
    train_df, val_df, test_df = create_split_dataframes(
        df=df,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    # Delete original df to free RAM
    if verbose:
        logger.info("Deleting original DataFrame to free RAM...")

    del df
    clean_ram()

    if verbose:
        log_ram_usage()

    # ==================================================================================
    # 4. PIPELINE FITTING & TRANSFORMATION
    # ==================================================================================

    if verbose:
        log_phase("PHASE 3: Pipeline Fitting & Transformation")

    train_df, val_df, test_df, pipeline, cat_features = fit_and_transform_pipeline(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config=pipeline_config,
        ensure_float32=preprocessing_config.ensure_float32_dtypes,
        verbose=verbose,
    )

    metadata["pipeline"] = pipeline
    metadata["cat_features"] = cat_features
    metadata["columns"] = train_df.columns.tolist() if isinstance(train_df, pd.DataFrame) else train_df.columns

    clean_ram()
    if verbose:
        log_ram_usage()

    # ==================================================================================
    # 5. MODEL TRAINING
    # ==================================================================================

    if verbose:
        log_phase("PHASE 4: Model Training")

    # Initialize pipeline components for MLP/NGB models
    trainset_features_stats = None

    # Extract from init_common_params or use defaults
    if init_common_params is None:
        init_common_params = {}

    category_encoder = init_common_params.get("category_encoder", None)
    imputer = init_common_params.get("imputer", None)
    scaler = init_common_params.get("scaler", None)

    # Initialize defaults if not provided
    if category_encoder is None and cat_features:
        category_encoder = ce.CatBoostEncoder()

    if imputer is None:
        imputer = SimpleImputer()

    if scaler is None:
        scaler = StandardScaler()

    # Default rfecv_models if not provided
    if rfecv_models is None:
        rfecv_models = []

    # Default mrmr_kwargs if not provided
    if mrmr_kwargs is None:
        mrmr_kwargs = dict(n_workers=max(1, psutil.cpu_count(logical=False)), verbose=2, fe_max_steps=0)

    # Initialize control_params and config_params if not provided
    if control_params is None:
        control_params = {}
    if config_params is None:
        config_params = {}
    if control_params_override is None:
        control_params_override = {}
    if config_params_override is None:
        config_params_override = {}
    if init_common_params is None:
        init_common_params = {}

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Actual training
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    # Cache pandas versions for select_target (avoid repeated conversion in loop)
    train_df_pd = train_df if isinstance(train_df, pd.DataFrame) else train_df.to_pandas()
    val_df_pd = val_df if val_df is None or isinstance(val_df, pd.DataFrame) else val_df.to_pandas()
    test_df_pd = test_df if test_df is None or isinstance(test_df, pd.DataFrame) else test_df.to_pandas()

    models = defaultdict(lambda: defaultdict(list))
    for target_type, targets in tqdmu(target_by_type.items(), desc="target type"):
        # !TODO ! optimize for creation of inner feature matrices of cb,lgb,xgb here. They should be created once per featureset, not once per target.
        for cur_target_name, cur_target_values in tqdmu(targets.items(), desc="target"):
            if mlframe_models:
                parts = slugify(target_name), slugify(model_name), slugify(target_type.lower()), slugify(cur_target_name)
                if data_dir is not None:
                    plot_file = join(data_dir, "charts", *parts) + os.path.sep
                    ensure_dir_exists(plot_file)
                else:
                    plot_file = None
                if models_dir is not None:
                    model_file = join(data_dir, models_dir, *parts) + os.path.sep
                    ensure_dir_exists(model_file)
                else:
                    model_file = None

                if verbose:
                    logger.info(f"select_target...")

                common_params, models_params, rfecv_models_params, cpu_configs, gpu_configs = select_target(
                    model_name=f"{target_name} {model_name} {cur_target_name}",
                    target=cur_target_values,
                    target_type=target_type,
                    df=None,
                    train_df=train_df_pd,
                    val_df=val_df_pd,
                    test_df=test_df_pd,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                    train_details=train_details,
                    val_details=val_details,
                    test_details=test_details,
                    group_ids=group_ids,
                    cat_features=cat_features,
                    config_params=config_params,
                    config_params_override=config_params_override,
                    control_params=control_params,
                    control_params_override=control_params_override,
                    common_params=dict(trainset_features_stats=trainset_features_stats, plot_file=plot_file, **init_common_params),
                )

            if verbose:
                log_ram_usage()

                pre_pipelines = []
                pre_pipeline_names = []

                if use_ordinary_models:
                    pre_pipelines.append(None)
                    pre_pipeline_names.append("")

                for rfecv_model_name in rfecv_models:
                    if rfecv_model_name not in rfecv_models_params:
                        logger.warning(f"RFECV model {rfecv_model_name} not known, skipping...")
                    else:
                        pre_pipelines.append(rfecv_models_params[rfecv_model_name])
                        pre_pipeline_names.append(f"{rfecv_model_name} ")

                if use_mrmr_fs:
                    pre_pipelines.append(MRMR(**mrmr_kwargs))
                    pre_pipeline_names.append("MRMR ")

                for pre_pipeline, pre_pipeline_name in zip(pre_pipelines, pre_pipeline_names):
                    if pre_pipeline_name == "cb_rfecv" and target_type == TargetTypes.REGRESSION and control_params_override.get("metamodel_func") is not None:
                        # File /venv/main/lib/python3.12/site-packages/sklearn/base.py:142, in _clone_parametrized(estimator, safe)
                        # RuntimeError: Cannot clone object <catboost.core.CatBoostRegressor object at 0x713048b0e840>, as the constructor either does not set or modifies parameter custom_metric
                        continue
                    ens_models = [] if use_mlframe_ensembles else None
                    orig_pre_pipeline = pre_pipeline

                    # Initialize caches for transformed DataFrames (reset for each pre_pipeline_name)

                    cached_hgb_dfs = None  # For hgb
                    cached_mlp_ngb_dfs = None  # For mlp, ngb
                    cached_original_dfs = None  # For cb, lgb, xgb, etc.

                    for mlframe_model_name in mlframe_models:
                        if mlframe_model_name == "cb" and target_type == TargetTypes.REGRESSION and control_params_override.get("metamodel_func") is not None:
                            continue
                        if mlframe_model_name not in models_params:
                            logger.warning(f"mlframe model {mlframe_model_name} not known, skipping...")
                        else:
                            # Determine pipeline type and caching strategy
                            if mlframe_model_name == "hgb" and cat_features:
                                pre_pipeline = Pipeline(
                                    steps=[
                                        *([("pre", orig_pre_pipeline)] if orig_pre_pipeline else []),
                                        ("ce", category_encoder),
                                    ]
                                )
                                use_cache = cached_hgb_dfs
                                cache_key = "hgb"
                            elif mlframe_model_name in ("mlp", "ngb") or is_linear_model(mlframe_model_name):
                                pre_pipeline = Pipeline(
                                    steps=[
                                        *([("pre", orig_pre_pipeline)] if orig_pre_pipeline else []),
                                        *([("ce", category_encoder)] if cat_features else []),
                                        *([("imp", imputer)] if imputer else []),
                                        *([("scaler", scaler)] if scaler else []),
                                    ]
                                )
                                use_cache = cached_mlp_ngb_dfs
                                cache_key = "mlp_ngb"
                            else:
                                # For tree models (cb, lgb, xgb) that handle NaN natively
                                pre_pipeline = orig_pre_pipeline
                                use_cache = cached_original_dfs
                                cache_key = "original"

                            # Call process_model with caching
                            if use_cache is not None:
                                # Use cached DataFrames
                                trainset_features_stats, pre_pipeline, train_df_transformed, val_df_transformed, test_df_transformed = process_model(
                                    model_file=model_file,
                                    model_name=mlframe_model_name,
                                    target_type=target_type,
                                    pre_pipeline=pre_pipeline,
                                    pre_pipeline_name=pre_pipeline_name,
                                    cur_target_name=cur_target_name,
                                    models=models,
                                    model_params=models_params[mlframe_model_name],
                                    common_params=common_params,
                                    ens_models=ens_models,
                                    trainset_features_stats=trainset_features_stats,
                                    verbose=verbose,
                                    skip_pre_pipeline_transform=True,
                                    cached_train_df=use_cache[0],
                                    cached_val_df=use_cache[1],
                                    cached_test_df=use_cache[2],
                                )
                            else:
                                # First model of this type - fit and cache
                                trainset_features_stats, pre_pipeline, train_df_transformed, val_df_transformed, test_df_transformed = process_model(
                                    model_file=model_file,
                                    model_name=mlframe_model_name,
                                    target_type=target_type,
                                    pre_pipeline=pre_pipeline,
                                    pre_pipeline_name=pre_pipeline_name,
                                    cur_target_name=cur_target_name,
                                    models=models,
                                    model_params=models_params[mlframe_model_name],
                                    common_params=common_params,
                                    ens_models=ens_models,
                                    trainset_features_stats=trainset_features_stats,
                                    verbose=verbose,
                                )
                                # Cache the transformed DataFrames
                                if cache_key == "hgb":
                                    cached_hgb_dfs = (train_df_transformed, val_df_transformed, test_df_transformed)
                                elif cache_key == "mlp_ngb":
                                    cached_mlp_ngb_dfs = (train_df_transformed, val_df_transformed, test_df_transformed)
                                else:
                                    cached_original_dfs = (train_df_transformed, val_df_transformed, test_df_transformed)

                            if mlframe_model_name not in ("hgb", "mlp", "ngb"):
                                orig_pre_pipeline = pre_pipeline

                    if ens_models and len(ens_models) > 1:
                        if verbose:
                            logger.info(f"evaluating simple ensembles...")
                        ensembles = score_ensemble(
                            models_and_predictions=ens_models,
                            ensemble_name=pre_pipeline_name + f"{len(ens_models)}models ",
                            **common_params,
                        )

    # ==================================================================================
    # 6. FINALIZATION
    # ==================================================================================

    # Save metadata
    if data_dir and models_dir:
        metadata_file = join(data_dir, models_dir, slugify(target_name), slugify(model_name), "metadata.joblib")
        joblib.dump(metadata, metadata_file)
        if verbose:
            logger.info(f"Saved metadata to {metadata_file}")

    if verbose:
        log_phase(f"Training suite completed for {model_name}, {sum(len(v) for targets in models.values() for v in targets.values())} models.")
        log_ram_usage()

    return dict(models), metadata


__all__ = [
    "train_mlframe_models_suite",
]
