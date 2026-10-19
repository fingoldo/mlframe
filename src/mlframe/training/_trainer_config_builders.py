"""Stages of ``_build_configs_from_params``."""

from __future__ import annotations

# --- imports (managed) ---
from functools import partial
from mlframe.training.configs import ConfidenceAnalysisConfig
from mlframe.training.configs import DataConfig
from mlframe.training.configs import FeatureImportanceConfig
from mlframe.training.configs import MetricsConfig
from mlframe.training.configs import OutputConfig
from mlframe.training.configs import TrainingControlConfig
from mlframe.training.trainer import _reporting_config_with_routed_kwargs
from mlframe.training.trainer import nn
from mlframe.training.trainer import torch
# --- end imports ---


def _build_reporting_config(feature_importance_config, figsize, print_report, show_perf_chart, show_fi, display_sample_size, show_feature_names, show_prob_histogram, prob_histogram_yscale, show_inline_population_labels, plot_dpi, title_metrics_template, plot_outputs, multiclass_panels, multilabel_panels, ltr_panels, quantile_panels, honest_estimator_diagnostics, mase_seasonality, _unused_reporting_kwargs):
    """Stage of ``_build_configs_from_params``."""
    if feature_importance_config is None:
        fi_cfg = FeatureImportanceConfig()
    elif isinstance(feature_importance_config, dict):
        fi_cfg = FeatureImportanceConfig(**feature_importance_config)
    else:
        fi_cfg = feature_importance_config

    _rep_kwargs = dict(
        figsize=figsize,
        print_report=print_report,
        show_perf_chart=show_perf_chart,
        show_fi=show_fi,
        feature_importance_config=fi_cfg,
        display_sample_size=display_sample_size,
        show_feature_names=show_feature_names,
        show_prob_histogram=show_prob_histogram,
        prob_histogram_yscale=prob_histogram_yscale,
        show_inline_population_labels=show_inline_population_labels,
        plot_dpi=plot_dpi,
    )
    # Pass only when explicitly set; otherwise ReportingConfig's field default is the single source.
    _rep_kwargs.update(
        {
            _name: _val
            for _name, _val in (
                ("title_metrics_template", title_metrics_template),
                ("plot_outputs", plot_outputs),
                ("multiclass_panels", multiclass_panels),
                ("multilabel_panels", multilabel_panels),
                ("ltr_panels", ltr_panels),
                ("quantile_panels", quantile_panels),
            )
            if _val is not None
        }
    )
    if honest_estimator_diagnostics is not None:
        _rep_kwargs["honest_estimator_diagnostics"] = honest_estimator_diagnostics
    if mase_seasonality is not None:
        _rep_kwargs["mase_seasonality"] = int(mase_seasonality)
    # Generic catch-all routing: any leftover kwarg that names a real
    # ReportingConfig field is forwarded so new fields auto-propagate
    # without a per-field gate above. Truly unknown keys are dropped at
    # DEBUG (the splat must survive future additions), never raised.
    reporting_config = _reporting_config_with_routed_kwargs(_rep_kwargs, _unused_reporting_kwargs)
    return reporting_config


def _build_data_config(df, train_df, val_df, test_df, target, train_target, val_target, test_target, train_idx, val_idx, test_idx, calib_df, calib_target, calib_idx, calib_df_pre_pipeline, group_ids, sample_weight, timestamps, merged_drop_columns, target_label_encoder, skip_infinity_checks, n_features, target_type):
    """Stage of ``_build_configs_from_params``."""
    data_config = DataConfig(
        df=df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target=target,
        train_target=train_target,
        val_target=val_target,
        test_target=test_target,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        calib_df=calib_df,
        calib_target=calib_target,
        calib_idx=calib_idx,
        calib_df_pre_pipeline=calib_df_pre_pipeline,
        group_ids=group_ids,
        sample_weight=sample_weight,
        timestamps=timestamps,
        drop_columns=merged_drop_columns,
        target_label_encoder=target_label_encoder,
        skip_infinity_checks=skip_infinity_checks,
        n_features=n_features,
        target_type=str(target_type) if target_type is not None else None,
    )
    return data_config


def _build_control_config(verbose, use_cache, just_evaluate, compute_trainset_metrics, compute_valset_metrics, compute_testset_metrics, pre_pipeline, skip_pre_pipeline_transform, skip_preprocessing, fit_params, callback_params, model_category, slice_stable_es, row_wise_extensions_config):
    """Stage of ``_build_configs_from_params``."""
    control_config = TrainingControlConfig(
        verbose=verbose,
        use_cache=use_cache,
        just_evaluate=just_evaluate,
        compute_trainset_metrics=compute_trainset_metrics,
        compute_valset_metrics=compute_valset_metrics,
        compute_testset_metrics=compute_testset_metrics,
        pre_pipeline=pre_pipeline,
        skip_pre_pipeline_transform=skip_pre_pipeline_transform,
        skip_preprocessing=skip_preprocessing,
        fit_params=fit_params,
        callback_params=callback_params,
        model_category=model_category,
        slice_stable_es=slice_stable_es,
        row_wise_extensions_config=row_wise_extensions_config,
    )
    return control_config


def _build_metrics_config(nbins, custom_ice_metric, custom_rice_metric, subgroups, train_details, val_details, test_details):
    """Stage of ``_build_configs_from_params``."""
    metrics_config = MetricsConfig(
        nbins=nbins,
        custom_ice_metric=custom_ice_metric,
        custom_rice_metric=custom_rice_metric,
        subgroups=subgroups,
        train_details=train_details,
        val_details=val_details,
        test_details=test_details,
    )
    return metrics_config


def _build_output_config(plot_file, data_dir, models_subdir):
    """Stage of ``_build_configs_from_params``."""
    output_config = OutputConfig(
        plot_file=plot_file or "",
        data_dir=data_dir or "",
        # ``models_subdir or "models"``
        # silently rewrote a legitimate ``models_subdir=""`` (intent: write
        # models flat in data_dir, no subfolder) to ``"models"`` subdir.
        # Use explicit None-check so the empty-string intent is preserved.
        # ``plot_file or ""`` and ``data_dir or ""`` ABOVE intentionally
        # collapse falsy -> "" because their None and "" both mean
        # "feature disabled"; ``models_subdir`` is asymmetric -- None
        # means "use default" but "" is a real caller choice.
        models_dir="models" if models_subdir is None else models_subdir,
    )
    return output_config


def _build_confidence_config(include_confidence_analysis, confidence_analysis_use_shap, confidence_analysis_max_features, confidence_analysis_cmap, confidence_analysis_alpha, confidence_analysis_ylabel, confidence_analysis_title, confidence_model_kwargs):
    """Stage of ``_build_configs_from_params``."""
    confidence_config = ConfidenceAnalysisConfig(
        include=include_confidence_analysis,
        use_shap=confidence_analysis_use_shap,
        max_features=confidence_analysis_max_features,
        cmap=confidence_analysis_cmap,
        alpha=confidence_analysis_alpha,
        ylabel=confidence_analysis_ylabel,
        title=confidence_analysis_title,
        model_kwargs=confidence_model_kwargs or {},
    )
    return confidence_config


def _default_mlp_network_params(_arch_cls):
    """Stage of ``_configure_mlp_params``."""
    mlp_network_params = dict(
        nlayers=2,
        first_layer_num_neurons=128,
        min_layer_neurons=16,
        neurons_by_layer_arch=_arch_cls.Declining,
        consec_layers_neurons_ratio=2.0,
        # activation chain history is
        #   LeakyReLU -> Tanh -> GELU + spectral_norm=True.
        # GELU is unbounded above but smoother than ReLU (gradient flows
        # everywhere). To keep the unseen-group test-split safety that
        # Tanh used to provide via saturation, ``spectral_norm=True``
        # (set below) caps each Linear's largest singular value at 1.0
        # via power iteration. Combined with GELU's Lipschitz-1
        # property the whole network is globally Lipschitz with bound
        # = depth, so OOD inputs cannot amplify magnitude geometrically.
        # The output stays bounded by ``output_activation='tanh_train_range'``
        # below, which gives a SECOND hard cap in y-scale.
        # Users may revert to Tanh via
        #   mlp_kwargs["network_params"]["activation_function"]=torch.nn.Tanh
        # or fully turn SN off via
        #   mlp_kwargs["network_params"]["spectral_norm"]=False.
        activation_function=torch.nn.GELU,
        # Kaiming-normal with relu nonlinearity is the standard
        # recommendation for GELU (close enough to relu's gain). Tanh
        # used xavier_normal_; switching to GELU + kaiming_normal_
        # together keeps init / activation aligned.
        weights_init_fcn=partial(
            nn.init.kaiming_normal_, nonlinearity="relu",
        ),
        # SN on by default: bounds Lipschitz constant of
        # the linear maps to 1.0 (after power-iter convergence), making
        # catastrophic OOD extrapolation (R^2=-326 / -30 historical
        # regressions) geometrically impossible.
        spectral_norm=True,
        dropout_prob=0.0,
        inputs_dropout_prob=0.0,
        # ``use_batchnorm=True`` is the suite default.
        # Prior default (False) + LeakyReLU + Declining 128->64->32->16
        # + Adam + kaiming-normal init produced saturated inner pre-
        # activations on a 4.1M-row, 206-feature TVT regression: with
        # ``output_activation='tanh_train_range'`` firing under the
        # extreme-AR gate, the inner net's unbounded outputs got hard-
        # capped to +-1 in standardised space and the destandardised
        # predictions clustered at the rails [y_mid - scale, y_mid +
        # scale] (R^2 = -30.84 vs Ridge R^2 = 1.00 on identical data).
        # BatchNorm normalises per-FEATURE across the batch -- exactly
        # the axis that LN destroys (LN is per-row across features and
        # was correctly disabled below). BN keeps inner pre-activations
        # in a usable range and prevents the saturation. Users may opt
        # out via ``mlp_kwargs["network_params"]["use_batchnorm"]=False``.
        use_batchnorm=True,
        # ``use_layernorm=False`` for the suite default.
        # ``generate_mlp`` defaults LN_in to True for transformer-style row-
        # independent batches, but it is WRONG for tabular regression: LN
        # normalises per-row across features, destroying the inter-row
        # absolute-scale signal that strong-linear / additive targets rely
        # on. Observed in prod (4M-row regression, y near-perfect
        # auto-regressive in a lag feature): LN_in + 5-epoch time budget +
        # group split collapsed MLP to a tight pred cluster (std ~0.17 *
        # y_std), test R^2 = -4.75 vs Ridge R^2 = 1.00 on identical data.
        # Synthetic-bench confirmed: LN_in off -> test R^2 =
        # 0.99 (matches Ridge); LN_in on -> 0.95 baseline that under
        # specific stress patterns collapses entirely. Pre-pipeline
        # already applies StandardScaler per-feature; further per-row
        # LayerNorm is double-norm noise. Users who really need LN can
        # opt in via ``mlp_kwargs["network_params"]["use_layernorm"]=True``.
        use_layernorm=False,
        # output bounded by default. The tanh_train_range
        # output activation hard-caps the regression head to
        # ``[(y_min+y_max)/2 - ((y_max-y_min)/2 + 3*y_std),
        #   (y_min+y_max)/2 + ((y_max-y_min)/2 + 3*y_std)]``. scale +
        # center are auto-derived from y_train in
        # ``neural.base._fit_inner_network`` when caller leaves them
        # unset, so this is a zero-config default. Previously only the
        # extreme-AR gate enabled this; making it the default closes
        # the failure mode at the source for ALL regression runs
        # (envelope-clip remains as a downstream defence). Users may
        # opt out via ``mlp_kwargs["network_params"]["output_activation"]="linear"``.
        output_activation="tanh_train_range",
    )
    return mlp_network_params
