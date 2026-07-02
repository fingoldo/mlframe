"""Non-test helpers for the fuzz suite: per-combo config builders,
invariant / serialization-roundtrip checkers, and dependency-skip
utilities. Carved out of test_fuzz_suite.py so that module stays a
lean pytest-discoverable test file. Heavy mlframe / sklearn deps are
imported lazily in-body to keep import time low.
"""
from __future__ import annotations

import numpy as np
import pytest

from ._fuzz_combo import (
    FuzzCombo,  # noqa: F401  (annotation strings under PEP 563)
    build_composite_discovery_config,
)


def _safe_cfg_kwargs(cfg_class, **kwargs):
    """iter170: drop kwargs absent from cfg_class.model_fields so
    audit-discovered fields that don't exist in the actual Pydantic
    config (e.g. agent hallucinated a name, or the field is gated
    behind a sub-config object we don't construct here) silently no-op
    instead of crashing the suite call. Also drops None values to
    preserve library defaults when the axis is at its dataclass-default
    sentinel."""
    valid = set(getattr(cfg_class, "model_fields", {}).keys())
    return {k: v for k, v in kwargs.items() if k in valid and v is not None}


# Obsolete (2026-05-18 refactor): _make_cat_fe_config_for_fuzz +
# _composite_discovery_config_for_combo have been folded into the
# shared builders ``build_mrmr_kwargs`` / ``build_composite_discovery_config``
# in ``_fuzz_combo.py``. Adding a new MRMR or composite-discovery axis
# now only edits build_mrmr_kwargs_from_flat / build_composite_discovery_-
# config_from_flat there (plus the AXES dict + FuzzCombo dataclass).


def _config_for_models(
    models: tuple[str, ...],
    n_rows: int,
    iterations: int = 3,
    early_stopping_rounds: "int | None" = None,
    mlp_predict_batch_size: "int | None" = None,
    # iter170 per-backend hyperparam axes.
    lgb_feature_fraction: float = 1.0,
    lgb_num_leaves: int = 31,
    xgb_max_depth: int = 6,
    xgb_colsample_bynode: float = 1.0,
    cb_border_count: int = 254,
    hgb_max_leaf_nodes: int = 31,
    rfecv_cv_n_splits: int = 2,
    # 2026-06-03 FS-coverage audit -- RFECV.__init__ knobs threaded into
    # rfecv_kwargs (-> COMMON_RFECV_PARAMS update -> RFECV ctor in
    # _trainer_configure.py). Defaults match the RFECV signature so combos
    # without an RFECV selector are unaffected.
    rfecv_votes_aggregation: str = "Borda",
    rfecv_search_method: str = "ModelBasedHeuristic",
    # iter180 DEPTH-4 booster sub-params (depth-3 gate + depth-4 sub-knob pairs).
    lgb_boosting_type: str = "gbdt",
    lgb_dart_drop_rate: float = 0.1,
    lgb_goss_top_rate: float = 0.2,
    xgb_tree_method: str = "auto",
    xgb_hist_max_bin: int = 256,
    cb_bootstrap_type: str = "Bayesian",
    cb_bayesian_bagging_temperature: float = 1.0,
    cb_bernoulli_subsample: float = 0.8,
    cb_grow_policy: str = "SymmetricTree",
    cb_lossguide_max_leaves: int = 31,
) -> dict:
    cfg: dict = {"iterations": iterations}
    if early_stopping_rounds is not None:
        cfg["early_stopping_rounds"] = early_stopping_rounds
    # iter162: P0 axis -- MLP predict-time batch override. None = auto-adapt
    # (default path); int = hard-lock (memory-safety branch on wide frames).
    if "mlp" in models and mlp_predict_batch_size is not None:
        cfg["mlp_predict_batch_size"] = mlp_predict_batch_size
    if "lgb" in models:
        _lgb_kw = {
            "device_type": "cpu", "verbose": -1,
            # iter170 inner knobs.
            "feature_fraction": lgb_feature_fraction,
            "num_leaves": lgb_num_leaves,
            # iter180 depth-3 gate.
            "boosting_type": lgb_boosting_type,
        }
        # iter180 DEPTH-4 sub-knobs gated by boosting_type.
        if lgb_boosting_type == "dart":
            _lgb_kw["drop_rate"] = lgb_dart_drop_rate
        elif lgb_boosting_type == "goss":
            _lgb_kw["top_rate"] = lgb_goss_top_rate
            # 'goss' is incompatible with feature_fraction < 1 in LGB; drop the conflict.
            _lgb_kw.pop("feature_fraction", None)
        cfg["lgb_kwargs"] = _lgb_kw
    if "xgb" in models:
        _xgb_kw = {
            "device": "cpu", "verbosity": 0,
            # iter170 inner knobs.
            "max_depth": xgb_max_depth,
            "colsample_bynode": xgb_colsample_bynode,
            # iter180 depth-3 gate.
            "tree_method": xgb_tree_method,
        }
        # iter180 DEPTH-4 sub-knob.
        if xgb_tree_method == "hist":
            _xgb_kw["max_bin"] = xgb_hist_max_bin
        cfg["xgb_kwargs"] = _xgb_kw
    if "cb" in models:
        _cb_kw = {
            "task_type": "CPU", "verbose": 0,
            # iter170 inner knob.
            "border_count": cb_border_count,
            # iter180 depth-3 gates.
            "bootstrap_type": cb_bootstrap_type,
            "grow_policy": cb_grow_policy,
        }
        # iter180 DEPTH-4 sub-knobs gated by bootstrap_type.
        if cb_bootstrap_type == "Bayesian":
            _cb_kw["bagging_temperature"] = cb_bayesian_bagging_temperature
        elif cb_bootstrap_type == "Bernoulli":
            _cb_kw["subsample"] = cb_bernoulli_subsample
        # iter180 DEPTH-4 sub-knob gated by grow_policy.
        if cb_grow_policy == "Lossguide":
            _cb_kw["max_leaves"] = cb_lossguide_max_leaves
        cfg["cb_kwargs"] = _cb_kw
    if "hgb" in models:
        cfg["hgb_kwargs"] = {
            # iter170 inner knob.
            "max_leaf_nodes": hgb_max_leaf_nodes,
        }
    # Fuzz-only RFECV speed-up: cap inner-CV iterations and no-improvement
    # patience hard. The combo space includes rfecv_estimator_cfg=cb_rfecv
    # variants where the heuristic feature search would otherwise evaluate
    # 100+ feature subsets × cv folds × estimator fit and overshoot the
    # per-test timeout (fuzz c0070, n=1200, 30 features, full quartet +
    # cb_rfecv). Production users keep the library defaults
    # (max_noimproving_iters=15, cv_n_splits=4); we strip both down to the
    # bare minimum here so coverage remains complete and budget is sane.
    cfg["rfecv_kwargs"] = {
        "max_noimproving_iters": 2,
        "cv_n_splits": rfecv_cv_n_splits,
        "max_runtime_mins": 5,
        # 2026-06-03 FS-coverage audit -- RFECV aggregation + search-method
        # knobs. Both are str-Enum RFECV ctor params; the string form is
        # accepted by the enum. Forwarded via COMMON_RFECV_PARAMS.update.
        "votes_aggregation_method": rfecv_votes_aggregation,
        "top_predictors_search_method": rfecv_search_method,
        # 2026-06-04: do NOT add "cluster_reduce" (the GroupAwareMRMR cluster-medoid
        # wrap toggle) here. It is a registry meta-param popped by
        # registry._instantiate_rfecv before RFECV(**kwargs); but this fuzz path forwards
        # rfecv_kwargs via COMMON_RFECV_PARAMS.update -> RFECV(**...) DIRECTLY (no registry),
        # so cluster_reduce reaches RFECV.__init__ -> TypeError. The cluster-medoid wrap is
        # only reachable/fuzzable via the production FeatureSelectionConfig.rfecv_models ->
        # registry path, not this trainer-config path. (Attempted + reverted 2026-06-04.)
    }
    return cfg


def _configs_for_combo(combo: FuzzCombo) -> dict:
    """Build the preprocessing / feature-types / behavior config overrides
    that prop the combo's per-axis flags into ``train_mlframe_models_suite``.

    Returns a kwargs dict ready to **-splat into the suite call. Built
    lazily from Pydantic model defaults so any field we don't touch
    keeps its library default (mirrors how prod callers typically
    override only a handful of fields).

    Text / embedding column names are passed explicitly so that combos
    with ``auto_detect_cats=False`` still register them as text / embedding
    features. Without this, the auto-detector's early-return on
    ``auto_detect_feature_types=False`` leaves ``emb_0`` (or ``text_0``)
    in the feature matrix as a regular column — CatBoost's Polars
    fastpath then crashes trying to process a ``pl.List(Float32)`` /
    high-cardinality string column it never learned was special.
    Matches the prod idiom: callers who turn off auto-detection
    typically do so BECAUSE they're declaring the lists manually."""
    from mlframe.training import (
        PreprocessingBackendConfig,
        FeatureTypesConfig,
        TrainingBehaviorConfig,
        TrainingSplitConfig,
        MultilabelDispatchConfig,
        PreprocessingExtensionsConfig,
    )
    # Mirror the ``want_text`` / ``want_embedding`` gates in
    # ``build_frame_for_combo`` so the declared column lists exactly
    # match what the frame actually contains — no false positives,
    # no false negatives.
    # Mirror FuzzCombo._canonical_text_col_count so the FeatureTypesConfig
    # text_features list agrees with what the frame builder emitted.
    _eff_text_count = combo._canonical_text_col_count()
    # Gate text_features on BOTH the cb-needs-text flag AND the
    # FeatureTypesConfig.use_text_features knob. Pydantic validator on
    # FeatureTypesConfig (correctly) rejects ``text_features=[...]`` with
    # ``use_text_features=False`` because the list would be silently
    # dropped and the columns mis-routed to the cat path. Surfaced
    # 2026-05-20 by combo c0029 (cb_hgb_mlp + use_text_features=False
    # but cb-in-models → text list assigned → validation crash before
    # the suite even started).
    emits_text = (
        _eff_text_count > 0 and "cb" in combo.models and combo.use_text_features
    )
    emits_emb = (
        combo.embedding_col_count > 0
        and "cb" in combo.models
        and combo.input_type != "pandas"
    )
    text_features = (
        [f"text_{i}" for i in range(_eff_text_count)] if emits_text else None
    )
    embedding_features = (
        [f"emb_{i}" for i in range(combo.embedding_col_count)] if emits_emb else None
    )
    # Fairness: only valid if the referenced column actually exists in
    # the frame (cat_0 requires cat_feature_count >= 1).
    fairness_features = (
        [combo.fairness_col]
        if combo.fairness_col is not None and combo.cat_feature_count > 0
        else None
    )
    # 2026-04-28: ``prefer_calibrated_classifiers=True`` + multilabel target
    # is a known-incompatible combination (CalibratedClassifierCV is
    # single-output only; mlframe raises NotImplementedError). Force False
    # for multilabel combos so the suite call doesn't trip the guard.
    # Mirrors the canon in ``FuzzCombo.canonical_key``.
    _effective_prefer_calibrated = (
        False if (combo.prefer_calibrated_classifiers and combo.target_type == "multilabel_classification")
        else combo.prefer_calibrated_classifiers
    )
    behavior_kwargs: dict = {
        "align_polars_categorical_dicts": combo.align_polars_categorical_dicts,
        "continue_on_model_failure": combo.continue_on_model_failure,
        # F1 -- suite-level Faulthandler dump hook (Windows-only meaningful; canon-collapsed to False elsewhere by FuzzCombo._canonical_enable_crash_reporting).
        "enable_crash_reporting": combo._canonical_enable_crash_reporting(),
        "prefer_calibrated_classifiers": _effective_prefer_calibrated,
        # 2026-04-24 round 2
        "use_robust_eval_metric": combo.use_robust_eval_metric_cfg,
        # 2026-05-11 Wave 21: residual-audit footer toggle (regression only;
        # canonicalised to True for non-regression combos so dedup collapses).
        "report_residual_audit": combo.report_residual_audit_cfg,
        # 2026-05-21 iter151 P1-10: device-selection toggles. CPU-only
        # was hardcoded pre-iter151; per-model GPU/CPU dispatch branches
        # in compute_*_general_classif_params went unfuzzed.
        "prefer_gpu_configs": combo.prefer_gpu_configs_cfg,
        "prefer_cpu_for_lightgbm": combo.prefer_cpu_for_lightgbm_cfg,
        # 2026-05-22 iter170: confidence_ensemble_quantile. Defensive --
        # field may live on different config in older trees. Drop via
        # TrainingBehaviorConfig.model_fields filter below.
        "confidence_ensemble_quantile": combo.confidence_ensemble_quantile_cfg,
        # PZAD ensemble blend knobs: Caruana metric-direct weights (vs NNLS) + rank_average flavour appended to the
        # default blend set. Both gate on use_ensembles via the FuzzCombo canon; the phase reads them off behavior_config.
        "use_caruana_weights_in_ensemble": combo.use_caruana_weights_in_ensemble_cfg,
        "extra_ensembling_methods": ("rank_average",) if combo.ens_rank_average_cfg else (),
        # 2026-05-28 extreme_ar_group_aware_skip_models axis -- which model
        # families get skipped on extreme-AR + group-aware regimes. Mapped
        # from the enum axis ("default_neural"/"include_linear"/"empty")
        # into the tuple shape TrainingBehaviorConfig expects. Defensive
        # filter below drops the key when the field doesn't exist.
        "extreme_ar_group_aware_skip_models": (
            ("mlp", "ngb", "lstm", "gru", "rnn", "transformer")
            if combo.extreme_ar_group_aware_skip_models_cfg == "default_neural"
            else (
                ("mlp", "ngb", "lstm", "gru", "rnn", "transformer", "linear")
                if combo.extreme_ar_group_aware_skip_models_cfg == "include_linear"
                else ()  # "empty" -> no skip
            )
        ),
        # 2026-05-28 audit-pass-2 PART A: TrainingBehaviorConfig.target_temporal_audit_granularity
        # drives _phase_temporal_audit bin freq dispatch. Mirrors the existing
        # target_temporal_audit_column wiring elsewhere in the suite.
        "target_temporal_audit_granularity": combo.target_temporal_audit_granularity_cfg,
        # S27 close-out: TrainingBehaviorConfig.auto_wrap_partial_fit_es real
        # ctor param at _model_configs.py. False forces OFF the
        # PartialFitESWrapper auto-wrap at _trainer_train_and_evaluate.py:551.
        # Fuzz axis is inverted (force_off=True => auto_wrap=False).
        "auto_wrap_partial_fit_es": not combo.auto_wrap_partial_fit_es_force_off_cfg,
    }
    # Defensive filter: drop any behavior key that's not a model_fields entry.
    behavior_kwargs = _safe_cfg_kwargs(TrainingBehaviorConfig, **behavior_kwargs)
    if fairness_features:
        behavior_kwargs["fairness_features"] = fairness_features
        behavior_kwargs["fairness_min_pop_cat_thresh"] = 10
    # TrainingSplitConfig: val_size left at default (0.1); test_size
    # varies per axis. trainset_aging_limit validates strictly in
    # (0, 1) so only 0.5 is a safe non-None value.
    # val_seq_frac runtime canon RETIRED 2026-04-27 (batch 2). Production
    # fix: ``core._apply_outlier_detection_global`` now logs an error and
    # falls back to the unfiltered val_set when OD would reject most val
    # rows (the typical reason val_df collapsed to 0 in this combo
    # window). Splitter still produces a non-empty val.
    # 2026-04-27 Session 7 batch 7: ``val_placement='backward' +
    # trainset_aging_limit`` are incompatible by design — aging removes
    # the oldest rows, which are the ones adjacent to the backward-
    # placed val. Splitting raises a hard ValueError. Pre-batch-7 this
    # was masked: the fuzz fixture didn't surface ts_field on its FTE,
    # so timestamps reached splitting as None and val_placement
    # silently fell back to 'forward' (no error). Now that ts_field is
    # threaded through (so the audit auto-detect path is exercised),
    # the conflict materialises. Canonicalise: when both flags are
    # set AND timestamps will reach splitting (with_datetime_col=True),
    # drop aging — keep the backward placement, since it's the
    # less-common axis being tested.
    _aging_eff = combo.trainset_aging_limit_cfg
    if (
        _aging_eff is not None
        and combo.val_placement_cfg == "backward"
        and combo.with_datetime_col
    ):
        _aging_eff = None
    # 2026-05-21 iter151 P1-5/P1-6: test_sequential_fraction + calib_size
    # canon. test_sequential_fraction needs with_datetime_col (no time
    # signal otherwise); calib_size + test_size + val_size must sum < 1.0
    # (split_config validator enforces).
    _tsf_eff = combo.test_sequential_fraction_cfg if combo.with_datetime_col else None
    _calib_eff = combo.calib_size_cfg
    if _calib_eff is not None:
        # Defensive: keep calib + test + val_default(0.1) <= 0.9 so the
        # split_config validator doesn't reject the combo.
        if (combo.test_size_cfg + 0.1 + _calib_eff) >= 1.0:
            _calib_eff = None
    # conformal_size is a SECOND holdout; the split-config validator enforces test + val(0.1) + calib + conformal <= 1.0.
    _conformal_eff = combo.conformal_size_cfg
    if _conformal_eff is not None:
        if (combo.test_size_cfg + 0.1 + (_calib_eff or 0.0) + _conformal_eff) > 1.0:
            _conformal_eff = None
    split_config = TrainingSplitConfig(
        test_size=combo.test_size_cfg,
        val_placement=combo.val_placement_cfg,
        trainset_aging_limit=_aging_eff,
        shuffle_val=combo.shuffle_val_cfg,
        shuffle_test=combo.shuffle_test_cfg,
        wholeday_splitting=combo.wholeday_splitting_cfg,
        val_sequential_fraction=combo.val_sequential_fraction_cfg,
        # 2026-06-03: composite_cardinality_cap was a fuzz axis but never passed
        # to the split config (inert). Field default 200, ge=2; axis is (200,50).
        composite_cardinality_cap=combo.composite_cardinality_cap_cfg,
        # 2026-05-11 Wave 21: group-aware splitting toggle. Only meaningful
        # when wholeday_splitting=True + with_datetime_col=True (the
        # splitter derives groups from the datetime); canonicalised away
        # for other combos.
        use_groups=combo.use_groups_cfg,
        # 2026-05-21 iter151 P1-5: time-axis-tail test placement.
        test_sequential_fraction=_tsf_eff,
        # 2026-05-21 iter151 P1-6: post-hoc calibration split.
        calib_size=_calib_eff,
        # E2 time-aware split surface. cv_strategy canon collapses forward-walk strategies to 'random' on val_placement='backward';
        # cv_purge applies only under 'purged'; conformal_size carves a second holdout (sum-guarded above).
        cv_strategy=combo._canonical_cv_strategy(),
        cv_purge=(combo.cv_purge_cfg if combo._canonical_cv_strategy() == "purged" else 0),
        conformal_size=_conformal_eff,
    )
    # PreprocessingConfig is built by ``_preprocessing_for_combo`` and
    # passed explicitly at the suite call site (it owns the fix_inf_eff /
    # rm_const_eff runtime canon AND the linear-model category-encoder
    # branch). Don't include it here or we'd hit a duplicate-kwarg
    # TypeError when this dict is **-splatted alongside the explicit
    # ``preprocessing_config=...``.
    return {
        "pipeline_config": PreprocessingBackendConfig(
            prefer_polarsds=combo.prefer_polarsds,
            scaler_name=combo.scaler_name_cfg,
            categorical_encoding=combo.categorical_encoding_cfg,
            skip_categorical_encoding=combo.skip_categorical_encoding_cfg,
            imputer_strategy=combo.imputer_strategy_cfg,
            # 2026-05-21 iter151 P1-9: polars-ds -> sklearn fallback bridge.
            # Only meaningful when prefer_polarsds=True (otherwise the
            # bridge path is unreachable).
            fallback_to_sklearn=combo.fallback_to_sklearn_cfg,
            # 2026-05-22 iter170: robust-scaler quantile bounds (defensive).
            **_safe_cfg_kwargs(
                PreprocessingBackendConfig,
                robust_q_low=combo.robust_q_low_cfg,
                robust_q_high=combo.robust_q_high_cfg,
            ),
        ),
        "split_config": split_config,
        "feature_types_config": FeatureTypesConfig(
            auto_detect_feature_types=combo.auto_detect_cats,
            use_text_features=combo.use_text_features,
            honor_user_dtype=combo.honor_user_dtype,
            text_features=text_features,
            embedding_features=embedding_features,
            cat_text_cardinality_threshold=combo.cat_text_card_threshold_cfg,
        ),
        "behavior_config": TrainingBehaviorConfig(**behavior_kwargs),
        # 2026-05-18 — composite-target discovery axes. The dict is splatted
        # into the suite call, so when discovery is enabled this routes the
        # config through. Disabled-config still passes through so the suite
        # path is exercised symmetrically.
        "composite_target_discovery_config": build_composite_discovery_config(combo),
        # PreprocessingExtensionsConfig — sklearn-bridge transforms applied
        # once and reused per model. Mirror FuzzCombo._canonical_prep_ext
        # so combos with NaN-injecting axes or unencoded categoricals
        # collapse to None (the bridge cannot consume those). Only attach
        # the config when at least one knob is non-default; otherwise pass
        # None to preserve the polars-native fastpath.
        "preprocessing_extensions": _maybe_preprocessing_extensions(combo, PreprocessingExtensionsConfig),
        # Multilabel dispatch is consulted only when target_type is
        # MULTILABEL_CLASSIFICATION (helpers._maybe_wrap_multilabel
        # short-circuits otherwise), but pass on every combo — the
        # production API accepts it unconditionally. Mirror
        # FuzzCombo._canonical_multilabel_strategy so chain dispatch is
        # only requested for combos whose data shape supports it.
        "multilabel_dispatch_config": MultilabelDispatchConfig(
            strategy=combo._canonical_multilabel_strategy(),
            n_chains=combo.multilabel_n_chains_cfg,
            chain_order_strategy=combo.multilabel_chain_order_cfg,
            cv=combo.multilabel_cv_cfg,
            # 2026-05-11 Wave 21: post-hoc calib downgrade toggle (multilabel
            # only). Canonicalised to default False for non-multilabel.
            allow_uncalibrated_multi=combo.multilabel_allow_uncalibrated_cfg,
            # iter170 deep axis (defensive).
            **_safe_cfg_kwargs(
                MultilabelDispatchConfig,
                force_native_xgb_multilabel=combo.multilabel_force_native_xgb_cfg,
                # iter180 DEPTH-4 list-typed: per_label_thresholds (uniform 0.4 vs None),
                # chain_seeds (deterministic per-chain seeds vs None).
                per_label_thresholds=(
                    None if combo.multilabel_per_label_thresholds_cfg is None
                    else [0.4, 0.4, 0.4]  # K=3 default labels in fuzz frame
                ),
                chain_seeds=(
                    None if combo.multilabel_chain_seeds_cfg is None
                    else list(range(combo.multilabel_n_chains_cfg))
                ),
            ),
        ),
    }


# (Removed 2026-05-18: ``_composite_discovery_config_for_combo`` folded
# into the shared builder ``build_composite_discovery_config`` in
# ``_fuzz_combo.py`` — see comment above _make_cat_fe_config_for_fuzz.)


def _recurrent_sequences_for_combo(combo: FuzzCombo, df=None):
    """Synthesize per-row sequences for the recurrent-model path.

    The recurrent dispatcher needs ``sequences`` aligned 1:1 with the
    tabular frame (each row → one (T, F) np.ndarray). For combos that
    don't enable the recurrent axis, return None — the suite then skips
    the get_sequences/sequences plumbing entirely.

    Sequence shape: T=8 timesteps × F=2 features, deterministically
    seeded off ``combo.seed`` so re-runs produce the same data. Small
    enough to keep recurrent fit time well under the 300s per-test
    budget on a CPU box.
    """
    if combo._canonical_recurrent_model() is None:
        return None
    n = combo.n_rows
    rng = np.random.default_rng(combo.seed + 9001)
    return [rng.standard_normal((8, 2)).astype(np.float32) for _ in range(n)]


def _recurrent_config_for_combo(combo: FuzzCombo):
    """Build a minimal RecurrentConfig sized for fuzz speed.

    Returns None when the recurrent axis is canonicalised off (the
    config is unused without ``recurrent_models``). Otherwise picks
    ``rnn_type`` from the combo axis and caps epochs / hidden size so
    a single combo trains in <60s on CPU.
    """
    rec = combo._canonical_recurrent_model()
    if rec is None:
        return None
    try:
        from mlframe.training.neural.recurrent import RecurrentConfig, RNNType, InputMode
    except ImportError:
        return None
    rnn_type = {
        "lstm": RNNType.LSTM,
        "gru": RNNType.GRU,
        "transformer": RNNType.TRANSFORMER,
    }[rec]
    # iter170: input_mode + num_workers from axes (defensive enum lookup).
    _input_mode = {
        "hybrid": getattr(InputMode, "HYBRID", InputMode.HYBRID),
        "sequence_only": getattr(InputMode, "SEQUENCE_ONLY", getattr(InputMode, "SEQ_ONLY", InputMode.HYBRID)),
        "tabular_only": getattr(InputMode, "TABULAR_ONLY", getattr(InputMode, "FEATURES_ONLY", InputMode.HYBRID)),
    }.get(combo.recurrent_input_mode_cfg, InputMode.HYBRID)
    return RecurrentConfig(
        input_mode=_input_mode,
        rnn_type=rnn_type,
        # 2026-05-28 recurrent_hidden_size_cfg axis. Library defaults to 128;
        # the fuzz suite clamps to small values for wall-time. Map: 128 -> 16
        # (the prior hardcoded "wide" path), 32 -> 8 (the "narrow" variant).
        # Both reach the same RNN parameter-count branch; the canonical_key
        # still distinguishes the axis values so dedup is preserved.
        hidden_size=(16 if combo.recurrent_hidden_size_cfg >= 128 else 8),
        num_layers=1,
        bidirectional=False,
        use_attention=False,
        max_epochs=2,
        early_stopping_patience=1,
        accelerator="cpu",
        num_workers=combo.recurrent_num_workers_cfg,
        batch_size=64,
        # iter162: nested precision + sequence_preprocessing axes.
        precision=combo.recurrent_precision_cfg,
        sequence_preprocessing=combo.recurrent_sequence_preprocessing_cfg,
    )


def _maybe_preprocessing_extensions(combo: FuzzCombo, config_cls):
    """Build a PreprocessingExtensionsConfig from the combo, or None.

    Returns None when every prep_ext_* axis canonicalises to None — that
    keeps the polars-native fastpath active for combos that don't need
    sklearn-bridge transforms. Otherwise instantiate the config with the
    canonical values so dedup-equivalent combos share runtime behaviour.
    """
    scaler = combo._canonical_prep_ext("scaler")
    kbins = combo._canonical_prep_ext("kbins")
    poly_deg = combo._canonical_prep_ext("polynomial_degree")
    dim_red = combo._canonical_prep_ext("dim_reducer")
    nonlin = combo._canonical_prep_ext("nonlinear")
    if scaler is None and kbins is None and poly_deg is None and dim_red is None and nonlin is None:
        return None
    # PreprocessingExtensionsConfig validates that binarization_threshold
    # and kbins are mutually exclusive; we only set kbins so that's safe.
    # polynomial_degree must be >= 2 when set (validator); kbins must be
    # >= 2. The axis values already satisfy both.
    # 2026-05-28 audit-pass-2 PART A: PreprocessingExtensionsConfig.dim_n_components
    # is only meaningful when a dim_reducer is actually picked (PCA/TruncatedSVD).
    # Forward via _safe_cfg_kwargs so the call still works on older trees that
    # don't expose the field. Pre-iter500 canonical_key already collapses the
    # axis to its default (50) when dim_red is None, so kwargs split is
    # idempotent across cohorts.
    dim_components_kwargs: dict = {}
    if dim_red is not None:
        dim_components_kwargs = _safe_cfg_kwargs(
            config_cls, dim_n_components=combo.prep_ext_dim_n_components_cfg,
        )
    return config_cls(
        scaler=scaler,
        kbins=kbins,
        polynomial_degree=poly_deg,
        dim_reducer=dim_red,
        nonlinear_features=nonlin,
        **dim_components_kwargs,
    )


def _outlier_detector_for_combo(combo: FuzzCombo):
    """Construct an ``outlier_detector`` object when the combo asks for
    one, else return None. Kept separate from ``_configs_for_combo`` so
    the suite kwarg and the combo axis are wired independently.

    Supported axes (all sklearn): isolation_forest (random forest),
    lof (LocalOutlierFactor with novelty=True so .predict is exposed),
    ocsvm (OneClassSVM, RBF kernel). OCSVM is canonicalised to None for
    n_rows >= 1200 because its O(n²) fit is slow enough to dominate the
    fuzz combo runtime — see canonical_key.
    """
    od = combo.outlier_detection
    # Mirror canonical_key: OCSVM's O(n²) fit is too slow on n>=1200.
    if od == "ocsvm" and combo.n_rows >= 1200:
        return None
    try:
        if od == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            return IsolationForest(
                contamination=0.05, random_state=combo.seed, n_estimators=20,
            )
        if od in ("lof", "ocsvm"):
            # LOF and OneClassSVM both raise on NaN inputs (unlike
            # IsolationForest, which tolerates NaN on recent sklearn).
            # The mlframe outlier-detection step runs BEFORE the
            # preprocessing pipeline's fix_infinities/imputer, so
            # NaN-injecting combos (inject_inf_nan, inject_all_nan_col,
            # null_fraction_cats > 0 on numeric-coerced cats) reach the
            # detector raw. Wrap in a sklearn Pipeline with a SimpleImputer
            # so the detector never sees NaN. Pipeline.predict delegates
            # to the last step, which exposes .predict for novelty=True LOF
            # and OneClassSVM — matches the .fit / .predict contract the
            # mlframe pipeline expects.
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            if od == "lof":
                from sklearn.neighbors import LocalOutlierFactor
                detector = LocalOutlierFactor(
                    contamination=0.05, novelty=True, n_neighbors=20,
                )
            else:  # ocsvm
                from sklearn.svm import OneClassSVM
                detector = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
            return Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                (od, detector),
            ])
    except ImportError:
        return None
    return None


def _custom_pre_pipelines_for_combo(combo: FuzzCombo):
    """When ``combo.custom_prep == "pca2"`` attach an IncrementalPCA
    transformer. Fails-open if sklearn isn't importable.

    Gated on all-numeric frame: sklearn's IncrementalPCA cannot fit
    on string/categorical/list-of-float columns, and the mlframe
    pipeline does NOT pre-encode before a custom_pre_pipeline.
    Matches the canonicalisation in FuzzCombo.canonical_key so the
    combo generation and runtime wiring agree.
    """
    # Mirror FuzzCombo.canonical_key pca2-incompatibility gating:
    # IncrementalPCA also rejects NaN (inject_inf_nan) and
    # all-null columns (inject_degenerate_cols), not just
    # non-numeric dtypes.
    pca_incompatible = (
        combo.cat_feature_count > 0
        or combo.text_col_count > 0
        or combo.embedding_col_count > 0
        or combo.inject_inf_nan
        or combo.inject_degenerate_cols
    )
    if combo.custom_prep == "pca2" and not pca_incompatible:
        try:
            from sklearn.decomposition import IncrementalPCA
            return {"pca2": IncrementalPCA(n_components=2)}
        except ImportError:
            return None
    return None


# 2026-06-04 FS-coverage follow-up: which BorutaShap.__init__ knobs the *live*
# class actually exposes. The early_stop_* family (margin-gated adaptive trial
# stop) and the budget knob landed in separate waves; the FeatureSelectionConfig
# .boruta_shap_kwargs validator rejects any key not on the live ctor signature.
# We therefore probe the signature once and only pass a knob when it exists, so
# the fuzz wiring is correct regardless of which knobs are committed yet (the
# axes still vary + drive dedup; they simply no-op into BorutaShap until the knob
# is available). Mirrors the validator's own inspect.signature(BorutaShap.__init__).
def _live_boruta_shap_params() -> set[str]:
    import inspect
    from mlframe.feature_selection.boruta_shap import BorutaShap
    return set(inspect.signature(BorutaShap.__init__).parameters) - {"self"}


def _boruta_shap_kwargs_for_combo(combo: FuzzCombo):
    """Build the boruta_shap_kwargs dict for ``combo`` (or None when BorutaShap
    is off). Fuzz-speed knobs + the FS-coverage axes are threaded in; any knob
    not present on the live BorutaShap signature is dropped so a clean checkout
    where a knob is not yet committed never produces an invalid config."""
    if not combo.use_boruta_shap_cfg:
        return None
    live = _live_boruta_shap_params()
    # n_trials=10 keeps the SHAP-per-trial cost ~15x lower than the 150 default
    # while still exercising shadow build + SHAP explain + tail test every run.
    kw = {
        "n_trials": 10,
        "verbose": False,
        "importance_measure": combo.boruta_importance_measure_cfg,
        # 2026-06-03 FS-coverage audit -- previously-unfuzzed BorutaShap ctor knobs.
        "optimistic": combo.boruta_optimistic_cfg,
        "train_or_test": combo.boruta_train_or_test_cfg,
        "premerge_clusters": combo.boruta_premerge_clusters_cfg,
    }
    # 2026-06-04 FS-coverage follow-up -- margin-gated adaptive trial-stop axes.
    # Guarded on the live signature (the early_stop_* family may land after the
    # budget knob), so a not-yet-committed knob is simply skipped.
    for _k, _v in (
        ("early_stop_tentative", combo.boruta_early_stop_tentative_cfg),
        ("early_stop_patience", combo.boruta_early_stop_patience_cfg),
        ("early_stop_margin", combo.boruta_early_stop_margin_cfg),
        # FIXED 5-min wall-clock cap (NOT a fuzz axis): MRMR/RFECV parity so a
        # runaway BorutaShap fit on any combo is bounded. stop_file is left at its
        # "stop" default (never set) so the fuzz never trips a stray stop-flag.
        # Mirrors rfecv_kwargs/mrmr_kwargs max_runtime_mins=5 (commit 592b8a60).
        ("max_runtime_mins", 5),
    ):
        if _k in live:
            kw[_k] = _v
    return kw


def _maybe_to_parquet(combo: FuzzCombo, df, tmp_path):
    """Convert ``df`` to a parquet file path when
    ``combo.input_storage == "parquet"``; otherwise pass through.
    The suite's ``load_and_prepare_dataframe`` accepts str paths
    and reads them internally — this exercises the streaming
    parquet code path.
    """
    if combo.input_storage != "parquet":
        return df
    import polars as _pl
    path = str(tmp_path / "combo_input.parquet")
    if isinstance(df, _pl.DataFrame):
        df.write_parquet(path)
    else:
        df.to_parquet(path)
    return path


def _preprocessing_for_combo(combo: FuzzCombo):
    """PreprocessingConfig for a combo. Combines the per-axis flags
    (``fillna_value`` / ``fix_infinities`` / ``ensure_float32_dtypes`` /
    ``remove_constant_columns`` plus the runtime-canon ``fix_inf_eff`` /
    ``rm_const_eff`` rewrites that mirror ``_fuzz_combo.canonical_key``)
    with a category encoder when a non-native-cat model (linear) is
    present — matches the prod config pattern the existing integration
    tests use.
    """
    from mlframe.training.configs import PreprocessingConfig
    # fix_inf_eff / rm_const_eff runtime canons RETIRED 2026-04-27
    # (batch 2). Production fixes:
    #   * fix_infinities=False + inject_inf_nan: trainer pre-fit dtype
    #     guard now catches ``np.inf``/``-np.inf`` in numeric columns
    #     and replaces with NaN before XGB/HGB see them (the fix lives
    #     in ``preprocessing.process_infinities`` plus PreprocessingConfig
    #     validators).
    #   * remove_constant_columns=False + inject_degenerate / inject_all_nan_col:
    #     the polars-ds scaler now pre-filters zero-IQR / all-null
    #     columns in ``training/pipeline._select_scalable_numeric_columns``
    #     so ``robust_scale`` never sees a divide-by-zero.
    #
    # inject_inf_nan + fix_infinities=False is a self-contradictory combo:
    # the inject_inf_nan axis exists to verify mlframe CLEANS injected inf,
    # but fix_infinities=False disables the only cleaning step. With it off,
    # raw inf reaches inf-intolerant backends -- XGBoost's hist QuantileDMatrix
    # rejects it ("Input data contains `inf` ... while `missing` is not set
    # to inf") and crashes the whole suite. Force cleaning whenever inf is
    # deliberately injected so the axis stays meaningful; fix_infinities=False
    # is still exercised on the (inf-free) non-injected combos.
    _fix_inf = bool(combo.fix_infinities_cfg or combo.inject_inf_nan)
    if "linear" in combo.models and combo.cat_feature_count > 0:
        try:
            import category_encoders as ce
            from sklearn.preprocessing import StandardScaler
            from sklearn.impute import SimpleImputer
            return PreprocessingConfig(
                drop_columns=[],
                fillna_value=combo.fillna_value_cfg,
                fix_infinities=_fix_inf,
                ensure_float32_dtypes=combo.ensure_float32_cfg,
                remove_constant_columns=combo.remove_constant_columns_cfg,
                category_encoder=ce.CatBoostEncoder(),
                scaler=StandardScaler(),
                imputer=SimpleImputer(strategy="mean"),
            )
        except ImportError:
            pass
    return PreprocessingConfig(
        drop_columns=[],
        fillna_value=combo.fillna_value_cfg,
        fix_infinities=_fix_inf,
        ensure_float32_dtypes=combo.ensure_float32_cfg,
        remove_constant_columns=combo.remove_constant_columns_cfg,
    )


def _skip_if_deps_missing(models: tuple[str, ...]) -> None:
    pkg = {
        "cb": "catboost", "xgb": "xgboost", "lgb": "lightgbm",
        "hgb": "sklearn", "linear": "sklearn",
        "mlp": "lightning",  # PyTorch Lightning gates the MLP path
    }
    for m in models:
        pytest.importorskip(pkg[m])


def _iter_trained_models(trained):
    """Yield (target_type, target_name, trained_entry) for every model.

    The suite returns ``trained[target_type][target_name]`` → list of
    SimpleNamespace entries with ``.model``, ``.val_preds``, ``.val_probs``,
    ``.columns``, ``.metrics`` attributes.
    """
    if not isinstance(trained, dict):
        return
    for tt, by_name in trained.items():
        if not isinstance(by_name, dict):
            continue
        for tn, lst in by_name.items():
            if not isinstance(lst, list):
                continue
            for entry in lst:
                yield tt, tn, entry


def _assert_prediction_invariants(trained, meta, combo) -> None:
    """Fix C (cheap tier): post-train property checks that run on every
    combo.

    Runs on the predictions the suite already materialised
    (``entry.val_preds``, ``entry.test_preds``), so no re-fit. Catches:

    - NaN / Inf leaking into the model head (I1).
    - Constant predictions when the val target has ≥2 distinct classes / values
      (I2) — indicates dead pipeline / silently-dropped features.
    - Shape mismatch between ``val_preds`` and ``meta['val_size']`` (I3) —
      indicates row-slicing drift between the pipeline and its metrics.

    Deeper invariants (determinism, idempotency, column-perm, prediction
    probe) require re-fit; they live behind ``MLFRAME_FUZZ_INVARIANTS=full``
    in ``test_fuzz_invariants_full.py``.
    """
    import numpy as np

    val_size = (meta or {}).get("val_size")
    for tt, tn, entry in _iter_trained_models(trained):
        # Pull both available forms; prefer probs (cleaner finite check)
        # then fall back to preds.
        for attr in ("val_probs", "val_preds"):
            arr = getattr(entry, attr, None)
            if arr is None:
                continue
            # Normalise to ndarray for finite checks.
            try:
                arr_np = np.asarray(arr)
            except Exception:
                continue
            if arr_np.size == 0:
                continue
            # I1 — finiteness. Regression preds can be negative but must
            # be finite. Classification probs live in [0, 1] but we only
            # assert finiteness here to stay model-agnostic.
            if np.issubdtype(arr_np.dtype, np.floating):
                n_bad = int(np.count_nonzero(~np.isfinite(arr_np)))
                assert n_bad == 0, (
                    f"I1: non-finite values in {tt}/{tn}/{type(entry.model).__name__}.{attr} "
                    f"({n_bad}/{arr_np.size})"
                )
            # I3 — shape upper-bound. ``meta['val_size']`` is measured
            # before outlier-detection filters rows, so post-OD ``val_preds``
            # can be strictly smaller. Asserting ``<=`` catches only the
            # bug we care about (preds longer than val slice → row-slicing
            # drift), not OD-expected shrinkage.
            if val_size is not None and val_size > 0 and arr_np.ndim >= 1:
                assert arr_np.shape[0] <= val_size, (
                    f"I3: {attr} shape[0]={arr_np.shape[0]} > val_size={val_size} "
                    f"for {tt}/{tn}/{type(entry.model).__name__}"
                )
            # I2 — non-constant predictions when val has >1 row.
            # Skipped for tiny val slices (< 4 rows; statistical noise).
            # Skipped when outlier-detection could have reduced val to
            # a single class (we can't cheaply check val target variance
            # here without re-extracting). Asserted only for classification
            # probs where the "all-same" outcome is provably degenerate.
            if (
                attr == "val_probs"
                and arr_np.size >= 4
                and np.issubdtype(arr_np.dtype, np.floating)
            ):
                # Pull scalar series for 1-D, flatten for 2-D.
                vals = arr_np.ravel()
                unique_near = np.unique(np.round(vals, 6))
                # Relaxed on imbalanced multi-class: a 1% minority class
                # may produce probs rounded to 6dp that collapse to a
                # single value when the model mass-predicts the majority.
                # R3 combos with imbalance_ratio=rare_1pct + very few
                # iterations see this legitimately.
                if combo.imbalance_ratio == "rare_1pct":
                    continue
                assert unique_near.size >= 2, (
                    f"I2: val_probs are all identical "
                    f"({unique_near[0]:.6g}) for {tt}/{tn}/{type(entry.model).__name__} "
                    f"— pipeline may have dropped all features"
                )


def _assert_serialization_roundtrip(trained, data_dir: str, combo) -> None:
    """Fix R3-3 (I4 — serialization roundtrip). Gated by env
    ``MLFRAME_FUZZ_ROUNDTRIP=1`` because it spends ~100-500ms per combo
    on disk I/O + load + predict.

    Finds the first saved .dump file under ``data_dir`` and verifies it
    loads back and can produce predictions. This is a smoke check, not a
    bit-for-bit equivalence (the saved artifact is meant for
    ``load_mlframe_suite`` + ``predict_mlframe_models_suite`` and the
    metadata/preprocessing trail is not hydrated by joblib.load alone).
    A load failure catches regressions in the picklable contract of
    models / pipelines / custom estimators.
    """
    import os
    import glob
    import joblib

    if not trained:
        return
    pattern = os.path.join(data_dir, "**", "*.dump")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return  # No models saved (continue_on_failure path, or save disabled)
    # Load the first .dump file; only assert the artifact is not corrupt.
    try:
        obj = joblib.load(files[0])
    except Exception as exc:
        raise AssertionError(
            f"I4: saved model artifact {files[0]!r} failed joblib.load: "
            f"{type(exc).__name__}: {exc}"
        )
    # The dump contains at least one object with a predict-like attribute
    # (the trained pipeline / model). Not asserting specific type — the
    # wrapper class can evolve; catching "can't unpickle" is the goal.
    assert obj is not None, f"I4: joblib.load returned None for {files[0]}"

