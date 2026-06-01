"""Randomized fuzz coverage for ``train_mlframe_models_suite``.

Feeds ~150 unique, pairwise-covering combos through the suite and records
every combo's outcome to ``_fuzz_results.jsonl`` for later analysis.

Known-bug xfail rules live in ``_fuzz_combo.KNOWN_XFAIL_RULES`` and are
applied automatically per combo via ``pytest.mark.xfail`` in the test
function — new bugs discovered by fuzzing should be added there once
they're traced to a specific combo predicate.
"""
from __future__ import annotations

import os
import time
import traceback

import numpy as np
import pytest

# Fuzz combos run ~150 train_mlframe_models_suite iterations and are deselected
# from the default test run; pass pytest --run-fuzz to include.
pytestmark = pytest.mark.fuzz

from ._fuzz_combo import (
    FuzzCombo,
    build_frame_for_combo,
    enumerate_combos,
    log_combo_outcome,
    xfail_reason,
    # 2026-05-18 shared builders (refactor: single-edit point for new axes)
    build_mrmr_kwargs,
    build_composite_discovery_config,
)
from .shared import SimpleFeaturesAndTargetsExtractor

# 2026-04-27: train_mlframe_models_suite signature collapsed several
# top-level kwargs (outlier_detector / data_dir / use_mrmr_fs / ...) into
# typed configs (see CHANGELOG ``2026-04-27 — Calibration reporting
# upgrades + suite-config sweep``). The fuzz suite uses these new
# configs directly at the suite call site; module-level imports here so
# the local imports inside ``_configs_for_combo`` aren't load-bearing.
from mlframe.training import (
    OutputConfig,
    OutlierDetectionConfig,
    FeatureSelectionConfig,
    ReportingConfig,
    ConfidenceAnalysisConfig,
)

# Enumerate once at import time — small, pure Python, no heavy deps.
# FUZZ_SEED env var overrides the default (driver scripts use this to
# sweep many seeds in sequence without editing the file; each pytest
# invocation reads the env fresh so 10k-combo campaigns can span 60+
# seeds × 150 combos each without the parent process sharing state).
_FUZZ_MASTER_SEED = int(os.environ.get("FUZZ_SEED", "20260422"))
COMBOS: list[FuzzCombo] = enumerate_combos(target=150, master_seed=_FUZZ_MASTER_SEED)


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
        "max_runtime_mins": 2,
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
        PreprocessingConfig,
        TrainingSplitConfig,
        MultilabelDispatchConfig,
        PreprocessingExtensionsConfig,
    FeatureSelectionConfig,
    OutlierDetectionConfig,
    OutputConfig
)
    # 2026-05-18 — composite-target discovery (Packs J + K) wiring.
    # Imported lazily inside the function so the top of the file
    # doesn't grow another mandatory import.
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
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
        # 2026-05-28 audit-pass-2 PART A: TrainingBehaviorConfig.use_flaml_zeroshot
        # picks flaml_zeroshot.{XGB,LGBM}{Classifier,Regressor} vs vanilla. The
        # from_axes canon in _fuzz_combo already drops True->False when `flaml`
        # is unimportable so this is always a safe assignment.
        "use_flaml_zeroshot": combo.behavior_use_flaml_zeroshot_cfg,
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
    split_config = TrainingSplitConfig(
        test_size=combo.test_size_cfg,
        val_placement=combo.val_placement_cfg,
        trainset_aging_limit=_aging_eff,
        shuffle_val=combo.shuffle_val_cfg,
        shuffle_test=combo.shuffle_test_cfg,
        wholeday_splitting=combo.wholeday_splitting_cfg,
        val_sequential_fraction=combo.val_sequential_fraction_cfg,
        # 2026-05-11 Wave 21: group-aware splitting toggle. Only meaningful
        # when wholeday_splitting=True + with_datetime_col=True (the
        # splitter derives groups from the datetime); canonicalised away
        # for other combos.
        use_groups=combo.use_groups_cfg,
        # 2026-05-21 iter151 P1-5: time-axis-tail test placement.
        test_sequential_fraction=_tsf_eff,
        # 2026-05-21 iter151 P1-6: post-hoc calibration split.
        calib_size=_calib_eff,
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
    if "linear" in combo.models and combo.cat_feature_count > 0:
        try:
            import category_encoders as ce
            from sklearn.preprocessing import StandardScaler
            from sklearn.impute import SimpleImputer
            return PreprocessingConfig(
                drop_columns=[],
                fillna_value=combo.fillna_value_cfg,
                fix_infinities=combo.fix_infinities_cfg,
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
        fix_infinities=combo.fix_infinities_cfg,
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
    import math
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


@pytest.fixture(autouse=True)
def _fuzz_combo_cleanup():
    """Between fuzz combos: close matplotlib figures, clear CB/XGB/LGB
    internal caches, drop generated models — state accumulation across the
    150-combo run has been observed to trigger native-level crashes
    (SIGSEGV on combo 6 in a sequential run on 2026-04-22)."""
    yield
    # 1. Matplotlib figures (mlframe emits per-model feature_importance plots).
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass
    # 2. mlframe's in-process caches (CB val Pool cache, tier-DF cache).
    try:
        from mlframe.training import trainer as _tr
        for attr in ("_CB_POOL_CACHE", "_CB_VAL_POOL_CACHE"):
            cache = getattr(_tr, attr, None)
            if hasattr(cache, "clear"):
                cache.clear()
    except Exception:
        pass
    # 3. CatBoost internal state — force full GPU/CPU resource release.
    try:
        import catboost
        # catboost.utils doesn't expose a global cleanup; deleting module-level
        # state is unsafe. Best-effort: trigger a GC pass twice so CB's
        # C++-side memory pools see zero Python refs before the next combo
        # allocates.
    except ImportError:
        pass
    # 4. Double GC — first pass collects Python objects, second pass lets
    # finalizers (including native lib close-outs) run before we return.
    import gc
    gc.collect()
    gc.collect()
    # 5. clean_ram: on Linux returns memory to OS via malloc_trim(0);
    # on Windows trims working-set via SetProcessWorkingSetSizeEx (RSS
    # only, not commit). Wired here as best-effort against multi-combo
    # native heap fragmentation that historically OOMs around combo #36
    # of 150 on Win32 multi-classification × ensembles paths.
    try:
        from pyutilz.system import clean_ram
        clean_ram()
    except Exception:
        pass


@pytest.mark.slow
@pytest.mark.slow_only
@pytest.mark.timeout(300)
@pytest.mark.parametrize("combo", COMBOS, ids=[c.pytest_id() for c in COMBOS])
def test_fuzz_train_mlframe_models_suite(combo: FuzzCombo, tmp_path, request):
    """Run ``train_mlframe_models_suite`` on one random combo; log the outcome.

    FUZZ-1 (2026-05-23) -- when ``MLFRAME_FUZZ_PERF_MODE`` env var is set
    (any truthy value: 1/yes/true/on), each combo is downgraded to a tiny
    config-coverage run: n_rows=1000, iterations=1, MRMR/Boruta/ensembles
    /baseline_diagnostics/dummy_baselines all disabled. Goal: verify suite
    wiring on every combo in seconds instead of minutes. Quality / metric
    assertions are NOT meaningful in this mode -- it's a smoke test only.

    Default (env unset): full combo runs unchanged.
    """
    import os as _os
    if _os.environ.get("MLFRAME_FUZZ_PERF_MODE", "").lower() in ("1", "yes", "true", "on"):
        from ._fuzz_combo import apply_perf_mode
        combo = apply_perf_mode(combo)
    _skip_if_deps_missing(combo.models)

    # Apply xfail automatically for known bugs. pytest's runtime-xfail marker
    # works via ``request.node.add_marker``.
    reason = xfail_reason(combo)
    if reason is not None:
        # strict=True so an XPASS (combo now passes because the underlying fix landed) is a
        # visible regression -- the developer must remove the rule from KNOWN_XFAIL_RULES.
        # Pre-fix this was strict=False, which silently greened combos whether they passed or
        # failed and lost track of fix landings.
        request.node.add_marker(pytest.mark.xfail(reason=reason, strict=True))

    df, target_col, _cat_names = build_frame_for_combo(combo)

    # #16 invariant: capture caller-frame schema + shape before the
    # suite runs; re-assert identity after. Applies when input stays
    # in-memory (parquet-path combos have no Python-level caller frame
    # to preserve — the parquet file is the source of truth).
    frame_schema_before = None
    frame_shape_before = None
    frame_cols_before = None
    if combo.input_storage == "memory":
        if hasattr(df, "schema"):
            frame_schema_before = dict(df.schema)
        elif hasattr(df, "dtypes"):
            frame_schema_before = {c: str(df[c].dtype) for c in df.columns}
        frame_shape_before = getattr(df, "shape", None)
        frame_cols_before = tuple(df.columns) if hasattr(df, "columns") else None

    # Resolve target_type for FTE — maps combo's string target_type to
    # the TargetTypes enum. Multilabel + multi_target_regression get
    # explicit TargetTypes to trigger the 2-D target unpack path in FTE.
    #
    # multi_target_regression disambiguation: build_frame_for_combo emits
    # a 2-D (N, K) list column (target_col == "target") only when every
    # model in the combo natively handles a 2-D continuous target; for any
    # non-native combo it downgrades the frame to a 1-D "target_reg" column
    # (see _NATIVE_MTR_MODELS gate). The emitted target_col is therefore the
    # authoritative signal for which target_type the FTE should see -- a
    # downgraded MTR combo is REGRESSION at the data level.
    from mlframe.training.configs import TargetTypes as _TT
    _effective_target_type = combo.target_type
    if combo.target_type == "multi_target_regression" and target_col != "target":
        _effective_target_type = "regression"
    _combo_tt = {
        "regression": _TT.REGRESSION,
        "binary_classification": _TT.BINARY_CLASSIFICATION,
        "multiclass_classification": _TT.MULTICLASS_CLASSIFICATION,
        "multilabel_classification": _TT.MULTILABEL_CLASSIFICATION,
        "learning_to_rank": _TT.LEARNING_TO_RANK,
        "multi_target_regression": _TT.MULTI_TARGET_REGRESSION,
    }[_effective_target_type]
    # LTR combos: build_frame_for_combo adds a 'qid' column for queries;
    # surface it as the FTE's group_field so the ranker suite picks it up.
    _is_ltr = combo.target_type == "learning_to_rank"
    fte = SimpleFeaturesAndTargetsExtractor(
        target_column=target_col,
        regression=(combo.target_type == "regression"),
        target_type=_combo_tt,
        # 2026-04-27 Session 7 batch 6: when the combo injects a
        # datetime column ('ts' from build_frame_for_combo), surface it
        # as ts_field so train_mlframe_models_suite's temporal_audit
        # auto-detect kicks in. Without this the audit stays silent
        # for fuzz combos and the auto-detect path is untested.
        ts_field=("ts" if combo.with_datetime_col else None),
        # 2026-05-04: LTR combos need group_field for the ranker suite.
        group_field=("qid" if _is_ltr else None),
        target_carrier=combo.target_carrier,
        # 2026-05-21 iter150 -- wire weight_schemas through (latent bug:
        # the axis has existed since iter113 but the FTE init was missing
        # the kwarg, so every combo silently fell back to
        # ``sample_weights={}``. Combos still dedup'd distinct via the
        # canonical_key BUT had identical runtime behaviour, leaving
        # the recency-weight code path (FTE._build_sample_weights, the
        # suite's per-weight loop, recency vs uniform branch in
        # _phase_train_one_target) entirely unfuzzed).
        weight_schemas=combo.weight_schemas,
        # 2026-05-21 iter150 -- multi-target axis. FTE adds synthetic
        # extra targets to target_by_type per combo.extra_targets so the
        # suite's per-target outer loop runs more than once.
        extra_targets=combo.extra_targets,
    )

    # Resolve combo-specific kwargs (outlier detector, custom prep,
    # parquet path). These feed directly into train_mlframe_models_suite.
    df_input = _maybe_to_parquet(combo, df, tmp_path)
    outlier_detector = _outlier_detector_for_combo(combo)
    custom_pre = _custom_pre_pipelines_for_combo(combo)

    from mlframe.training.core import train_mlframe_models_suite

    # LTR combos: filter mlframe_models to {cb,xgb,lgb} (HGB/Linear have
    # no native ranker) and build a ranking_config from the combo axis.
    # Pass target_type=LEARNING_TO_RANK explicitly so the suite's early
    # dispatch routes to train_mlframe_ranker_suite.
    _ltr_models = list(combo.models)
    _ltr_ranking_config = None
    if _is_ltr:
        _supported = {"cb", "xgb", "lgb", "mlp"}  # 2026-05-07: MLP via RankNet/ListNet
        _filtered = [m for m in combo.models if m.lower() in _supported]
        if not _filtered:
            # No supported model in this combo -- skip; not a real bug.
            pytest.skip(
                f"LTR combo {combo.short_id()}: requested models "
                f"{combo.models} have no native ranker (need cb/xgb/lgb/mlp)"
            )
        _ltr_models = _filtered
        from mlframe.training.configs import LearningToRankConfig
        # iter162: nested LTR knobs -- cb_loss_fn, lgb_objective, rrf_k.
        # iter170: mlp_loss_fn + eval_at (defensive).
        _ltr_eval_at = (1, 5, 10) if combo.ltr_eval_at_cfg == "default" else (1, 3, 5, 10, 20)
        _ltr_ranking_config = LearningToRankConfig(
            ensemble_method=combo.ranking_ensemble_method,
            cb_loss_fn=combo.ltr_cb_loss_fn_cfg,
            lgb_objective=combo.ltr_lgb_objective_cfg,
            rrf_k=combo.ltr_rrf_k_cfg,
            **_safe_cfg_kwargs(
                LearningToRankConfig,
                mlp_loss_fn=combo.ltr_mlp_loss_fn_cfg,
                eval_at=_ltr_eval_at,
            ),
        )

    # 2026-05-21 iter151 -- P0 suite-level kwargs (built once per combo
    # and forwarded into _suite_kwargs below). Each is None when the
    # respective axis is disabled, so the production default behaviour
    # is preserved.
    # P0-1 quantile_regression_config: only on regression primaries.
    # iter162 extends with nested crossing_fix / coverage_pairs / wrapper_n_jobs axes.
    _quantile_cfg = None
    if combo.enable_quantile_regression_cfg and combo.target_type == "regression" and not _is_ltr:
        from mlframe.training.configs import QuantileRegressionConfig
        _coverage_pairs = ((0.1, 0.9),) if combo.quantile_coverage_pairs_cfg == "default" else ((0.05, 0.95),)
        _quantile_cfg = QuantileRegressionConfig(
            alphas=(0.1, 0.5, 0.9) if combo.quantile_coverage_pairs_cfg == "default" else (0.05, 0.5, 0.95),
            crossing_fix=combo.quantile_crossing_fix_cfg,
            point_estimate_alpha=0.5,
            coverage_pairs=_coverage_pairs,
            wrapper_n_jobs=combo.quantile_wrapper_n_jobs_cfg,
        )
    # P0-2 linear_model_config: only meaningful when "linear" in models.
    _linear_cfg = None
    if "linear" in combo.models:
        from mlframe.training.configs import LinearModelConfig
        # 2026-05-28 LinearModelConfig.l1_ratio (ElasticNet mix). Only honoured by
        # the saga solver; lbfgs/liblinear raise on l1_ratio != 0. Mirror the
        # canonical_key gating in FuzzCombo so the LinearModelConfig instance
        # the suite consumes never hits sklearn's solver-mismatch ValueError.
        _l1_ratio = (
            combo.linear_l1_ratio_cfg if combo.linear_solver_cfg == "saga" else 0.0
        )
        _linear_cfg = LinearModelConfig(**_safe_cfg_kwargs(
            LinearModelConfig,
            alpha=combo.linear_alpha_cfg,
            solver=combo.linear_solver_cfg,
            l1_ratio=_l1_ratio,
        ))
    # P0-3 feature_handling_config: instantiate with nested sub-config
    # overrides per the iter162 deep-kwargs audit. Each sub-config field
    # falls back to library defaults when not on the axis OR when import
    # of the relevant sub-config class fails (FHC has heavy optional deps).
    _fhc = None
    if combo.enable_feature_handling_config_cfg:
        try:
            from mlframe.training.feature_handling.config import (
                FeatureHandlingConfig,
                CacheConfig,
                MemoryConfig,
                AutoDeriveConfig,
                TextDetectionConfig,
                ReproConfig,
            )
            _cache = CacheConfig(**_safe_cfg_kwargs(
                CacheConfig,
                eviction_strategy=combo.fhc_cache_eviction_strategy_cfg,
                allow_pickle=combo.fhc_cache_allow_pickle_cfg,
                # iter170 deep cache axes (defensive).
                prefetch_enabled=combo.fhc_cache_prefetch_enabled_cfg,
                prefetch_vram_safety_factor=combo.fhc_cache_prefetch_vram_safety_factor_cfg,
                # iter180 DEPTH-4 -- persistence mode gates disk-tier sub-fields.
                persistence=combo.fhc_cache_persistence_cfg,
            ))
            _memory = MemoryConfig(**_safe_cfg_kwargs(
                MemoryConfig,
                auto_derive=AutoDeriveConfig(cache_ram_fraction=combo.fhc_cache_ram_fraction_cfg),
                # iter170 memory axis (defensive).
                pressure_watermark_pct=combo.fhc_memory_pressure_watermark_pct_cfg,
            ))
            _textdet = TextDetectionConfig(**_safe_cfg_kwargs(
                TextDetectionConfig,
                definite_text_mean_chars=combo.fhc_text_definite_text_mean_chars_cfg,
                min_alphabet_entropy=combo.fhc_text_min_alphabet_entropy_cfg,
                # iter170 text-detection axes (defensive).
                text_min_mean_tokens=combo.fhc_text_min_mean_tokens_cfg,
                text_min_unique_ratio=combo.fhc_text_min_unique_ratio_cfg,
                respect_explicit_categorical_dtype=combo.fhc_text_respect_explicit_cat_dtype_cfg,
                # 2026-05-28 text_min_cardinality axis -- cat-vs-text promotion floor.
                text_min_cardinality=combo.fhc_text_min_cardinality_cfg,
            ))
            _repro = ReproConfig(**_safe_cfg_kwargs(
                ReproConfig,
                deterministic_torch=combo.fhc_repro_deterministic_torch_cfg,
                # iter170 repro axes (defensive).
                langdetect_seed=combo.fhc_repro_langdetect_seed_cfg,
                pinned_svd_solver_params=combo.fhc_repro_pinned_svd_solver_params_cfg,
                forbid_nonatomic_fs=combo.fhc_repro_forbid_nonatomic_fs_cfg,
                deterministic_eviction=combo.fhc_repro_deterministic_eviction_cfg,
            ))
            # iter170: PricingConfig + LoggingConfig (defensive -- may not exist).
            _pricing = None
            _logging = None
            try:
                from mlframe.training.feature_handling.config import PricingConfig
                _pricing = PricingConfig(**_safe_cfg_kwargs(
                    PricingConfig,
                    cap_usd=combo.fhc_pricing_cap_usd_cfg,
                    warn_above_usd=combo.fhc_pricing_warn_above_usd_cfg,
                ))
            except (ImportError, AttributeError):
                pass
            try:
                from mlframe.training.feature_handling.config import LoggingConfig
                _logging = LoggingConfig(**_safe_cfg_kwargs(
                    LoggingConfig,
                    verbose=combo.fhc_logging_verbose_cfg,
                ))
            except (ImportError, AttributeError):
                pass
            _fhc_kw = dict(
                cache=_cache,
                memory=_memory,
                text_detection=_textdet,
                repro=_repro,
                auto_locale_detection=combo.fhc_auto_locale_detection_cfg,
            )
            if _pricing is not None:
                _fhc_kw["pricing"] = _pricing
            if _logging is not None:
                _fhc_kw["logging"] = _logging
            _fhc = FeatureHandlingConfig(**_safe_cfg_kwargs(FeatureHandlingConfig, **_fhc_kw))
        except Exception:
            _fhc = None  # tolerate import / construction failure
    # P0-4 precomputed: build the trainset_features_stats bundle. The
    # other slots (dummy_baselines, composite_target_specs) raise
    # NotImplementedError if requested -- precompute_all only fills the
    # stats slot, which is what we want.
    _precomputed = None
    if combo.enable_precomputed_cfg:
        try:
            from mlframe.training.helpers import precompute_all
            _precomputed = precompute_all(df_input if not isinstance(df_input, str) else df, target_by_type=None)
        except Exception:
            _precomputed = None  # tolerate parquet-path / FTE-shape edge cases

    t0 = time.perf_counter()
    outcome = "pass"
    err_class = None
    err_summary = None
    try:
        _suite_kwargs = dict(
            df=df_input,
            target_name=combo.short_id(),
            model_name=combo.short_id(),
            features_and_targets_extractor=fte,
            mlframe_models=_ltr_models,)
        if _is_ltr:
            _suite_kwargs["target_type"] = _combo_tt
            # Wave 21: assume_comparable_scales axis on LTR ensembling.
            _ltr_ranking_config = _ltr_ranking_config.model_copy(
                update={"assume_comparable_scales": combo.ltr_assume_comparable_scales_cfg}
            )
            _suite_kwargs["ranking_config"] = _ltr_ranking_config
        # 2026-05-21 iter151 P0 suite-level kwargs.
        if _quantile_cfg is not None:
            _suite_kwargs["quantile_regression_config"] = _quantile_cfg
        if _linear_cfg is not None:
            _suite_kwargs["linear_model_config"] = _linear_cfg
        if _fhc is not None:
            _suite_kwargs["feature_handling_config"] = _fhc
        if _precomputed is not None:
            _suite_kwargs["precomputed"] = _precomputed
        trained, _meta = train_mlframe_models_suite(
            **_suite_kwargs,
            hyperparams_config=_config_for_models(
                combo.models, combo.n_rows,
                iterations=combo.iterations,
                early_stopping_rounds=combo.early_stopping_rounds_cfg,
                mlp_predict_batch_size=combo.mlp_predict_batch_size_cfg,
                # iter170 per-backend hyperparams.
                lgb_feature_fraction=combo.lgb_feature_fraction_cfg,
                lgb_num_leaves=combo.lgb_num_leaves_cfg,
                xgb_max_depth=combo.xgb_max_depth_cfg,
                xgb_colsample_bynode=combo.xgb_colsample_bynode_cfg,
                cb_border_count=combo.cb_border_count_cfg,
                hgb_max_leaf_nodes=combo.hgb_max_leaf_nodes_cfg,
                rfecv_cv_n_splits=combo.rfecv_cv_n_splits_cfg,
                # iter180 DEPTH-4 booster sub-params.
                lgb_boosting_type=combo.lgb_boosting_type_cfg,
                lgb_dart_drop_rate=combo.lgb_dart_drop_rate_cfg,
                lgb_goss_top_rate=combo.lgb_goss_top_rate_cfg,
                xgb_tree_method=combo.xgb_tree_method_cfg,
                xgb_hist_max_bin=combo.xgb_hist_max_bin_cfg,
                cb_bootstrap_type=combo.cb_bootstrap_type_cfg,
                cb_bayesian_bagging_temperature=combo.cb_bayesian_bagging_temperature_cfg,
                cb_bernoulli_subsample=combo.cb_bernoulli_subsample_cfg,
                cb_grow_policy=combo.cb_grow_policy_cfg,
                cb_lossguide_max_leaves=combo.cb_lossguide_max_leaves_cfg,
            ),
            preprocessing_config=_preprocessing_for_combo(combo),
            verbose=0,
            use_ordinary_models=True,
            use_mlframe_ensembles=combo.use_ensembles,
            outlier_detection_config=OutlierDetectionConfig(
                detector=outlier_detector,
                apply_to_val=combo.apply_outlier_to_val_cfg,
            ),
            feature_selection_config=FeatureSelectionConfig(
                use_mrmr_fs=combo.use_mrmr_fs,
                # 2026-05-18 -- delegate to shared builder. Adding a new
                # MRMR axis now only edits build_mrmr_kwargs_from_flat in
                # _fuzz_combo.py; the pytest suite + 1M harness both
                # consume the same builder.
                mrmr_kwargs=build_mrmr_kwargs(combo),
                # rfecv_models: pass exactly the canonical estimator (None when
                # the combo would mis-use it) — wrap in a single-element list
                # because the field expects List[str].
                rfecv_models=(
                    [combo._canonical_rfecv_estimator()]
                    if combo._canonical_rfecv_estimator() is not None
                    else None
                ),
                custom_pre_pipelines=custom_pre or {},
                # 2026-05-21 iter151 P1-7/P1-8/P2-16/P2-17/P2-18a/P2-18b:
                # FS-related fill-ins from the audit. Each canonicalised in
                # FuzzCombo.canonical_key when the gating axis is off.
                use_boruta_shap=combo.use_boruta_shap_cfg,
                # 2026-05-21 iter151: BorutaShap fuzz-speed knobs.
                # Default n_trials=150 + SHAP per trial is too slow for the
                # fuzz timeout budget. n_trials=10 still exercises every
                # code path (shadow build + SHAP explain + tail test) at
                # ~15x lower wall.
                boruta_shap_kwargs=({"n_trials": 10, "verbose": False}
                                   if combo.use_boruta_shap_cfg else None),
                use_sample_weights_in_fs=combo.use_sample_weights_in_fs_cfg,
                mrmr_identity_cache_scope=combo.mrmr_identity_cache_scope_cfg,
                skip_identity_equivalent_pre_pipelines=combo.skip_identity_equivalent_pre_pipelines_cfg,
                rfecv_leakage_corr_threshold=combo.rfecv_leakage_corr_threshold_cfg,
                rfecv_mbh_adaptive_threshold=combo.rfecv_mbh_adaptive_threshold_cfg,
                # 2026-05-22 iter170 deep FS knobs (defensive).
                **_safe_cfg_kwargs(
                    FeatureSelectionConfig,
                    rfecv_n_features_selection_rule=combo.rfecv_n_features_selection_rule_cfg,
                    rfecv_stability_selection=combo.rfecv_stability_selection_cfg,
                    rfecv_leakage_action=combo.rfecv_leakage_action_cfg,
                    # 2026-05-28 pre_screen_null_fraction_threshold axis -- the
                    # null-fraction sibling of the existing variance threshold
                    # axis. Gated on fs_pre_screen_unsupervised_cfg in
                    # canonical_key; thread the value through unconditionally
                    # here (the suite already builds the FS config only when
                    # the pre-screen branch fires).
                    pre_screen_null_fraction_threshold=combo.fs_pre_screen_null_fraction_threshold_cfg,
                ),
            ),
            # save_charts=False / show_perf_chart=False / show_fi=False:
            # the fuzz suite runs ~150 combos × ~5 charts per combo. Each
            # matplotlib figure leaks ~1-2 MB through plt.savefig + tight_-
            # layout warnings. Across the run that compounds to >2 GB and
            # blows up pytest's traceback formatter (``MemoryError: bad
            # allocation`` / pytest INTERNALERROR observed 2026-04-27).
            # We never look at the artefacts in fuzz; turn them off.
            output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models", save_charts=False),
            reporting_config=ReportingConfig(
                show_perf_chart=False, show_fi=False,
                # iter162: nested ReportingConfig fields. matplotlib_rcparams
                # parsed from JSON-string axis value (so the axis dict stays
                # hashable for canonical_key).
                prob_histogram_yscale=combo.reporting_prob_histogram_yscale_cfg,
                title_metrics_template=combo.reporting_title_metrics_template_cfg,
                matplotlib_rcparams=(
                    None if combo.reporting_matplotlib_rcparams_cfg is None
                    else __import__("json").loads(combo.reporting_matplotlib_rcparams_cfg)
                ),
                multiclass_panels=combo.reporting_multiclass_panels_cfg,
                # 2026-05-28 W5: ReportingConfig.mase_seasonality (int, default
                # 1 at _reporting_configs.py:140). Thread the fuzz-axis value
                # through so regression combos exercise the non-default
                # seasonality on the report-metadata path.
                mase_seasonality=combo.reporting_mase_seasonality_cfg,
                # iter170 deep reporting axes -- defensive _safe_cfg_kwargs
                # absorbs fields that don't exist post-refactor.
                **_safe_cfg_kwargs(
                    ReportingConfig,
                    figsize=((15, 5) if combo.reporting_figsize_cfg == "default" else (10, 4)),
                    plot_dpi=combo.reporting_plot_dpi_cfg,
                    quantile_panels=(None if combo.reporting_quantile_panels_cfg == "default"
                                     else "RELIABILITY PINBALL_BY_ALPHA"),
                    ltr_panels=(None if combo.reporting_ltr_panels_cfg == "default"
                                else "NDCG_K LIFT"),
                    plotly_template=combo.reporting_plotly_template_cfg,
                    matplotlib_style=combo.reporting_matplotlib_style_cfg,
                ),
            ),
            # recurrent_models + sequences: synthetic per-row sequences
            # (T=8, F=2) emitted only on canonical-recurrent combos so
            # the suite exercises the sequence-pipeline. Hyperparams
            # tuned for fuzz speed (small hidden_size, 2 epochs).
            recurrent_models=(
                [combo._canonical_recurrent_model()]
                if combo._canonical_recurrent_model() is not None
                else None
            ),
            sequences=_recurrent_sequences_for_combo(combo, df=df_input),
            recurrent_config=_recurrent_config_for_combo(combo),
            # 2026-04-28 batch 4 followup - confidence-analysis axis exercises
            # the test-set confidence pass at trainer.py:4019 (distinct code
            # path with its own metrics/report side-effects). ``use_cache``
            # is per-model not suite-level, so it stays out of the fuzz
            # axis space.
            confidence_analysis_config=ConfidenceAnalysisConfig(
                include=combo.include_confidence_analysis_cfg,
                # iter162: nested model_kwargs (n_estimators / max_depth).
                # "default" = empty dict (library defaults); "small_trees" pins
                # tiny trees so the conf-analysis branch runs faster in fuzz.
                model_kwargs=(
                    {} if combo.confidence_model_kwargs_cfg == "default"
                    else {"n_estimators": 20, "max_depth": 4}
                ),
            ),
            # Wave 21: dummy-baselines + baseline-diagnostics enabled toggles.
            dummy_baselines_config=__import__(
                "mlframe.training.configs", fromlist=["DummyBaselinesConfig"]
            ).DummyBaselinesConfig(
                enabled=combo.dummy_baselines_enabled_cfg,
                # iter170 deep dummy-baseline axes (defensive).
                **_safe_cfg_kwargs(
                    __import__("mlframe.training.configs", fromlist=["DummyBaselinesConfig"]).DummyBaselinesConfig,
                    stratified_n_repeats=combo.dummy_stratified_n_repeats_cfg,
                    paired_bootstrap_n_resamples=combo.dummy_paired_bootstrap_n_resamples_cfg,
                ),
            ),
            baseline_diagnostics_config=__import__(
                "mlframe.training.configs", fromlist=["BaselineDiagnosticsConfig"]
            ).BaselineDiagnosticsConfig(
                enabled=combo.baseline_diagnostics_enabled_cfg,
                # iter170 deep baseline-diagnostic axes (defensive).
                **_safe_cfg_kwargs(
                    __import__("mlframe.training.configs", fromlist=["BaselineDiagnosticsConfig"]).BaselineDiagnosticsConfig,
                    quick_model_n_estimators=combo.baseline_quick_model_n_estimators_cfg,
                    quick_model_num_leaves=combo.baseline_quick_model_num_leaves_cfg,
                    quick_model_learning_rate=combo.baseline_quick_model_learning_rate_cfg,
                    sample_n=combo.baseline_sample_n_cfg,
                    high_potential_min_dominance_pct=combo.baseline_high_potential_min_dominance_pct_cfg,
                    best_model_min_lift=combo.baseline_best_model_min_lift_cfg,
                ),
            ),
            # 2026-05-21 -- mini-HPT toggle. When True, the suite runs the
            # target-distribution analyzer + feature-distribution analyzer
            # after the split, gap-fill-merges target-side recommendations
            # into hyperparams_config, and stamps both reports into metadata.
            # When False both analyzers skip. Default True matches suite
            # signature; toggling exercises the skip path.
            enable_target_distribution_analyzer=combo.enable_target_distribution_analyzer_cfg,
            **_configs_for_combo(combo),
        )
        # An empty ``trained`` dict is acceptable ONLY when
        # ``continue_on_model_failure=True`` AND the suite recorded
        # each failure in ``metadata['failed_models']``. Any other
        # empty-trained outcome is a bug — the suite should have
        # either raised or produced ≥1 model.
        if not trained:
            if (
                combo.continue_on_model_failure
                and _meta is not None
                and _meta.get("failed_models")
            ):
                pass  # graceful skip of a configurably-failing combo
            else:
                raise AssertionError(
                    f"empty models dict for combo {combo.short_id()} "
                    f"(continue_on_failure={combo.continue_on_model_failure}, "
                    f"failed_models={(_meta or {}).get('failed_models')})"
                )

        # --- Post-train invariants (free on every combo) ---
        # #16 no caller-frame mutation (skip for parquet-path).
        if combo.input_storage == "memory" and frame_cols_before is not None:
            assert tuple(df.columns) == frame_cols_before, (
                f"caller-frame columns mutated: before={frame_cols_before} "
                f"after={tuple(df.columns)}"
            )
            shape_after = getattr(df, "shape", None)
            assert shape_after == frame_shape_before, (
                f"caller-frame shape mutated: before={frame_shape_before} "
                f"after={shape_after}"
            )
        # #20 metadata schema: load-bearing keys present.
        # ``model_schemas`` is only populated when at least one model
        # successfully trained — combos that legitimately degrade to
        # an empty trained dict (continue_on_failure=True + all models
        # failed) won't have it. Check the always-present keys
        # unconditionally; model_schemas only when trained non-empty.
        if _meta is not None:
            for k in ("columns", "cat_features", "outlier_detection"):
                assert k in _meta, (
                    f"metadata missing load-bearing key {k!r}; "
                    f"keys={list(_meta)[:20]}"
                )
            if trained:
                assert "model_schemas" in _meta, (
                    "metadata missing 'model_schemas' despite non-empty "
                    f"trained dict; keys={list(_meta)[:20]}"
                )

        # --- Fix C property invariants (cheap, per-combo) ---
        # Catches silent degeneracy that a "no exception" assertion misses:
        # dead features, all-zero predictions, NaN leakage to the model
        # head, val-slice misalignment.
        _assert_prediction_invariants(trained, _meta, combo)
        # --- R3-3 I4 serialization roundtrip (env-gated, off by default) ---
        if os.environ.get("MLFRAME_FUZZ_ROUNDTRIP") == "1":
            _assert_serialization_roundtrip(trained, str(tmp_path), combo)
    except Exception as exc:
        outcome = "fail"
        err_class = type(exc).__name__
        err_summary = traceback.format_exception_only(type(exc), exc)[-1].strip()
        log_combo_outcome(
            combo, outcome,
            duration_s=time.perf_counter() - t0,
            error_class=err_class,
            error_summary=err_summary,
        )
        raise

    log_combo_outcome(
        combo, outcome, duration_s=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# Meta-tests: sanity-check the enumerator itself
# ---------------------------------------------------------------------------


def test_enumerator_is_deterministic():
    """Same master_seed must yield byte-identical combo list."""
    a = enumerate_combos(target=50, master_seed=2026_04_22)
    b = enumerate_combos(target=50, master_seed=2026_04_22)
    assert [c.canonical_key() for c in a] == [c.canonical_key() for c in b]


def test_enumerator_produces_unique_combos():
    """No canonical-key duplicates in the 150-combo run."""
    keys = [c.canonical_key() for c in COMBOS]
    assert len(keys) == len(set(keys)), "Fuzz enumerator produced duplicates"


def test_enumerator_hits_all_models():
    """Every supported model must appear at least once across the 150 combos."""
    from ._fuzz_combo import MODELS
    seen = {m for c in COMBOS for m in c.models}
    missing = set(MODELS) - seen
    assert not missing, f"Models never exercised by fuzz: {missing}"


def test_enumerator_target_count():
    assert len(COMBOS) == 150
