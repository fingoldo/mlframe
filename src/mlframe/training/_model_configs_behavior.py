"""Behaviour / dispatch / LTR / quantile config models carved out of ``_model_configs.py`` to keep that file under the 1k-LOC monolith ceiling. The parent re-exports these from its bottom so existing ``from mlframe.training._model_configs import X`` and ``from mlframe.training.configs import X`` imports keep resolving."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional, Union

from pydantic import Field, model_validator

from ._configs_base import (
    DEFAULT_CALIBRATION_BINS,
    DEFAULT_FAIRNESS_MIN_POP_CAT_THRESH,
    BaseConfig,
)


class TrainingBehaviorConfig(BaseConfig):
    """Training behavior flags and control settings.

    Replaces the legacy untyped ``control_params`` / ``control_params_override`` dicts.
    Controls *how* training runs (GPU, calibration, fairness, verbosity) rather than
    model hyperparameters.

    Parameters
    ----------
    prefer_gpu_configs : bool
        Whether to prefer GPU model configurations.
    prefer_cpu_for_lightgbm : bool
        Force LightGBM to CPU even when GPU is available.
    prefer_calibrated_classifiers : bool
        Use calibrated classifier variants (CalibratedClassifierCV wrappers).
    use_robust_eval_metric : bool
        Use robust evaluation metrics.
    nbins : int
        Number of bins for calibration reports.
    xgboost_verbose : int
        Verbosity level for XGBoost training.
    rfecv_model_verbose : int
        Verbosity level for RFECV models.
    fairness_features : list of str, optional
        Feature names for fairness analysis.
    fairness_min_pop_cat_thresh : int
        Minimum population per category for fairness analysis.
    metamodel_func : Callable, optional
        Function to wrap models (e.g., for target transformation).
    default_classification_scoring : dict, optional
        Custom classification scoring configuration.
    default_regression_scoring : dict, optional
        Custom regression scoring configuration.
    callback_params : dict, optional
        Parameters for training callbacks (patience, verbose).
    prefer_cpu_for_xgboost : bool
        Force XGBoost to CPU even when GPU is available.
    cont_nbins : int
        Number of bins for continuous features in fairness subgroups.
    cb_fit_params : dict, optional
        Extra kwargs passed to CatBoost .fit() (e.g. early_stopping_rounds, custom callbacks).
    enable_crash_reporting : bool
        Default True. At suite start, enable faulthandler (SIGSEGV /
        SIGABRT -> Python traceback) and on Windows suppress the
        "Python has stopped working" WER popup so Jupyter kernels exit
        cleanly instead of hanging. No-op if already enabled in the
        process.
    continue_on_model_failure : bool
        If True, catch exceptions from individual per-model training
        (e.g. XGBoost ``bad_malloc`` on too-large frames) and continue
        the suite with the next model/weighting instead of aborting
        the whole run. Crashes that kill the process at the OS level
        (access violation in a worker thread that faulthandler can't
        catch) will still terminate -- for true isolation use subprocess
        training, which this flag does NOT provide.
    """

    prefer_gpu_configs: bool = True
    # Opt-in (B): after each regression model trains, evaluate TTA predictive-uncertainty quality on the
    # model-ready transformed test frame (live in the per-target body) and stamp metrics into
    # metadata["uncertainty_eval"] (TTA-vs-point RMSE, spread<->error corr). Numeric features only; default OFF.
    uncertainty_eval: bool = False
    # Opt-in (E3): when the target-distribution analyzer flags a heavy-tail / skew / multi-modal target, train the
    # matching composite estimator (TailComposite / CompositeDistribution) alongside the requested models so it is
    # actually fit/evaluated, not merely advised. Regression-only; the base column is auto-picked (max |corr| to y).
    # Default OFF -- it adds one extra model per regression target. Requires ``enable_target_distribution_analyzer``.
    distribution_driven_estimator: bool = False
    prefer_cpu_for_lightgbm: bool = True
    prefer_cpu_for_xgboost: bool = False
    prefer_calibrated_classifiers: bool = True
    use_robust_eval_metric: bool = True
    nbins: int = DEFAULT_CALIBRATION_BINS
    xgboost_verbose: int = 0
    rfecv_model_verbose: int = 0
    fairness_features: Optional[List[str]] = None
    fairness_min_pop_cat_thresh: int = DEFAULT_FAIRNESS_MIN_POP_CAT_THRESH
    cont_nbins: int = 6
    metamodel_func: Optional[Callable] = None
    default_classification_scoring: Optional[Dict[str, Any]] = None
    default_regression_scoring: Optional[Dict[str, Any]] = None
    callback_params: Optional[Dict[str, Any]] = None
    cb_fit_params: Optional[Dict[str, Any]] = None
    # Default True: faulthandler + Windows WER suppression are pure diagnostics -- they don't change training behavior, only replace the "Python has stopped working" modal with a Python traceback. Users who rely on the WER popup (rare) can opt out.
    enable_crash_reporting: bool = True
    # Default False: silently skipping a failed model is a semantic shift that users must opt into explicitly.
    continue_on_model_failure: bool = False
    # Per-target binary decision-threshold tuning on val/OOF (NEVER test), maximising ``tune_decision_threshold_metric``; the tuned value is stamped into ``metadata["decision_thresholds"]`` so predict reuses it (0.5 stays the leak-safe fallback). Affects only hard-label ``preds``; probabilities are untouched.
    # Tri-state: "auto" (default) tunes only when the val target is imbalanced (minority fraction < DECISION_THRESHOLD_IMBALANCE_FRACTION) and leaves 0.5 on balanced targets where val-tuning only adds variance; True always tunes; False forces 0.5. bool kept for back-compat.
    tune_decision_threshold: Union[bool, str] = "auto"
    # "balanced_accuracy" (default) recovers the Bayes-optimal operating point ~10x closer than "f1" and wins test balanced-accuracy in 29/30 imbalance x seed cells (bench_threshold_objective.py); F1 chases precision/recall trade and drifts the threshold high under imbalance.
    tune_decision_threshold_metric: str = "balanced_accuracy"  # "f1" or "balanced_accuracy"

    # Canonical monotonic strict-decline overfitting-stop knob (default 5). Threads through to the lgb / xgb
    # shims' ``.fit(monotonic_decline_patience=...)`` and the CatBoost ``callback_params`` so a single value
    # controls the byte-identical ``MonotonicDeclineStopper`` rule across all three boosters (and mirrors the
    # neural / sklearn-wrapper paths). The detector stops once the monitored val metric strictly worsens for
    # this many CONSECUTIVE rounds since the global best (a new best, a plateau, or a bounce-up all reset the
    # streak). ``None`` disables it entirely -- the off-switch -- so the booster trains to its full iteration
    # cap unless its native ``early_stopping_rounds`` fires first.
    # N=5 calibrated by bench_worsening_vs_monotonic_stop.py (18 lgb fits): ties patience-level holdout
    # accuracy while stopping ~3.6x earlier than the full budget; N=3 was too aggressive (catastrophic
    # 7-tree stops). The old budget-scaled ``early_stop_on_worsening`` detector was REMOVED (benchmarked
    # no-op: stopped at 499/500 in 18/18 fits, and had the worst test accuracy).
    monotonic_decline_patience: Optional[int] = 7

    # Per-iteration FULL-metric-suite capture for meta-learning / HPO-from-early-observation. When enabled, a
    # per-round (booster) / per-epoch (neural) callback computes the complete target-type metric suite
    # (``metrics.compute_all_metrics``) on the val set and stores the trajectory in ``model.iteration_metrics_``
    # ({iteration -> {metric_name -> float}}). Downstream meta-learners predict final holdout performance from the
    # first K iterations and prune bad configs early. Defaults differ by model family because the cost profile does:
    #   - NEURAL: default ON. Val predictions are already concatenated at each validation-epoch-end, so the marginal
    #     cost is only the (cheap, numba-kernel) metric suite -- essentially free.
    #   - BOOSTERS (lgb/xgb/cb): default OFF. The cost driver is re-PREDICTING the val set at each round via the
    #     booster's native per-iteration prediction; that is non-trivial at production val sizes, so it is opt-in and
    #     stride-sampled. ``None`` is treated as the family default; True / False force it on / off for both families.
    capture_iteration_metrics: Optional[bool] = None
    # Booster stride for ``capture_iteration_metrics``: capture every Kth round (the best/last round is always
    # captured regardless). 1 = every round. Bounds the per-round predict+metric overhead. Ignored for neural
    # (every validated epoch is captured -- the preds are already materialised so there is nothing to amortise).
    iteration_metrics_stride: int = 1

    # Default True: ``_trainer_train_and_evaluate.maybe_wrap_for_partial_fit_es``
    # auto-wraps linear/ridge/lasso/elasticnet/huber/sgd/ransac models in
    # ``PartialFitESWrapper`` when an X_val/y_val pair is available so val drives
    # ES via partial_fit (SGD-family) or a dichotomic budget search
    # (iterative-solver linear-family). Set to False to skip the wrap entirely
    # and reach the underlying estimator unchanged -- intended for parity
    # benchmarks against the pre-wrap legacy path and for off-switch operator
    # control; NOT a perf knob (the wrap is essentially free at construction
    # time; the bench value is the ES decision shift it enables).
    auto_wrap_partial_fit_es: bool = True
    # Default False: feature-drift-driven per-target MLP HPT override
    # is OFF by default. The drift sensor still runs and stamps the
    # recommendation into metadata + emits a WARN log line so the
    # operator sees the recommendation, but the MLP config the user
    # passed is NOT mutated. Set True to enable auto-apply (regression
    # only by default; classification requires the shape-detector gate
    # below to also be satisfied).
    #
    # Rationale: the override is a black-box config rewrite and the
    # paired study showed classification doesn't have a clean trigger
    # (Pearson r=-0.101 overall; interaction-rich classification
    # targets are actively hurt by the override). Operators who want
    # the override can opt in after reading the docs; everyone else
    # gets the MLP they configured.
    feature_drift_auto_apply_neural_overrides: bool = False

    # Default True: align Polars Categorical dicts across
    # train/val/test via shared pl.Enum(union_of_categories) before
    # model training. Mechanism not fully understood but empirically
    # prevents a silent process kill on Windows when XGB constructs
    # val IterativeDMatrix with ref=train on large frames (7.3M+ rows,
    # 15+ cat features).
    # Theory: pl.Categorical assigns physical codes per-Series
    # (order-of-first-occurrence), so the same string can have
    # different physical codes in train vs val vs test. XGB's native
    # layer at scale appears to treat val's physical codes as indices
    # into train's bin structure without re-reading the dict,
    # corrupting memory. pl.Enum(list) enforces a shared dict
    # by construction so physical codes are consistent across splits.
    # Disable to reproduce the pre-fix behavior or if the alignment
    # cost (O(n_rows) per cat column) is prohibitive.
    align_polars_categorical_dicts: bool = True

    # Silencing knobs for verbose report blocks.
    #
    # ``report_residual_audit``: when False, ``report_model_perf`` skips the multi-line residual-audit footer (moments / shape / hetero / hypothesis / suggested-loss block). Default True (informative for regression diagnostics); set False on production runs where the block adds 6-8 noisy lines per (model x split).
    #
    # ``confidence_ensemble_quantile``: top-quantile of MOST-CONFIDENT rows used by the "Conf Ensemble" flavors. Default 0.1 (= top 10%); set 0.0 to disable Conf Ensembles entirely (saves ~6 flavor x 2 split = 12 log blocks + their charts per ensemble pass). The raw ensemble metrics still print - only the confidence-subset variant is suppressed.
    report_residual_audit: bool = True
    confidence_ensemble_quantile: float = 0.1

    # When True (default), the simple-ensembling blends (arithm / harm / quad / qube / geo / median) consume AP12-calibrated probs stamped by ``post_calibrate_model`` (``member.calibrated_val_probs`` / ``calibrated_test_probs``) instead of raw ``member.val_probs`` / ``test_probs``. This dampens the heterogeneous-scale dominance bug flagged by ensembling-critique A3#3 (well-calibrated tree probs in [0.1, 0.9] dominated by raw sigmoid in [0.005, 0.01] under arithmetic mean). When False, every blend uses raw probs (legacy pre-W16D behaviour). RRF is rank-based and is unaffected either way (scale-invariant). Members without the AP12 stamp transparently fall back to raw probs -- the knob never raises on missing calibration.
    use_ap12_calibrated_probs_in_ensemble: bool = True

    # Pre-pipeline LRU bound. Default 4 covers the common Linear+MLP+RFECV+catboost suite without thrashing; long-running services with bigger model rosters can bump this without monkey-patching the module global.
    pre_pipeline_cache_max: int = 4

    # Fix 8 (2026-04-21): append a per-model input-schema fingerprint
    # (``__sch_<10 hex>``) to model filenames so two runs with different
    # feature-type configs (text vs cat promotion, encoding, alignment)
    # don't silently overwrite each other. Default True. Set False to
    # restore the pre-2026-04-21 naming scheme (``{model}_{weight}.dump``);
    # load-time schema verification is also skipped for those artefacts.
    model_file_hash_suffix: bool = True

    # 2026-04-26 Session 7: temporal target audit. When set, per-target
    # the suite computes a time-series view of the target (P(y=1) for
    # binary, mean(y) for regression) at the configured granularity,
    # detects change points / regime shifts, and warns when the rate
    # diverges across segments. Saves a chart to the per-target charts
    # folder. Skipped silently when the timestamp column is absent or
    # not datetime-typed.
    target_temporal_audit_column: Optional[str] = None
    """Column name (datetime-typed) used as the time axis for the
    per-target temporal audit. ``None`` (default) disables the audit.
    Set to e.g. ``'job_posted_at'`` to enable."""

    target_temporal_audit_granularity: str = "auto"
    """One of ``"auto"`` (default; picks granularity that yields 30-50
    bins) or one of ``"minute"`` / ``"hour"`` / ``"day"`` / ``"week"`` /
    ``"month"`` / ``"quarter"`` / ``"year"``."""

    target_temporal_audit_save_plot: bool = True
    """Save the time-series chart to the per-target charts folder."""

    # mini-HPT feature_distribution_analyzer auto-drop knobs. Both flags
    # operate on the TRAIN frame only - the analyzer's drop_candidates list
    # is derived purely from per-column train-side stats (NaN fraction,
    # variance, finite-count) and its redundant_feature_pairs from train-only
    # Pearson |corr| on a sampled stride. Dropping is therefore train-derived
    # only; val and test simply lose the same columns to keep schemas aligned.
    # The dropped columns are subsequently absent from the pipeline, polars-
    # pandas conversion, composite discovery, and model training -- the prod
    # frame regularly carries 40-100 such columns survivable only because
    # downstream filters re-screen them, at material RAM cost.
    auto_drop_distribution_analyzer_candidates: bool = True
    """When True (default), drop columns the feature_distribution_analyzer
    flagged as NaN-heavy (>=nan_fraction_threshold) or low-variance from
    train/val/test before Phase 3 pipeline fit. Set False to preserve the
    legacy diagnostic-only behaviour (columns stay; only metadata records
    the recommendation)."""

    auto_drop_near_duplicate_threshold: float = 0.999
    """When the feature_distribution_analyzer's redundant-pair detector
    finds a numeric pair with |Pearson corr| >= this threshold on the train
    frame (sampled to ``_REDUNDANT_SAMPLE_MAX_ROWS``), one of the two columns
    is auto-dropped. Default 0.999 targets EXACT duplicates only -- below
    that, redundancy can carry asymmetric information (different finite
    masks, different NaN handling, leverage on out-of-distribution rows).
    Set >1.0 to disable. The kept column is the one alphabetically smaller
    so the choice is reproducible across runs."""

    # Extreme-AR + group-aware skip for UNBOUNDED-OUTPUT models. When set,
    # skips fitting the models listed in ``extreme_ar_group_aware_skip_models``
    # on targets where lag1_corr >= mlp_extreme_ar_threshold AND the split is
    # group-aware. Default TRUE.
    #
    # Why default ON now: on this regime an unbounded-output model
    # (MLP / linear) trained on the ABSOLUTE target catastrophically
    # extrapolates on unseen test groups (observed: MLP TEST R2=-35,
    # RMSE 3928 vs lag-baseline 13.5; the pre-tanh activations saturate on
    # shifted-group features so predictions collapse to the train-range
    # extremes). The model is then ALWAYS dropped by the ensemble quality
    # gate anyway -- so training it just burns wall-time + RAM + a multi-GB
    # checkpoint for a result that never reaches the blend. Skipping is the
    # pragmatic default until a substantive fix lands.
    #
    # Criteria for what to gate (``extreme_ar_group_aware_skip_models``):
    # this is a COST optimisation, gated on EMPIRICAL collapse, NOT on the
    # architectural "bounded vs unbounded output" distinction (linear models
    # are unbounded too yet behave fine here -- see below). The real
    # correctness backstop is the ensemble quality gate, which drops members
    # by MEASURED performance regardless of model type.
    #
    # What collapses on a group-aware split + lag1~1.0 RAW target:
    #   * NEURAL nets (dense "mlp"/"ngb"; recurrent "lstm"/"gru"/"rnn"/
    #     "transformer"): a nonlinear body + BatchNorm + a saturating
    #     (tanh-to-train-range) output head means that under the test-group
    #     feature-distribution shift the activations saturate and the WHOLE
    #     prediction distribution collapses bimodally (observed MLP TEST
    #     R2=-35, pred_std=551% of target). They are also the most expensive
    #     members (Lightning fit + GB-scale checkpoint). Known-bad + costly
    #     + always dropped by the gate => skip the fit by default.
    # What does NOT collapse (so NOT gated):
    #   * LINEAR family (Ridge/Lasso/...): also unbounded output, but an
    #     L2-bounded LINEAR map of near-identity AR features just predicts
    #     ~= lag(target) and degrades GRACEFULLY -- good RMSE (Ridge 13.9 vs
    #     lag-baseline 13.5), only a few tail rows drift (caught by the
    #     prediction-envelope clip), and it is frequently useful in the
    #     blend. Add "linear" to the list only if a run shows it collapsing.
    #   * TREE boosters (cb/xgb/lgb/hgb): bounded by training leaf values,
    #     cannot predict outside the train target range at all.
    # Defaulting to the whole neural family means enabling an RNN/LSTM later
    # is protected automatically without re-reading this comment.
    #
    # IMPORTANT: the skip fires only on the RAW/absolute target. Composite
    # targets (diff / linres / residual) bound the variance and are exactly
    # where these models belong -- they are NEVER skipped (the per-target
    # body guards on is_composite_target_name before applying the skip).
    #
    # Backstops that still ship regardless of this skip:
    #   * ``_TTRWithEvalSetScaling.predict`` / prediction-envelope clip
    #     bound y_hat to [y_train_min - 3*std, y_train_max + 3*std].
    #   * Ensemble quality gate drops any member whose MAE exceeds 5x the
    #     member-median, so a bad model never poisons the final ensemble.
    # Substantive fix paths (orthogonal to this knob):
    #   * Residual-target neural net (composite diff/linres bound variance).
    #   * Drop per-group aggregate features from the neural input set.
    mlp_extreme_ar_group_aware_skip: bool = False
    extreme_ar_group_aware_skip_models: tuple = (
        "mlp",
        "ngb",
        "lstm",
        "gru",
        "rnn",
        "transformer",
    )
    mlp_extreme_ar_threshold: float = 0.99

    # PipelineCache byte budget as a FRACTION of TOTAL host RAM. The cache
    # holds transformed train/val/test frames per (imputer, scaler, encoder,
    # tier) variant (each ~7-10 GB on a 4M x 470 frame); the budget is also
    # clamped to currently-available RAM minus a 4 GB floor so it never
    # starves the in-flight model + transform allocations. A fraction of
    # TOTAL is host-predictable (unlike the old "available - 8 GB" which
    # drifted up to 64 GB and contributed to an OOM at 174 GB when training
    # itself used 100 GB+). Exported to MLFRAME_PIPELINE_CACHE_RAM_FRACTION
    # during config setup; an explicit MLFRAME_PIPELINE_CACHE_BYTES_LIMIT /
    # MLFRAME_PIPELINE_CACHE_RAM_FRACTION env still wins.
    pipeline_cache_ram_budget_fraction: float = 0.15

    # Drop per-group AGGREGATE features from the MLP's view of
    # X. Pattern matches columns like
    # ``group_<feature>_mean`` / ``group_<feature>_std`` / ``group_*_(mean|std|min|max)``:
    # these encode the train-only group mean of some other feature and
    # are CONSTANT within a group. On an unseen-group test row the
    # value is necessarily extrapolated (the test group never appears
    # in the train aggregate) and the MLP, which composes to a near-
    # affine map on whitened inputs, picks up the resulting train-vs-
    # test direction as the dominant signal -> catastrophic rank
    # inversion on unseen groups.
    # TREE models still see these features (they handle the OOD
    # categorical signal via leaves, not via affine slope). Only the
    # MLP fit-path drops them. Default False (opt-in): enable when the
    # calling project ships per-group aggregate columns matching the
    # pattern; benign no-op otherwise.
    mlp_drop_per_group_constants: bool = False
    # Regex pattern. Match is case-INSENSITIVE on column names. Default
    # captures a generic ``group_*_<reducer>`` naming; tweak for the
    # calling project's aggregate convention (e.g. ``well_.*_(mean|std)``
    # or ``rig_.*_(mean|std)``).
    mlp_drop_per_group_constants_pattern: str = r"^group_.*_(mean|std|min|max)$"

    # L2 weight-decay auto-bump for MLP on extreme-AR + group-aware
    # regimes (Fix 3, 2026-05-26). When the trigger fires
    # (lag1_corr_per_group >= mlp_extreme_ar_threshold AND the active
    # split sets ``prefer_group_aware=True``), multiply the MLP
    # optimizer's ``weight_decay`` by this factor. AdamW is forced ON
    # (Adam ignores weight_decay) and weight_decay defaults are bumped
    # from 0.0 -> base * factor. Heavier L2 bounds the effective slope
    # of the MLP's affine composition, capping extrapolation magnitude
    # on unseen-group test rows.
    mlp_extreme_ar_weight_decay_factor: float = 100.0
    # Base weight_decay when bumping. The default (1e-4) * factor=100
    # produces 1e-2 -- the upper end of "moderate" L2 for tabular
    # MLPs. Override for very high-noise regimes.
    mlp_extreme_ar_weight_decay_base: float = 1e-4


class MultilabelDispatchConfig(BaseConfig):
    """Configuration for multilabel-classification dispatch.

    Bundles every multilabel-only knob so per-strategy code only sees one
    parameter (instead of an exploding ``_maybe_wrap_multilabel(...)``
    signature) and so adding a new strategy choice (e.g. ``stacking``)
    doesn't require touching every dispatch site.

    Only consulted when ``target_type == MULTILABEL_CLASSIFICATION``.

    Strategy choices
    ----------------
    auto      : let the strategy pick -- CatBoost uses native MultiLogloss,
                everyone else uses ``MultiOutputClassifier(estimator)`` (OvR)
    wrapper   : force ``MultiOutputClassifier(estimator)`` even on CB
                (degrades CB native to OvR -- useful for A/B vs native)
    chain     : ``_ChainEnsemble`` of ``n_chains`` random-ordered
                ``ClassifierChain(estimator, cv=cv)`` instances; averages
                ``predict_proba`` outputs. Empirically +2-5% Jaccard on
                correlated labels (sklearn ``plot_classifier_chain_yeast``).
    native    : assert strategy supports native multilabel; raise if not.
                For users who explicitly want CB MultiLogloss and want to
                fail loud if mis-configured.
    """

    strategy: str = "auto"  # Literal["auto","wrapper","chain","native"]
    # n_chains>=1: 0 builds an empty _ChainEnsemble that averages nothing.
    n_chains: int = Field(default=3, ge=1)
    chain_order_strategy: str = "random"  # Literal["random","by_frequency","user"]
    chain_order_user: Optional[List[List[int]]] = None  # one ordering per chain
    chain_seeds: Optional[List[int]] = None
    # None = no cross-val of chain features; when set sklearn needs cv>=2.
    cv: Optional[int] = Field(default=5, ge=2)  # ClassifierChain.cv
    per_label_thresholds: Optional[List[float]] = None  # decision-rule thresholds
    wrapper_n_jobs: Union[int, str] = "auto"  # MultiOutputClassifier n_jobs
    allow_uncalibrated_multi: bool = False  # downgrade post-hoc calib skip from raise to warn
    # 2026-04-24 Session-2: opt-in for native XGB multilabel (multi_strategy=
    # 'multi_output_tree' + objective='binary:logistic'). XGB 3.x ships this
    # as experimental -- vector-output trees share structure across labels
    # (smaller model, integrated GPU/SHAP, faster inference). Marked WIP
    # by upstream until v3.1; default False uses MultiOutputClassifier
    # wrapper. Set True to opt in (only takes effect with strategy='native'
    # or 'auto' + XGBoostStrategy with the flag set). Combined with
    # XGBoostStrategy.supports_native_multilabel which is gated on this
    # flag at runtime.
    force_native_xgb_multilabel: bool = False

    @model_validator(mode="after")
    def _check_chain_strategy_invariants(self):
        """Validate strategy choice + chain_order_user shape.

        Pre-2026-05-20 a typo ``strategy="wrappr"`` was silently accepted (no
        Literal validation on the string), and ``chain_order_strategy="user"``
        with missing chain_order_user was accepted as well -- ClassifierChain
        silently fell back to a default ordering, the operator's hand-crafted
        order was ignored with no log line.
        """
        _STRATEGY = {"auto", "wrapper", "chain", "native"}
        _ORDER = {"random", "by_frequency", "user"}
        if self.strategy not in _STRATEGY:
            raise ValueError(f"MultilabelDispatchConfig.strategy={self.strategy!r} not in {sorted(_STRATEGY)}")
        if self.chain_order_strategy not in _ORDER:
            raise ValueError(f"MultilabelDispatchConfig.chain_order_strategy={self.chain_order_strategy!r} " f"not in {sorted(_ORDER)}")
        if self.chain_order_strategy == "user":
            if self.chain_order_user is None:
                raise ValueError(
                    "MultilabelDispatchConfig: chain_order_strategy='user' but "
                    "chain_order_user is None. Either supply chain_order_user=[[...], ...] "
                    "with one ordering per chain, or pick chain_order_strategy='random' / "
                    "'by_frequency'."
                )
            if len(self.chain_order_user) != self.n_chains:
                raise ValueError(
                    f"MultilabelDispatchConfig: chain_order_user has {len(self.chain_order_user)} " f"orderings but n_chains={self.n_chains}. Sizes must match."
                )
        return self


class LearningToRankConfig(BaseConfig):
    """Configuration for ``LEARNING_TO_RANK`` target dispatch.

    Holds knobs that are LTR-only so per-strategy ranking code sees one
    parameter (mirrors ``MultilabelDispatchConfig`` for multilabel).

    Library defaults verified empirically on the installed stack
    (CatBoost 1.2.10, XGBoost 3.x, LightGBM 4.6.0):

    - **CB** ``YetiRankPairwise`` is the listwise default; alternatives
      via ``cb_loss_fn``: ``YetiRank``, ``QuerySoftMax``, ``PairLogit``,
      ``PairLogitPairwise``, ``StochasticRank:metric=NDCG``.
    - **XGB** ``rank:ndcg`` works on graded relevance. ``rank:map``
      requires binary y (``is_binary`` C++ check). The dispatcher
      auto-falls-back to ``rank:ndcg`` with WARN if y.max()>1 even when
      user pinned ``rank:map``.
    - **LGB** ``lambdarank`` (default) or ``rank_xendcg``.

    Ensemble: RRF default (TREC standard, scale-invariant — survives
    softmax/sigmoid/raw-score divergence across CB/XGB/LGB). Borda is a
    simpler scale-invariant alternative; ``score_mean`` requires the
    user to assert ``assume_comparable_scales=True``.
    """

    cb_loss_fn: str = "YetiRankPairwise"
    """CatBoost loss_function for the ranker. Listwise pairwise default."""

    xgb_objective: str = "rank:ndcg"
    """XGBoost objective. ``rank:map`` is rejected at fit-time when
    ``y.max() > 1`` -- use ``rank:ndcg`` for graded relevance."""

    lgb_objective: str = "lambdarank"
    """LightGBM objective. ``lambdarank`` is robust on both binary and
    graded labels; ``rank_xendcg`` is an alternative."""

    mlp_loss_fn: str = "ranknet"
    """MLPRanker loss. ``ranknet`` (default; pairwise BCE on score
    differences, Burges 2005) or ``listnet`` (listwise softmax
    cross-entropy, Cao 2007). Both handle binary + graded relevance."""

    eval_at: tuple = (1, 5, 10)
    """Cutoffs for NDCG@k / MAP@k metrics. Mirrors LightGBM ``eval_at``."""

    ensemble_method: str = "rrf"
    """Ensembling method for combining ranker scores. ``rrf`` (Reciprocal
    Rank Fusion, TREC default) is invariant to monotone score transforms
    -- safe for cross-library blends. ``borda`` per-query rank averaging.
    ``score_mean`` requires comparable scales (asserted via
    ``assume_comparable_scales``)."""

    ltr_ensemble_method: Literal["rrf", "borda"] = "rrf"
    """Typed rank-fusion choice for LTR ensembling: ``rrf`` (scale-invariant,
    TREC default) or ``borda`` (per-query rank averaging, simpler and also
    scale-invariant but underweights long lists). Distinct from
    ``ensemble_method`` because this field is restricted to the two
    rank-fusion strategies that survive cross-library score-scale divergence
    without external calibration; ``score_mean`` is intentionally excluded
    here (use ``ensemble_method=score_mean`` with ``assume_comparable_scales``
    if you have calibrated scores)."""

    rrf_k: int = 60
    """RRF damping constant. 60 is the TREC default. Larger ``k`` flattens
    the position weight; smaller emphasises top-1."""

    assume_comparable_scales: bool = False
    """When True, ``ensemble_method=score_mean`` is allowed without warn.
    Set this only after externally calibrating model scores onto a
    comparable scale (e.g. via Platt / isotonic per-model)."""

    autodetect_label_format: bool = True
    """When True, dispatcher inspects ``y`` at fit-time:
    ``y.max() > 1`` -> graded (force XGB to ``rank:ndcg``);
    ``y in {0,1}`` -> binary (XGB ``rank:map`` allowed). When False,
    pass user-pinned objectives through unchanged (will crash on
    mismatched format -- caller takes responsibility)."""

    feature_selection: bool = False
    """Opt-in MRMR feature selection for the LTR ranker suite. When True, an LTR-LOCAL selector
    (``ranking._ranker_fs.select_ltr_features``) is fit on the TRAIN split's (features, graded relevance) and
    every ranker trains on the selected subset. Default OFF -- it changes which features the rankers see, so it
    is opt-in. The core MRMR procedures are NOT modified; this path runs a standard ``MRMR.fit`` with the
    relevance label as the MI target (groups handled separately, see ``fs_group_aware_mi``)."""

    fs_mrmr_kwargs: Optional[dict] = None
    """Extra kwargs forwarded to the LTR feature-selection ``MRMR(...)`` constructor (e.g. ``use_simple_mode``,
    ``quantization_nbins``, ``max_runtime_mins``). ``None`` -> a light, fast default preset."""

    fs_group_aware_mi: bool = False
    """When True, LTR feature selection ranks feature relevance with a SEPARATE group-aware mutual-information
    estimator (per-query MI averaged across queries) instead of the pointwise relevance MI. This is an LTR-only
    variant that does NOT touch the core (group-naive) MRMR MI; it pre-ranks features by group-aware relevance and
    feeds the top set as a ``must_include`` shortlist to the standard MRMR pass. Default OFF (pointwise)."""


class QuantileRegressionConfig(BaseConfig):
    """Configuration for ``QUANTILE_REGRESSION`` target dispatch.

    Holds quantile-regression-specific knobs: which alphas to predict,
    crossing-fix strategy, point-estimate alpha, coverage pairs for
    interval reports.

    Library support matrix (verified 2026-05-08 against installed stack
    CB 1.2.10 / XGB 3.x / LGB 4.6 / sklearn 1.7+):

    - **CatBoost** ``loss_function="MultiQuantile:alpha=0.1,0.5,0.9"``
      single fit, returns (N, K).
    - **XGBoost >=2.0** ``objective="reg:quantileerror",
      quantile_alpha=[0.1,0.5,0.9]`` single fit, returns (N, K).
    - **LightGBM** ``objective="quantile", alpha=0.5`` -- scalar only;
      multi-quantile via K independent fits stacked
      (_QuantileMultiOutputWrapper).
    - **HGB** ``loss="quantile", quantile=0.5`` -- scalar only; same
      wrapper path.
    - **Linear** ``QuantileRegressor(quantile=0.5)`` -- scalar only;
      same wrapper path. Slow on n>100K (LP solver O(n^2)).
    - **MLP / Recurrent** K-output head + summed pinball loss; single
      fit, returns (N, K).
    """

    alphas: tuple = (0.1, 0.5, 0.9)
    """Quantile levels to predict. Must be sorted ascending and all
    strictly between 0 and 1. Default targets the 10/50/90 percentiles
    (80% prediction interval + median)."""

    crossing_fix: str = "sort"
    """Post-prediction crossing-fix strategy:
    - ``sort``: ``np.sort(preds, axis=1)`` -- cheap, idempotent, default
    - ``isotonic``: per-row IsotonicRegression(increasing=True) -- more
      accurate when crossings are frequent, slower
    - ``none``: leave predictions unchanged (caller handles crossings)
    No library natively enforces non-crossing; even CB MultiQuantile and
    XGB quantile_alpha=[...] can produce crossings on rare configurations.
    """

    point_estimate_alpha: float = 0.5
    """Which alpha to use as the point-prediction (for downstream
    consumers that need a single y_hat). Must be present in ``alphas``
    -- default 0.5 (median). Mean-of-alphas is the alternative if
    user picks an alpha not in the set; validator enforces membership.
    """

    coverage_pairs: tuple = ((0.1, 0.9),)
    """List of (alpha_low, alpha_high) pairs for interval-coverage
    reporting. Each pair must be present in ``alphas`` and lo < hi.
    Default reports the (0.1, 0.9) -> nominal-80% interval.
    """

    wrapper_n_jobs: Any = "auto"
    """Joblib n_jobs for ``_QuantileMultiOutputWrapper`` (LGB / HGB /
    Linear paths). ``"auto"`` -> ``min(K, os.cpu_count() // 2)`` to
    avoid nested-parallelism thrashing when the inner estimator has
    its own thread pool. Set to 1 to serialise."""

    @model_validator(mode="after")
    def _validate_alphas(self) -> "QuantileRegressionConfig":
        alphas = self.alphas
        if not alphas:
            raise ValueError("QuantileRegressionConfig.alphas must be non-empty.")
        if any(not (0.0 < a < 1.0) for a in alphas):
            raise ValueError(f"QuantileRegressionConfig.alphas must be in (0, 1) " f"strict; got {list(alphas)}")
        if list(alphas) != sorted(alphas):
            raise ValueError(f"QuantileRegressionConfig.alphas must be sorted ascending; " f"got {list(alphas)}")
        if len(set(alphas)) != len(alphas):
            raise ValueError(f"QuantileRegressionConfig.alphas must be unique; " f"got {list(alphas)}")
        if self.crossing_fix not in ("sort", "isotonic", "none"):
            raise ValueError(f"crossing_fix must be one of sort/isotonic/none; " f"got {self.crossing_fix!r}")
        # point_estimate_alpha membership is enforced loosely (closest match)
        # so callers don't need to update both fields in lockstep.
        if self.point_estimate_alpha not in alphas:
            # Find closest alpha (silent snap to nearest grid point).
            closest = min(alphas, key=lambda a: abs(a - self.point_estimate_alpha))
            object.__setattr__(self, "point_estimate_alpha", closest)
        for lo, hi in self.coverage_pairs:
            if lo not in alphas or hi not in alphas:
                raise ValueError(f"coverage_pair ({lo}, {hi}) not in alphas {list(alphas)}")
            if lo >= hi:
                raise ValueError(f"coverage_pair lo={lo} must be < hi={hi}")
        return self
