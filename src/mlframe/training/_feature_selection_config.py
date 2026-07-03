"""Feature-selection config (mRMR + RFECV + Boruta-SHAP) for ``mlframe.training.configs``.

Split out from ``configs.py`` so the sibling config modules that need to
reference ``FeatureSelectionConfig`` as a field type (notably ``TrainingConfig``
in ``_training_runtime_configs.py``) can import it without re-entering
``configs.py``. That closes the last ``configs <-> sibling`` import-cycle path
the project's no-cycles meta-test flagged after the monolith split.

Behaviour preserved bit-for-bit; ``configs.py`` re-exports the class so
``from mlframe.training.configs import FeatureSelectionConfig`` continues to
resolve identity-equal.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import Field, field_validator, model_validator

from ._configs_base import BaseConfig

# Keys consumed by ``registry._instantiate_boruta_shap`` (popped from the kwargs to drive the
# default-ON GroupAwareMRMR cluster-medoid pre-reduction wrap), NOT forwarded to the underlying
# BorutaShap ctor. The boruta_shap_kwargs validator allows them through so the (default-ON)
# cluster-reduce wrap is reachable. NOT allowed for rfecv_kwargs: the suite constructs RFECV
# directly (configure_training_params), bypassing registry._instantiate_rfecv, so these keys would
# be forwarded verbatim to RFECV(**kwargs) and crash with a TypeError at construction. The suite's RFECV
# cluster-reduce wrap is driven instead by the first-class ``rfecv_cluster_*`` fields below (applied in
# ``_build_pre_pipelines``), so the documented default-ON cluster-medoid behaviour now actually holds for the suite RFECV.
_REGISTRY_CLUSTER_REDUCE_KEYS = frozenset({"cluster_reduce", "cluster_corr_threshold", "cluster_min_reduction", "cluster_corr_method"})


class FeatureSelectionConfig(BaseConfig):
    """Configuration for feature selection methods.

    Controls mRMR (minimum Redundancy Maximum Relevance) and RFECV
    (Recursive Feature Elimination with Cross-Validation) feature selection.

    Parameters
    ----------
    use_mrmr_fs : bool
        Whether to use mRMR feature selection (default: False).
    mrmr_kwargs : dict, optional
        Arguments for mRMR. Expected keys: features_to_select, show_progress, redundancy_metric.
    rfecv_models : list of str, optional
        Model types for RFECV feature selection (e.g., ["cb", "lgb"]).
    rfecv_kwargs : dict, optional
        Arguments for RFECV. Expected keys: step, min_features_to_select, cv, scoring.
    """

    # Default FS is UNSUPERVISED-ONLY: only the variance==0 / nulls>99% pre-screen (pre_screen_unsupervised)
    # runs by default; no supervised filter is applied unless the operator opts into MRMR / RFECV / BorutaShap.
    # A cheap default-on supervised filter (univariate MI top-k) was benched (_benchmarks/bench_supervised_fs_default.py)
    # and REJECTED as a default: it wins on linear downstreams + wide/noisy data but HURTS noise-robust tree
    # downstreams on low-noise data (does not win on the majority across model families), so supervised FS stays opt-in.
    use_mrmr_fs: bool = False
    mrmr_kwargs: Optional[Dict[str, Any]] = None
    rfecv_models: Optional[List[str]] = None
    rfecv_kwargs: Optional[Dict[str, Any]] = None
    custom_pre_pipelines: Dict[str, Any] = Field(default_factory=dict)

    # BorutaShap (SHAP-driven Boruta wrapper) is OFF by default: it adds 10-20x runtime over MRMR / RFECV because each trial fits a shap.TreeExplainer on a doubled feature matrix (real + shadow). Enable when the orthogonal SHAP-based signal is worth the extra compute -- typically on small frames where the shap-based feature attribution disagrees with permutation / gini.
    use_boruta_shap: bool = False
    # Forwarded verbatim to ``BorutaShap.__init__``; keys validated against the constructor signature so misspelt knobs fail at config time rather than deep inside fit.
    # Operational tips for runtime control:
    #   * ``n_trials`` (default 150): drives wall-time linearly. Small frames usually converge in 30-50; consider lowering to 50 in long suites.
    #   * ``optimistic`` (default True): keeps tentative features alongside accepted. Flip to False for strict Boruta semantics (fewer features kept).
    #   * ``train_or_test`` (default "train"): SHAP attributed on training data over-fits to noise on tree models; pass "test" for an internal train-test split when n is large enough that the held-out estimate is reliable.
    boruta_shap_kwargs: Optional[Dict[str, Any]] = None

    # ShapProxiedFS (SHAP-coalition-proxy selector) is OFF by default for the same reason as BorutaShap: it fits a SHAP TreeExplainer (OOF) and then runs a subset search + honest re-validation, so it is markedly more expensive than MRMR / RFECV. Enable when the SHAP-coalition proxy's subset-search signal is worth the compute -- typically on narrow-to-medium frames where the subset interactions the additive filters miss matter. Mirrors the BorutaShap wiring: registered in the selector registry AND reachable from the suite via this flag + a ``_build_pre_pipelines`` branch.
    use_shap_proxied_fs: bool = False
    # Forwarded verbatim to ``ShapProxiedFS.__init__``; keys validated against the constructor signature so misspelt knobs fail at config time rather than deep inside fit. ShapProxiedFS clusters correlated features internally, so it is intentionally NOT wrapped in the GroupAwareMRMR cluster-medoid reduction (double-clustering) -- the cluster-reduce keys are therefore NOT whitelisted here. ``classification`` is auto-derived from the target type when unset (mirrors BorutaShap), so a regression target picks the regressor inner model.
    shap_proxied_fs_kwargs: Optional[Dict[str, Any]] = None

    # ACE (Artificial Contrasts with Ensembles, Tuv et al. 2009) is OFF by default: it fits the estimator on [X | contrasts] once per replicate (default 20) with an optional masking-removal loop, so it is markedly more expensive than a single MRMR / RFECV pass. Enable when the contrast-percentile parametric t-test signal (continuous importance margin vs a permuted-contrast null) is worth the compute -- typically on small-to-medium frames where the per-feature significance verdict matters. Mirrors the ShapProxiedFS wiring: registered in the selector registry AND reachable from the suite via this flag + a ``_build_pre_pipelines`` branch.
    use_ace_fs: bool = False
    # Forwarded verbatim to ``ACESelector.__init__``; keys validated against the constructor signature so misspelt knobs fail at config time rather than deep inside fit. ACE auto-derives classification/regression from the target dtype internally (no target_type threading), so unlike BorutaShap / ShapProxiedFS there is no ``classification`` key to auto-fill here.
    ace_kwargs: Optional[Dict[str, Any]] = None

    # When a feature-selection pipeline (MRMR / RFECV / custom) is identity-equivalent - keeps every input column and creates no new ones - training models on it duplicates the ordinary (no-pipeline) branch. Set False to still train both (eg for ensembling diversities from different random seeds). Default True skips the duplicate branch, logging a [Dedup] info.
    skip_identity_equivalent_pre_pipelines: bool = True

    # Suite-level override for the RFECV leakage check that was previously hardcoded as a constructor default. Exposed here so operators can retune the threshold without instantiating RFECV objects manually.
    # rfecv_leakage_corr_threshold: at fit entry RFECV checks |Pearson(X_i, y)| against this; columns above the threshold are routed through ``leakage_action`` ('warn'/'exclude'/'raise'). Set ``None`` to disable the check.
    rfecv_leakage_corr_threshold: Optional[float] = 0.95
    # rfecv_mbh_adaptive_threshold: when the per-fit MBH evaluation budget is <= this value, the surrogate switches from a CatBoost model (~500ms fixed overhead per fit) to a sklearn ExtraTreesRegressor (~20ms). 30 was the historical hardcoded crossover; tune up on tiny outer estimators (LR / Ridge) where CB overhead still dominates at larger budgets, tune down when the surrogate noise from a 20-tree ETR hurts selection quality.
    rfecv_mbh_adaptive_threshold: int = 30

    # Cluster-medoid pre-reduction for the suite's RFECV. DEFAULT ON: the suite builds its RFECV instances directly in configure_training_params (not via registry._instantiate_rfecv), so this is the suite-side switch that wraps each prebuilt RFECV in GroupAwareMRMR(expand=True) at _build_pre_pipelines time -- making the documented "cluster-medoid is DEFAULT-ON for the suite's RFECV" actually hold (previously the registry default was dead for the suite RFECV path). Multi-seed validation (bench_cross_selector_diverse, 3 seeds x synthetic varied-redundancy + signal-in-non-medoid risk case) gives OOS AUC delta in [-0.0058, +0.0009] (mean -0.0005), never materially hurting (>= -0.01 floor), with the min_reduction guard making it a no-op on near-uncorrelated data (bare RFECV on full X) so it only acts where genuine correlated redundancy exists. Set False for the bare RFECV. The cluster knobs below tune the wrap.
    rfecv_cluster_reduce: bool = True
    rfecv_cluster_corr_threshold: float = 0.9
    rfecv_cluster_min_reduction: float = 0.05
    # ``rfecv_cluster_corr_method`` (pearson | spearman | kendall | su). Pearson default (cheapest; tied SU on the broad bench within noise); pin "su" for known non-monotone redundancy.
    rfecv_cluster_corr_method: str = "pearson"
    # When True, FS becomes weight-aware (correctness over speed) and re-runs per weight schema: MRMR.fit and
    # RFECV.fit receive the suite's sample_weight via fit_params, so the selected features reflect the active
    # weighting (e.g. recency emphasis). When False (default), FS is computed ONCE per target and reused across
    # weight schemas (faster, FS cache stays valid across weight iterations, but selected features reflect the
    # uniform-weight assumption). Flip ON only when you are confident weight-aware FS adds business signal that
    # outweighs the cache-miss cost; the default-OFF contract is the FS-cache reuse invariant relied on by the
    # suite's per-weight-schema training loop.
    use_sample_weights_in_fs: bool = False

    # Scope of the MRMR cross-target identity cache (see mrmr.py:_MRMR_IDENTITY_FP_CACHE).
    #   "ctx"     (default, safe): cache lives on the suite's TrainingContext; sibling suites cannot
    #             poison each other's MRMR results. Tied to A-Arch-004 mitigation of P0-003.
    #   "process": cache lives at the module level for the lifetime of the Python process; CI matrices
    #             that intentionally reuse cached identity results across suites opt in here.
    # Unsupervised pre-screen filters (variance=0 / nulls>99%) applied ONCE per suite to the train
    # split BEFORE per-target FS so obviously-useless columns never enter the expensive MRMR /
    # RFECV / BorutaShap path. Train-only fit by contract; val / test see the same drop set so
    # the pre-screen never leaks distribution information from held-out data. Conservative defaults
    # only - aggressive correlation / cardinality filters would risk dropping joint-informative
    # features and are not enabled here.
    pre_screen_unsupervised: bool = True
    pre_screen_variance_threshold: float = 0.0  # drop columns where variance == this exactly
    pre_screen_null_fraction_threshold: float = 0.99  # drop columns where null_fraction > this
    mrmr_identity_cache_scope: str = "ctx"

    # USABILITY-AWARE MULTI-LIST FEATURES (2026-06-13). When True, MRMR runs its usability-aware second
    # pass (``usability_aware_lists``) after the pure-MI fit and ``transform`` materialises the UNION of
    # all three selection lists -- pure-MI (``support_``, the tree list), strict-linear
    # (``support_linear_``) and blend (``support_universal_``) -- deduped by name, with a
    # ``usability_feature_groups_`` map recording which emitted column belongs to which list. MI is
    # rank-based and blind to linear usability, so the pure-MI list can carry raw operands (c, d) without
    # the engineered interaction (c*d) a LINEAR model needs; materialising the union puts that engineered
    # feature in EVERY model's input, so a linear model simply assigns it a coefficient and reaches the
    # f/5 floor (on F2: linear test MAE ~0.096 with the pure-MI list alone -> ~0.05 with the union), while
    # a tree just ignores the columns it does not split on (and can optionally subset to ``support_`` via
    # the groups map). DEFAULT OFF: the usability pass runs a CV-MAE forward selection that costs
    # seconds-to-minutes, so it is opt-in; existing suites are byte-identical with it off. Requires the
    # selector to run on the raw frame (MRMR's default) so the recipe replay has the raw operand columns.
    mrmr_usability_aware_lists: bool = False

    # First-class FS levers (D-surface). Each is an undocumented MRMR/RFECV constructor knob promoted to a named
    # field so it is discoverable from the suite call; ALL default to the unset sentinel so a config that does not
    # set them merges NOTHING into mrmr_kwargs / rfecv_kwargs and is byte-identical to today. ``None`` (or False for
    # the boolean enable-flags) = unset. The ``_merge_fs_levers`` validator folds set levers into the kwargs dicts
    # and RAISES if the same key is also passed explicitly in mrmr_kwargs / rfecv_kwargs (silent-override guard).
    # Whether each becomes default-ON is a SEPARATE bench-gated decision (D-flip); surfacing here changes no default.
    rfecv_must_include: Optional[List[str]] = None
    rfecv_must_exclude: Optional[List[str]] = None
    rfecv_feature_groups: Optional[Dict[str, List[str]]] = None
    rfecv_n_features_selection_rule: Optional[str] = None
    rfecv_enable_stability_selection: bool = False
    rfecv_enable_permutation_importance: bool = False
    rfecv_prescreen: Optional[str] = None
    rfecv_swap_top_k: Optional[int] = None
    mrmr_mi_normalization: Optional[str] = None
    mrmr_redundancy_aggregator: Optional[str] = None
    mrmr_cpt_test: bool = False
    mrmr_uaed_auto_size: bool = False
    mrmr_pid_synergy_bonus: Optional[float] = None
    mrmr_mi_correction: Optional[str] = None
    mrmr_group_aware_mi: bool = False
    mrmr_group_mi_aggregate: Optional[str] = None
    mrmr_group_mi_min_rows: Optional[int] = None

    @field_validator("mrmr_identity_cache_scope")
    @classmethod
    def _validate_mrmr_identity_cache_scope(cls, v: str) -> str:
        if v not in {"ctx", "process"}:
            raise ValueError(f"FeatureSelectionConfig.mrmr_identity_cache_scope must be 'ctx' or 'process', got {v!r}")
        return v

    @field_validator("mrmr_kwargs")
    @classmethod
    def _validate_mrmr_kwargs(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not v:
            return v
        import inspect
        from mlframe.feature_selection.filters import MRMR

        valid_keys = set(inspect.signature(MRMR.__init__).parameters) - {"self"}
        unknown = sorted(set(v) - valid_keys)
        if unknown:
            raise ValueError(f"FeatureSelectionConfig.mrmr_kwargs: unknown key(s) {unknown}. " f"Valid keys: {sorted(valid_keys)}")
        return v

    @field_validator("rfecv_kwargs")
    @classmethod
    def _validate_rfecv_kwargs(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not v:
            return v
        import inspect
        from mlframe.feature_selection.wrappers import RFECV

        # ``cv_n_splits`` is consumed by get_training_configs to construct a CV splitter; not a direct RFECV.__init__ arg.
        # The cluster-reduce keys are NOT whitelisted for RFECV: the suite builds its RFECV instances directly in
        # configure_training_params (NOT via registry._instantiate_rfecv), so any key in rfecv_kwargs is forwarded verbatim
        # to RFECV(**rfecv_kwargs). RFECV.__init__ rejects cluster_reduce/cluster_corr_threshold/... with a TypeError at
        # construction. Accepting them here was a config-time-green / fit-time-crash trap. (BorutaShap DOES route through the
        # registry wrap, so its validator still allows them.)
        valid_keys = (set(inspect.signature(RFECV.__init__).parameters) - {"self"}) | {"cv_n_splits"}
        unknown = sorted(set(v) - valid_keys)
        if unknown:
            raise ValueError(f"FeatureSelectionConfig.rfecv_kwargs: unknown key(s) {unknown}. " f"Valid keys: {sorted(valid_keys)}")
        return v

    @field_validator("boruta_shap_kwargs")
    @classmethod
    def _validate_boruta_shap_kwargs(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not v:
            return v
        import inspect
        from mlframe.feature_selection.boruta_shap import BorutaShap

        valid_keys = (set(inspect.signature(BorutaShap.__init__).parameters) - {"self"}) | _REGISTRY_CLUSTER_REDUCE_KEYS
        unknown = sorted(set(v) - valid_keys)
        if unknown:
            raise ValueError(f"FeatureSelectionConfig.boruta_shap_kwargs: unknown key(s) {unknown}. " f"Valid keys: {sorted(valid_keys)}")
        return v

    @field_validator("shap_proxied_fs_kwargs")
    @classmethod
    def _validate_shap_proxied_fs_kwargs(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not v:
            return v
        import inspect
        from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

        valid_keys = set(inspect.signature(ShapProxiedFS.__init__).parameters) - {"self"}
        unknown = sorted(set(v) - valid_keys)
        if unknown:
            raise ValueError(f"FeatureSelectionConfig.shap_proxied_fs_kwargs: unknown key(s) {unknown}. " f"Valid keys: {sorted(valid_keys)}")
        return v

    @field_validator("ace_kwargs")
    @classmethod
    def _validate_ace_kwargs(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not v:
            return v
        import inspect
        from mlframe.feature_selection.ace import ACESelector

        valid_keys = set(inspect.signature(ACESelector.__init__).parameters) - {"self"}
        unknown = sorted(set(v) - valid_keys)
        if unknown:
            raise ValueError(f"FeatureSelectionConfig.ace_kwargs: unknown key(s) {unknown}. " f"Valid keys: {sorted(valid_keys)}")
        return v

    @field_validator("rfecv_cluster_corr_method")
    @classmethod
    def _validate_rfecv_cluster_corr_method(cls, v: str) -> str:
        if v not in {"pearson", "spearman", "kendall", "su"}:
            raise ValueError(f"FeatureSelectionConfig.rfecv_cluster_corr_method must be one of pearson/spearman/kendall/su, got {v!r}")
        return v

    @model_validator(mode="after")
    def _merge_fs_levers(self):
        """Fold the first-class FS lever fields into ``mrmr_kwargs`` / ``rfecv_kwargs`` (D-surface).

        Each set lever maps to its MRMR/RFECV constructor key; unset levers (None / False) merge nothing, so a
        config that touches no lever is byte-identical to today. RAISES if a lever's target key is ALSO present in
        the explicit kwargs dict -- the silent-override hazard (the kwarg validator accepts the key, then a naive
        setdefault would drop it). Runs before the master-flag check so the merged dict triggers that gate too.
        """
        rfecv_levers: dict = {}
        if self.rfecv_must_include is not None:
            rfecv_levers["must_include"] = self.rfecv_must_include
        if self.rfecv_must_exclude is not None:
            rfecv_levers["must_exclude"] = self.rfecv_must_exclude
        if self.rfecv_feature_groups is not None:
            rfecv_levers["feature_groups"] = self.rfecv_feature_groups
        if self.rfecv_n_features_selection_rule is not None:
            rfecv_levers["n_features_selection_rule"] = self.rfecv_n_features_selection_rule
        if self.rfecv_enable_stability_selection:
            rfecv_levers["stability_selection"] = True
        if self.rfecv_enable_permutation_importance:
            rfecv_levers["importance_getter"] = "permutation"
        if self.rfecv_prescreen is not None:
            rfecv_levers["prescreen"] = self.rfecv_prescreen
        if self.rfecv_swap_top_k is not None:
            rfecv_levers["swap_top_k"] = self.rfecv_swap_top_k

        mrmr_levers: dict = {}
        if self.mrmr_mi_normalization is not None:
            mrmr_levers["mi_normalization"] = self.mrmr_mi_normalization
        if self.mrmr_redundancy_aggregator is not None:
            mrmr_levers["redundancy_aggregator"] = self.mrmr_redundancy_aggregator
        if self.mrmr_cpt_test:
            mrmr_levers["cpt_test"] = True
        if self.mrmr_uaed_auto_size:
            mrmr_levers["uaed_auto_size"] = True
        if self.mrmr_pid_synergy_bonus is not None:
            mrmr_levers["pid_synergy_bonus"] = self.mrmr_pid_synergy_bonus
        if self.mrmr_mi_correction is not None:
            mrmr_levers["mi_correction"] = self.mrmr_mi_correction
        if self.mrmr_group_aware_mi:
            mrmr_levers["group_aware_mi"] = True
        if self.mrmr_group_mi_aggregate is not None:
            mrmr_levers["group_mi_aggregate"] = self.mrmr_group_mi_aggregate
        if self.mrmr_group_mi_min_rows is not None:
            mrmr_levers["group_mi_min_rows"] = self.mrmr_group_mi_min_rows

        for name, levers, current in (("rfecv_kwargs", rfecv_levers, self.rfecv_kwargs), ("mrmr_kwargs", mrmr_levers, self.mrmr_kwargs)):
            if not levers:
                continue
            merged = dict(current or {})
            conflicts = sorted(set(levers) & set(merged))
            if conflicts:
                raise ValueError(
                    f"FeatureSelectionConfig: key(s) {conflicts} set BOTH as a first-class lever field AND inside "
                    f"{name}. Set one or the other, not both (the first-class field would otherwise silently override)."
                )
            merged.update(levers)
            # object.__setattr__ bypasses validate_assignment so this merge does not re-enter the model
            # validators (which would re-see the now-merged key as a false conflict on the second pass).
            object.__setattr__(self, name, merged)
        return self

    @model_validator(mode="after")
    def _check_kwargs_have_matching_master_flag(self):
        """Raise when kwargs are configured but the matching master toggle is off.

        Pre-2026-05-20 the field validators above (lines 813-860) verified the kwarg
        keys exist on MRMR.__init__ / BorutaShap.__init__, then the dict was silently
        IGNORED if use_mrmr_fs / use_boruta_shap stayed False (or rfecv_models stayed
        None). Operator pattern: hyperparameter-sweep notebook copy-pastes mrmr_kwargs
        from a previous run without re-enabling use_mrmr_fs=True; the sweep reports
        "no effect" and the operator suspects the kwargs instead of the off toggle.

        Refuse the silent ignore at config-construction time so the gate is loud.
        """
        if self.mrmr_kwargs and not self.use_mrmr_fs:
            raise ValueError(
                "FeatureSelectionConfig: mrmr_kwargs supplied but use_mrmr_fs=False. "
                "The kwargs would be silently ignored. Set use_mrmr_fs=True OR drop "
                "mrmr_kwargs to make the intent explicit."
            )
        if self.rfecv_kwargs and not self.rfecv_models:
            raise ValueError(
                "FeatureSelectionConfig: rfecv_kwargs supplied but rfecv_models is None/empty. "
                "The kwargs would be silently ignored. Set rfecv_models=[...] OR drop "
                "rfecv_kwargs to make the intent explicit."
            )
        if self.boruta_shap_kwargs and not self.use_boruta_shap:
            raise ValueError(
                "FeatureSelectionConfig: boruta_shap_kwargs supplied but use_boruta_shap=False. "
                "The kwargs would be silently ignored. Set use_boruta_shap=True OR drop "
                "boruta_shap_kwargs to make the intent explicit."
            )
        if self.shap_proxied_fs_kwargs and not self.use_shap_proxied_fs:
            raise ValueError(
                "FeatureSelectionConfig: shap_proxied_fs_kwargs supplied but use_shap_proxied_fs=False. "
                "The kwargs would be silently ignored. Set use_shap_proxied_fs=True OR drop "
                "shap_proxied_fs_kwargs to make the intent explicit."
            )
        if self.ace_kwargs and not self.use_ace_fs:
            raise ValueError(
                "FeatureSelectionConfig: ace_kwargs supplied but use_ace_fs=False. "
                "The kwargs would be silently ignored. Set use_ace_fs=True OR drop "
                "ace_kwargs to make the intent explicit."
            )
        return self
