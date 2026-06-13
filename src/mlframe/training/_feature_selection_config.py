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


# Keys consumed by ``registry._instantiate_rfecv`` / ``_instantiate_boruta_shap``
# (popped from the kwargs to drive the default-ON GroupAwareMRMR cluster-medoid
# pre-reduction wrap), NOT forwarded to the underlying RFECV / BorutaShap ctor. The
# kwargs validators must allow them through -- same rationale as ``cv_n_splits`` for
# RFECV -- otherwise the (default-ON) cluster-reduce wrap is unreachable via
# FeatureSelectionConfig: the validator rejects the keys before the registry pops them.
_REGISTRY_CLUSTER_REDUCE_KEYS = frozenset(
    {"cluster_reduce", "cluster_corr_threshold", "cluster_min_reduction", "cluster_corr_method"}
)


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

    # When a feature-selection pipeline (MRMR / RFECV / custom) is identity-equivalent - keeps every input column and creates no new ones - training models on it duplicates the ordinary (no-pipeline) branch. Set False to still train both (eg for ensembling diversities from different random seeds). Default True skips the duplicate branch, logging a [Dedup] info.
    skip_identity_equivalent_pre_pipelines: bool = True

    # Suite-level override for the RFECV leakage check that was previously hardcoded as a constructor default. Exposed here so operators can retune the threshold without instantiating RFECV objects manually.
    # rfecv_leakage_corr_threshold: at fit entry RFECV checks |Pearson(X_i, y)| against this; columns above the threshold are routed through ``leakage_action`` ('warn'/'exclude'/'raise'). Set ``None`` to disable the check.
    rfecv_leakage_corr_threshold: Optional[float] = 0.95
    # rfecv_mbh_adaptive_threshold: when the per-fit MBH evaluation budget is <= this value, the surrogate switches from a CatBoost model (~500ms fixed overhead per fit) to a sklearn ExtraTreesRegressor (~20ms). 30 was the historical hardcoded crossover; tune up on tiny outer estimators (LR / Ridge) where CB overhead still dominates at larger budgets, tune down when the surrogate noise from a 20-tree ETR hurts selection quality.
    rfecv_mbh_adaptive_threshold: int = 30
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

    @field_validator("mrmr_identity_cache_scope")
    @classmethod
    def _validate_mrmr_identity_cache_scope(cls, v: str) -> str:
        if v not in {"ctx", "process"}:
            raise ValueError(
                f"FeatureSelectionConfig.mrmr_identity_cache_scope must be 'ctx' or 'process', got {v!r}"
            )
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
            raise ValueError(
                f"FeatureSelectionConfig.mrmr_kwargs: unknown key(s) {unknown}. "
                f"Valid keys: {sorted(valid_keys)}"
            )
        return v

    @field_validator("rfecv_kwargs")
    @classmethod
    def _validate_rfecv_kwargs(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not v:
            return v
        import inspect
        from mlframe.feature_selection.wrappers import RFECV
        # ``cv_n_splits`` is consumed by get_training_configs to construct a CV splitter; not a direct RFECV.__init__ arg.
        valid_keys = (set(inspect.signature(RFECV.__init__).parameters) - {"self"}) | {"cv_n_splits"} | _REGISTRY_CLUSTER_REDUCE_KEYS
        unknown = sorted(set(v) - valid_keys)
        if unknown:
            raise ValueError(
                f"FeatureSelectionConfig.rfecv_kwargs: unknown key(s) {unknown}. "
                f"Valid keys: {sorted(valid_keys)}"
            )
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
            raise ValueError(
                f"FeatureSelectionConfig.boruta_shap_kwargs: unknown key(s) {unknown}. "
                f"Valid keys: {sorted(valid_keys)}"
            )
        return v

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
        return self

