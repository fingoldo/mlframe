"""Grouped pydantic configs for ``RFECV`` (2026-05-28).

The ``RFECV.__init__`` accumulated ~50 flat parameters after the Wave 1-5
audit (correctness fixes, search-strategy knobs, FI semantics, robustness
hardening, literature extensions). This module groups the implementation
knobs into three focused configs following the canonical mlframe pattern
established by ``mlframe.training._feature_selection_config.FeatureSelectionConfig``:

    - SearchConfig       : MBH surrogate / acquisition / convergence / init design
    - FIConfig           : feature-importance aggregation + voting + coef rescale + CPI
    - RobustnessConfig   : input validation + leakage + edge-case knobs

The configs are ADDITIVE: ``RFECV.__init__`` continues to accept every flat
parameter for back-compat with the existing test suite + suite-level callers.
When a config object is passed, its non-default fields OVERRIDE the matching
flat kwarg. Set ``ConfigDict(extra="allow")`` per the project convention so
new knobs can land without breaking serialised configs.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator
    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore[assignment]


if _PYDANTIC_AVAILABLE:

    class _RFECVBaseConfig(BaseModel):
        """Base for the RFECV grouped configs. Pydantic v2 with ``extra='allow'``
        so a typo doesn't kill an experiment but unknown kwargs surface in logs.
        """

        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


    class SearchConfig(_RFECVBaseConfig):
        """MBH surrogate + acquisition + convergence + init-design knobs.

        Groups Wave-2 (S1-S10) parameters + a few historical search-loop knobs.
        Field defaults MUST match RFECV.__init__ flat defaults; the validator at
        the bottom asserts this invariant.
        """

        # Stopping / budget
        max_refits: Optional[int] = None
        max_runtime_mins: Optional[float] = None
        max_noimproving_iters: int = 30
        best_desired_score: Optional[float] = None
        convergence_tol: Optional[float] = None
        convergence_tol_window: int = 10

        # MBH surrogate + acquisition
        optimizer_config: Optional[Dict[str, Any]] = None
        mbh_adaptive_threshold: int = 30
        init_design_size: Union[int, str, None] = "auto"
        dichotomic_epsilon: float = Field(default=0.1, ge=0.0, le=1.0)
        submit_dummy_to_optimizer: bool = True
        optimizer_target: str = "mean"

        # Final-mile SFFS
        swap_top_k: int = 0
        swap_top_k_allow_no_es: bool = False

        @field_validator("optimizer_target")
        @classmethod
        def _ck_target(cls, v: str) -> str:
            if v not in ("mean", "final_score"):
                raise ValueError(f"optimizer_target must be 'mean' or 'final_score'; got {v!r}")
            return v


    class FIConfig(_RFECVBaseConfig):
        """Feature-importance aggregation + voting + scale-correction knobs.

        Groups Wave-1 F1-F3 + Wave-3 F4-F14 + voting-method knob.
        """

        # Voting + aggregation
        votes_aggregation_method: Any = None  # VotesAggregation enum; left untyped to avoid circular import
        use_all_fi_runs: bool = True
        use_last_fi_run_only: bool = False
        use_one_freshest_fi_run: bool = False
        use_fi_ranking: bool = False
        fi_missing_policy: str = "worst"
        fi_decay_rate: float = Field(default=0.0, ge=0.0, lt=1.0)

        # FI semantics
        importance_getter: Union[str, Callable, None] = None
        multiclass_coef_aggregation: str = "max"
        coef_scale_source: str = "train"
        cpi_max_depth: Optional[int] = None
        cpi_min_samples_leaf: int = 10
        n_repeats: int = 5  # repeats for 'permutation'/'conditional_permutation' importance (surfaced for tuning)

        # Wide-data perm-FI cost guard (2026-06-04). Permutation / conditional-permutation importance rescore the model
        # O(p * n_repeats) times PER FOLD, so on wide frames a single RFECV iteration can exceed the whole runtime budget
        # (measured: madelon p=500, n_repeats=5 -> ~208s/iter > the 180s budget -> only 2-3 iters complete -> a 3-point CV
        # curve -> one_se_min lands at the over-selection N). When ``wide_data_fi_fallback`` is True (NEW default) and the
        # search universe exceeds ``wide_data_fi_threshold`` features, RFECV falls back to the estimator's native (gain /
        # impurity) importance for the elimination ranking so the outer loop can build a REAL multi-point curve in budget;
        # ``wide_data_fi_n_repeats`` caps n_repeats just below the threshold to soften the cliff. Set the flag False to keep
        # exact permutation FI regardless of p (and a generous max_runtime_mins).
        wide_data_fi_fallback: bool = True
        wide_data_fi_threshold: int = 200
        wide_data_fi_n_repeats: int = 2

        # Loss-of-trust handling
        keep_loser_subset_fi: bool = False
        drop_nan_score_fi: bool = True
        allow_unsafe_aggregation: bool = False

        # Result-final selection
        conduct_final_voting: bool = False
        n_features_selection_rule: str = "auto"
        mean_perf_weight: float = 1.0
        std_perf_weight: float = 0.1
        feature_cost: float = 0.0
        smooth_perf: int = 0

        @field_validator("fi_missing_policy")
        @classmethod
        def _ck_policy(cls, v: str) -> str:
            if v not in ("worst", "median", "skip"):
                raise ValueError(f"fi_missing_policy must be 'worst', 'median' or 'skip'; got {v!r}")
            return v

        @field_validator("multiclass_coef_aggregation")
        @classmethod
        def _ck_coef_agg(cls, v: str) -> str:
            if v not in ("max", "sum"):
                raise ValueError(f"multiclass_coef_aggregation must be 'max' or 'sum'; got {v!r}")
            return v

        @field_validator("coef_scale_source")
        @classmethod
        def _ck_scale(cls, v: str) -> str:
            if v not in ("train", "test", "none"):
                raise ValueError(f"coef_scale_source must be 'train' / 'test' / 'none'; got {v!r}")
            return v

        @field_validator("n_features_selection_rule")
        @classmethod
        def _ck_rule(cls, v: str) -> str:
            if v not in ("auto", "argmax", "one_se_min", "one_se_max", "plateau"):
                raise ValueError(
                    f"n_features_selection_rule must be 'auto' / 'argmax' / 'one_se_min' / 'one_se_max' / 'plateau'; got {v!r}"
                )
            return v


    class RobustnessConfig(_RFECVBaseConfig):
        """Input validation + leakage + edge-case-hardening knobs.

        Groups Wave-1 E1+E3+E2 + Wave-4 E5-E15 + must_include/exclude + leakage.
        """

        # Pinned / excluded features
        must_include: Optional[List[str]] = None
        must_exclude: Optional[List[str]] = None
        must_exclude_strict: bool = True
        feature_groups: Optional[Dict[str, List[str]]] = None

        # Leakage scan
        leakage_corr_threshold: Optional[float] = 0.95
        leakage_action: str = "warn"

        # Search-loop accounting
        noimprove_counts_revisit: bool = False

        # L7 prescreen (in-tree univariate-HT)
        prescreen: Union[str, Callable, None] = None
        prescreen_top_k: Optional[int] = None
        prescreen_fdr_level: float = Field(default=0.05, gt=0.0, lt=1.0)

        @field_validator("leakage_action")
        @classmethod
        def _ck_action(cls, v: str) -> str:
            if v not in ("warn", "exclude", "raise"):
                raise ValueError(f"leakage_action must be 'warn' / 'exclude' / 'raise'; got {v!r}")
            return v


else:
    # Pydantic missing in some constrained environments (e.g. micro-Python images).
    # Provide no-op stubs so ``isinstance(..., SearchConfig)`` checks at the RFECV
    # __init__ don't crash; users on those environments stick with flat kwargs.

    class SearchConfig:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self) -> dict:
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    class FIConfig(SearchConfig):  # type: ignore[no-redef, misc]
        pass

    class RobustnessConfig(SearchConfig):  # type: ignore[no-redef, misc]
        pass


__all__ = ["SearchConfig", "FIConfig", "RobustnessConfig"]
