"""Leaf module: shared constants + ``TargetTypes`` enum + ``BaseConfig`` for
``mlframe.training.configs``.

Split out from ``configs.py`` so the sibling config modules
(``_preprocessing_configs``, ``_model_configs``,
``_training_runtime_configs``, ``_composite_target_discovery_config``,
``_reporting_configs``) can import their shared dependencies from a leaf
module instead of from ``configs.py``.

That dodges the otherwise-unavoidable ``configs -> sibling -> configs``
import-cycle that ``tests/test_meta/test_no_import_cycles.py`` flags as a
hard fail: the parent has to import each sibling at the bottom (to re-export
the moved classes), and previously each sibling had to import ``BaseConfig``
+ the constants from the parent at the top.

Everything here is bit-for-bit identical to the pre-split definitions in
``configs.py``; the parent re-exports each name so historical
``from mlframe.training.configs import TargetTypes`` (and the other moved
names) imports continue to resolve.
"""
from __future__ import annotations

import sys
from typing import ClassVar, FrozenSet

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    # Polyfill for Python 3.9 / 3.10 (StrEnum landed in 3.11). The (str, Enum)
    # MRO gives the same equality + hashability + string-coercion behaviour
    # downstream code relies on (e.g. models.get(str_key) hash-matches
    # models.get(enum_key)).
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return str(self.value)

from pydantic import BaseModel, ConfigDict, model_validator

DEFAULT_RANDOM_SEED = 42
"""Random seed for reproducibility across all operations."""

DEFAULT_TREE_ITERATIONS = 5000
"""Default number of iterations for tree-based models (CB, LGB, XGB)."""

DEFAULT_CALIBRATION_BINS = 10
"""Default number of bins for calibration reports."""

DEFAULT_FAIRNESS_MIN_POP_CAT_THRESH = 1000
"""Default minimum population per category for fairness analysis."""

DEFAULT_RFECV_MAX_RUNTIME_MINS = 180
"""Default RFECV max runtime in minutes (3 hours)."""

DEFAULT_RFECV_CV_SPLITS = 4
"""Default number of CV splits for RFECV."""

DEFAULT_RFECV_MAX_NOIMPROVING_ITERS = 15
"""Default max non-improving iterations for RFECV early stopping."""

VALID_MODEL_TYPES = {"cb", "lgb", "xgb", "hgb", "mlp", "ngb", "linear", "ridge", "lasso", "elasticnet", "huber", "ransac", "sgd"}
"""Valid model type identifiers for mlframe_models parameter."""

VALID_LINEAR_MODEL_TYPES = {"linear", "ridge", "lasso", "elasticnet", "huber", "ransac", "sgd"}
"""Valid linear model type identifiers."""

VALID_SCALER_NAMES = {"standard", "min_max", "abs_max", "robust", None}
"""Valid scaler names for Polars pipeline."""

VALID_TASK_TYPES = {"CPU", "GPU"}
"""Valid task types for tree-based models (uppercase)."""

VALID_MATMUL_PRECISIONS = {"high", "medium", "highest"}
"""Valid float32 matmul precision settings for PyTorch."""


class TargetTypes(StrEnum):
    """Enumeration for ML task types.

    Attributes
    ----------
    REGRESSION : str
        Regression task type for continuous targets.
    BINARY_CLASSIFICATION : str
        Binary classification task type for two-class targets.
    MULTICLASS_CLASSIFICATION : str
        K>2 single-label classification (exclusive labels via softmax).
        Target shape is (N,) integer in {0, ..., K-1}.
    MULTILABEL_CLASSIFICATION : str
        K>=1 independent binary outputs (per-label sigmoid).
        Target shape is (N, K) binary matrix.
    LEARNING_TO_RANK : str
        Pairwise / listwise ranking with per-row group_id. Targets are
        per-document graded relevance (graded 0..K) or binary clicks
        (0/1). Output is a per-row score; ordering within each query
        group is what matters. CB / XGB / LGB have native rankers
        (CatBoostRanker / XGBRanker / LGBMRanker); HGB / Linear are
        not supported and skipped with NotImplementedError.
    QUANTILE_REGRESSION : str
        Predict K conditional quantiles instead of a single
        conditional mean. Target shape (N,); model output shape (N, K)
        where K = len(alphas). Use cases: prediction intervals,
        risk modelling, time-series uncertainty quantification.
        CatBoost (MultiQuantile) and XGBoost (>=2.0, quantile_alpha)
        support single-fit multi-quantile natively; LGB/HGB/Linear
        fan out to K independent fits via _QuantileMultiOutputWrapper;
        MLP / Recurrent use a K-output head + summed pinball loss.
    """

    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    LEARNING_TO_RANK = "learning_to_rank"
    QUANTILE_REGRESSION = "quantile_regression"
    # F-24 (2026-05-31): K independent continuous targets sharing a trunk.
    # Target shape ``(N, K)`` float. Output shape ``(N, K)``: column k is the
    # point prediction for target k. MLP / sklearn / Ridge / RandomForest /
    # CatBoost (loss_function="MultiRMSE") have native support; LightGBM
    # needs sklearn.multioutput.MultiOutputRegressor wrap. The MLP path
    # already lands here via F-24 commit 2d300944 (PytorchLightningRegressor
    # auto-detects (N, K>=2) y); full suite-side dispatch + per-target
    # reporting + ensemble strategy is in `docs/multi_target_regression_design.md`.
    MULTI_TARGET_REGRESSION = "multi_target_regression"

    @property
    def is_classification(self) -> bool:
        """True for binary, multiclass, and multilabel; False for regression
        and ranking.

        Use this instead of `target_type == BINARY_CLASSIFICATION` so new
        classification flavours route correctly without touching every
        call site (8 sites previously hardcoded the binary equality check).
        """
        return self in (
            TargetTypes.BINARY_CLASSIFICATION,
            TargetTypes.MULTICLASS_CLASSIFICATION,
            TargetTypes.MULTILABEL_CLASSIFICATION,
        )

    @property
    def is_regression(self) -> bool:
        """True for plain REGRESSION (single continuous target). For
        MULTI_TARGET_REGRESSION (K independent continuous targets) use
        ``is_multi_target_regression`` -- both are "regression flavours"
        but require different output-shape handling at most call sites."""
        return self == TargetTypes.REGRESSION

    @property
    def is_multi_target_regression(self) -> bool:
        """True only for ``MULTI_TARGET_REGRESSION``. Output shape (N, K)
        with K independent continuous targets sharing a trunk. Branches
        that gate on ``is_regression`` must add an explicit
        ``is_multi_target_regression`` branch when MTR is in scope."""
        return self == TargetTypes.MULTI_TARGET_REGRESSION

    @property
    def is_any_regression(self) -> bool:
        """True for any regression flavour: REGRESSION, MULTI_TARGET_REGRESSION,
        QUANTILE_REGRESSION. Convenience predicate for sites that route
        per regression-vs-classification dichotomy regardless of shape."""
        return self in (
            TargetTypes.REGRESSION,
            TargetTypes.MULTI_TARGET_REGRESSION,
            TargetTypes.QUANTILE_REGRESSION,
        )

    @property
    def is_binary(self) -> bool:
        return self == TargetTypes.BINARY_CLASSIFICATION

    @property
    def is_multiclass(self) -> bool:
        return self == TargetTypes.MULTICLASS_CLASSIFICATION

    @property
    def is_multilabel(self) -> bool:
        return self == TargetTypes.MULTILABEL_CLASSIFICATION

    @property
    def is_ranking(self) -> bool:
        """True only for ``LEARNING_TO_RANK``.

        LTR is its own class -- neither classification nor regression.
        Ranking outputs are scores (not probabilities, not real-valued
        regression targets), evaluated per-query (NDCG/MAP/MRR).
        Sites that branch on classification-vs-regression must add an
        explicit LTR branch when LTR is in scope.
        """
        return self == TargetTypes.LEARNING_TO_RANK

    @property
    def is_multi_output(self) -> bool:
        """True when model output is (N, K) with K>=2 logically.

        Convenience predicate for ``[:, 1]`` slicing sites that should
        bail out / dispatch differently for multi-* targets. LTR is NOT
        multi-output (output is a single score per row). MULTI_TARGET_REGRESSION
        and QUANTILE_REGRESSION both produce (N, K) outputs and are
        included so existing multi-output gates fire on them too.
        """
        return self in (
            TargetTypes.MULTICLASS_CLASSIFICATION,
            TargetTypes.MULTILABEL_CLASSIFICATION,
            TargetTypes.MULTI_TARGET_REGRESSION,
            TargetTypes.QUANTILE_REGRESSION,
        )

    @property
    def is_quantile(self) -> bool:
        """True only for ``QUANTILE_REGRESSION``.

        Quantile-regression output is (N, K) where K = len(alphas) and
        each column is a conditional-quantile estimate. NOT classification
        (no class probabilities), NOT plain regression (no single point
        prediction); branches that gate on regression vs classification
        must add an explicit QR branch when QR is in scope.
        """
        return self == TargetTypes.QUANTILE_REGRESSION


class BaseConfig(BaseModel):
    """Base configuration class with flexible dict support.

    Uses ``extra="allow"`` so user-supplied kwargs flow through to downstream
    callees (e.g. ``hyperparams_config={"mae_weight": 1.0}`` is not declared
    on ``ModelHyperparamsConfig`` but is consumed by ``get_training_configs``
    via ``**config_params``). Downside: typos like ``iterations=100`` get
    silently absorbed. The ``_warn_on_unknown_extras`` validator below issues
    a WARNING so typos are noticed (unless a subclass sets the
    ``_known_extras`` class attribute to list the legitimate extras).
    """

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for flexibility
        arbitrary_types_allowed=True,  # Allow numpy, torch, etc.
        validate_assignment=True,
        protected_namespaces=(),  # Allow model_ prefix for field names
    )

    #: Subclasses may list extra kwargs that are legitimately consumed
    #: downstream (e.g. ``ModelHyperparamsConfig`` -> ICE metric weights
    #: ``mae_weight`` / ``std_weight`` / ...). Entries here do not emit
    #: the "unknown extra" warning. Declared on the subclass like:
    #:     _known_extras: ClassVar[FrozenSet[str]] = frozenset({"mae_weight", ...})
    _known_extras: "ClassVar[FrozenSet[str]]" = frozenset()

    @model_validator(mode="after")
    def _warn_on_unknown_extras(self) -> "BaseConfig":
        """Log a WARNING for each extra field that is not a known pass-through.

        Catches the common typo class (``iterations`` for ``iterations``,
        ``prefer_calibrated_classifer`` missing an ``i``, etc.) that
        ``extra="allow"`` otherwise swallows without feedback.
        """
        extras = self.model_extra or {}
        if not extras:
            return self
        known = type(self)._known_extras
        unknown = [k for k in extras if k not in known]
        if unknown:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "%s received unknown field(s) %s -- these are accepted (extra='allow') "
                "but NOT declared on the model. If this is a typo for a real field, "
                "the value will have no effect. Known pass-through extras: %s",
                type(self).__name__, sorted(unknown), sorted(known) or "(none declared)",
            )
        return self
