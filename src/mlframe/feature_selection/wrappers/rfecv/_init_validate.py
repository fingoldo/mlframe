"""Constructor-time parameter validation for ``RFECV`` (carved out of ``RFECV.__init__``).

Both functions receive the dict of constructor arguments built by ``get_parent_func_args()`` and raise exactly the errors the inline
checks used to raise, in the same order relative to the staticmethod checks that stay on the class.
"""
from __future__ import annotations

import logging
from typing import Any

from .._enums import VotesAggregation
from ._futility_stop import validate_futility_anchor

logger = logging.getLogger("mlframe.feature_selection.wrappers.rfecv")

_N_FEATURES_SELECTION_RULES = ("auto", "argmax", "one_se_min", "one_se_max", "one_se_min_foldstd", "one_se_max_foldstd", "plateau")


def validate_init_params_early(params: dict[str, Any]) -> None:
    """Validate futility_anchor, frac, max_refits and cv (the checks that run before the stability / feature-group staticmethod checks)."""
    futility_anchor = params["futility_anchor"]
    frac = params["frac"]
    verbose = params["verbose"]
    max_refits = params["max_refits"]
    cv = params["cv"]
    validate_futility_anchor(futility_anchor)
    if frac is not None:
        if not (frac > 0.0 and frac < 1.0):
            raise ValueError(f"frac must be between 0 and 1, got {frac}")
        if verbose:
            logger.info("Using %s fraction of the training dataset.", frac)

    # 0 = None = no refit limit (mlframe.utils.budgets); a negative count is a mistake, not a limit.
    if max_refits is not None and max_refits < 0:
        raise ValueError(f"max_refits must be >= 0 (0 or None = unlimited); got {max_refits}.")

    # cv=1 is degenerate (no train/test split possible).
    if isinstance(cv, int) and cv < 2:
        raise ValueError(f"cv must be >= 2 (or a CV splitter object); got cv={cv}. " f"k-fold CV requires at least 2 splits.")


def validate_init_params_late(params: dict[str, Any]) -> None:
    """Validate the remaining enumerated knobs, the estimators= type-family consistency and warn on multi-estimator AM/GM aggregation."""
    nan_in_X_policy = params["nan_in_X_policy"]
    leakage_action = params["leakage_action"]
    n_features_selection_rule = params["n_features_selection_rule"]
    fi_missing_policy = params["fi_missing_policy"]
    multioutput_strategy = params["multioutput_strategy"]
    importance_agg = params["importance_agg"]
    elimination_rule = params["elimination_rule"]
    optimizer_target = params["optimizer_target"]
    dichotomic_epsilon = params["dichotomic_epsilon"]
    dichotomic_step = params["dichotomic_step"]
    fi_decay_rate = params["fi_decay_rate"]
    multiclass_coef_aggregation = params["multiclass_coef_aggregation"]
    coef_scale_source = params["coef_scale_source"]
    estimators = params["estimators"]
    votes_aggregation_method = params["votes_aggregation_method"]
    verbose = params["verbose"]
    if nan_in_X_policy not in ("impute", "raise"):
        raise ValueError(f"nan_in_X_policy must be 'impute' or 'raise'; got {nan_in_X_policy!r}.")

    if leakage_action not in ("warn", "exclude", "raise"):
        raise ValueError(f"leakage_action must be 'warn', 'exclude', or 'raise'; " f"got {leakage_action!r}.")

    if n_features_selection_rule not in _N_FEATURES_SELECTION_RULES:
        raise ValueError(f"n_features_selection_rule must be one of {_N_FEATURES_SELECTION_RULES}; got {n_features_selection_rule!r}.")

    if fi_missing_policy not in ("worst", "median", "skip"):
        raise ValueError(f"fi_missing_policy must be 'worst', 'median', or 'skip'; " f"got {fi_missing_policy!r}.")

    if multioutput_strategy not in (None, "union", "intersect"):
        raise ValueError(f"multioutput_strategy must be None, 'union', or 'intersect'; got {multioutput_strategy!r}.")
    if importance_agg not in ("legacy", "dispatched"):
        raise ValueError(f"importance_agg must be 'legacy' or 'dispatched'; got {importance_agg!r}.")

    if elimination_rule not in ("importance", "stability"):
        raise ValueError(f"elimination_rule must be 'importance' or 'stability'; got {elimination_rule!r}.")

    if optimizer_target not in ("mean", "final_score"):
        raise ValueError(f"optimizer_target must be 'mean' or 'final_score'; got {optimizer_target!r}.")

    if not (0.0 <= dichotomic_epsilon <= 1.0):
        raise ValueError(f"dichotomic_epsilon must be in [0, 1]; got {dichotomic_epsilon}.")

    if dichotomic_step not in ("auto", "midpoint"):
        raise ValueError(f"dichotomic_step must be 'auto' or 'midpoint'; got {dichotomic_step!r}.")

    if not (0.0 <= fi_decay_rate < 1.0):
        raise ValueError(f"fi_decay_rate must be in [0, 1); got {fi_decay_rate}.")

    if multiclass_coef_aggregation not in ("max", "sum"):
        raise ValueError(f"multiclass_coef_aggregation must be 'max' or 'sum'; got {multiclass_coef_aggregation!r}.")

    if coef_scale_source not in ("train", "test", "none"):
        raise ValueError(f"coef_scale_source must be 'train', 'test', or 'none'; got {coef_scale_source!r}.")

    # E9: assert all entries in ``estimators=`` share the
    # same type-family (classifier vs regressor). Mixing silently picks the
    # first estimator's family for CV stratification / scoring and the other
    # estimator crashes mid-fold with cryptic errors.
    if estimators:
        from sklearn.base import is_classifier as _is_clf
        from sklearn.base import is_regressor as _is_reg
        _est_list = list(estimators)
        if len(_est_list) >= 2:
            _is_clf_flags = [_is_clf(e) for e in _est_list]
            _is_reg_flags = [_is_reg(e) for e in _est_list]
            # Allow estimators that are NEITHER (custom transformers used as importers)
            # but reject mixing detectable classifier+regressor.
            if any(_is_clf_flags) and any(_is_reg_flags):
                raise ValueError(
                    "RFECV: estimators=[...] mixes classifier and regressor "
                    "families. All entries must share type-family for CV "
                    "stratification + scoring to be consistent. Got: "
                    f"{[type(e).__name__ for e in _est_list]}"
                )

    # F6: multi-estimator + AM/GM is unsafe.
    # LR coef ~ 0.01..1, RF Gini ~ 0..0.1, CB split-gain ~ thousands - raw
    # arithmetic / geometric mean is dominated by the largest-magnitude
    # estimator (AM) or zeroed by any single zero (GM). Force rank-based
    # rules (Borda / Copeland) for multi-estimator. User can opt back in
    # via allow_unsafe_aggregation=True for benchmarks.
    if estimators and votes_aggregation_method in (VotesAggregation.AM, VotesAggregation.GM):
        if verbose:
            logger.warning(
                "RFECV: multi-estimator + votes_aggregation_method=%s mixes raw "
                "FI values from incomparable scales (LR coef ~ 0.01-1, RF Gini ~ "
                "0-0.1, CB split-gain ~ 1000s). Switching to Borda (rank-based). "
                "Pass allow_unsafe_aggregation=True to keep the original choice.",
                votes_aggregation_method,
            )
