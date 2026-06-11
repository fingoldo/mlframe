"""mlframe.feature_selection.wrappers - RFECV and supporting helpers."""

from __future__ import annotations

from ._enums import OptimumSearch, VotesAggregation
from ._helpers import (
    _detect_multithreaded,
    _MULTITHREADED_ESTIMATOR_PATTERNS,
    _pin_threads_to_one,
    _THREAD_PARAMS,
    get_feature_importances,
    get_next_features_subset,
    get_actual_features_ranking,
    knockoff_importance,
    make_gaussian_knockoffs,
    select_features_fdr,
    select_appropriate_feature_importances,
    split_into_train_test,
    store_averaged_cv_scores,
    suppress_irritating_3rdparty_warnings,
)
from ._noise_floor import select_features_noise_floor, noise_floor_plateau
from .rfecv import RFECV
from .rfecv._configs import SearchConfig, FIConfig, RobustnessConfig

__all__ = [
    "select_features_noise_floor",
    "noise_floor_plateau",
    "OptimumSearch",
    "VotesAggregation",
    "RFECV",
    "SearchConfig",
    "FIConfig",
    "RobustnessConfig",
    "split_into_train_test",
    "store_averaged_cv_scores",
    "get_feature_importances",
    "get_next_features_subset",
    "get_actual_features_ranking",
    "select_appropriate_feature_importances",
    "suppress_irritating_3rdparty_warnings",
    "make_gaussian_knockoffs",
    "knockoff_importance",
    "select_features_fdr",
]
