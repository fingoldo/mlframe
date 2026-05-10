"""mlframe.feature_selection.wrappers - RFECV and helpers.

This package replaces the prior monolithic ``wrappers.py`` (1936 lines).
Public API is unchanged - all callers that did
``from mlframe.feature_selection.wrappers import RFECV`` keep working.

Module layout:
    _enums.py    - OptimumSearch, VotesAggregation
    _helpers.py  - split_into_train_test, store_averaged_cv_scores,
                   get_feature_importances, get_actual_features_ranking,
                   select_appropriate_feature_importances,
                   get_next_features_subset, multi-thread detection,
                   suppress_irritating_3rdparty_warnings
    _rfecv.py    - RFECV class itself
"""
from ._enums import OptimumSearch, VotesAggregation
from ._helpers import (
    _detect_multithreaded,
    _MULTITHREADED_ESTIMATOR_PATTERNS,
    _pin_threads_to_one,
    _THREAD_PARAMS,
    get_feature_importances,
    get_next_features_subset,
    get_actual_features_ranking,
    select_appropriate_feature_importances,
    split_into_train_test,
    store_averaged_cv_scores,
    suppress_irritating_3rdparty_warnings,
)
from ._rfecv import RFECV

__all__ = [
    "OptimumSearch",
    "VotesAggregation",
    "RFECV",
    "split_into_train_test",
    "store_averaged_cv_scores",
    "get_feature_importances",
    "get_next_features_subset",
    "get_actual_features_ranking",
    "select_appropriate_feature_importances",
    "suppress_irritating_3rdparty_warnings",
]
