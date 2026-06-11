"""Pre-loop setup helpers carved out of ``RFECV.fit``.

Three pure blocks lifted verbatim so the main ``fit`` body in ``_rfecv_fit.py`` shrinks without touching the deeply-coupled while-loop / fold-closure logic:

1. ``filter_cat_features_by_dtype`` -- strip cat_features whose columns have already been numerically encoded by an upstream pipeline step (target-encoders etc.). Pure on (X, cat_features); returns the consumable subset.
2. ``resolve_effective_n_jobs`` -- decide effective n_jobs by detecting multi-threaded estimators that already use all cores natively.
3. ``resolve_default_scoring`` -- pick probabilistic_multiclass_error for classifiers / mean_squared_error for regressors when caller passed ``scoring=None``.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import make_scorer, mean_squared_error

from mlframe.metrics.core import compute_probabilistic_multiclass_error

from .._helpers import _detect_multithreaded


logger = logging.getLogger("mlframe.feature_selection.wrappers.rfecv")


def filter_cat_features_by_dtype(
    X: Any,
    cat_features: Optional[List[str]],
    verbose: int,
) -> Optional[List[str]]:
    """Strip cat_features whose columns are no longer categorical/object dtype.

    Numerical-cast cats (CatBoostEncoder turning ``cat_0`` into a float column) trip the inner CatBoost.fit with ``Invalid type for cat_feature``. LOCAL only -- callers must NOT mutate the caller-supplied list (back-to-back fits across encoded/un-encoded frames must each pick the right subset for their X).
    """
    if not (cat_features and isinstance(X, pd.DataFrame)):
        return cat_features
    try:
        _consumable_kinds = {"O", "U", "S"}
        _consumable = []
        for _c in cat_features:
            if _c not in X.columns:
                continue
            _dt = X[_c].dtype
            if str(_dt).startswith("category") or getattr(_dt, "kind", "") in _consumable_kinds:
                _consumable.append(_c)
        if len(_consumable) != len(cat_features):
            if verbose:
                logger.info(
                    "wrappers.fit: %d/%d cat_features kept after dtype check; "
                    "the rest (%s) appear numerically encoded upstream and "
                    "are skipped for the inner estimator.",
                    len(_consumable), len(cat_features),
                    [c for c in cat_features if c not in _consumable],
                )
            return _consumable
    except Exception:
        pass
    return cat_features


def resolve_effective_n_jobs(
    n_jobs_requested,
    estimator,
    force_parallel: bool,
    verbose: int,
) -> Tuple[int, bool]:
    """Decide effective n_jobs vs the estimator's native multi-threading.

    Returns ``(n_jobs_effective, _is_multithreaded)``. Multi-threaded estimators (CB/LGB/XGB/RF/...) already use all cores natively; parallelising folds on top oversubscribes and SLOWS DOWN. Auto-fallback to sequential unless ``force_parallel=True`` (then pin inner threads to 1 in the fold runner).
    """
    n_jobs_effective = int(n_jobs_requested) if n_jobs_requested else 1
    if n_jobs_effective < 0:
        # joblib semantics: -1 = all cores.
        try:
            import os as _os
            n_jobs_effective = max(1, (_os.cpu_count() or 1))
        except Exception:
            n_jobs_effective = 1
    _is_multithreaded = _detect_multithreaded(estimator)
    if n_jobs_effective > 1 and _is_multithreaded and not force_parallel:
        if verbose:
            logger.info(
                "RFECV: n_jobs=%d requested, but %s already uses native "
                "multi-threading. Auto-falling back to sequential CV folds "
                "to avoid core oversubscription. Pass ``force_parallel=True`` "
                "to override (will pin inner threads to 1).",
                n_jobs_effective, type(estimator).__name__,
            )
        n_jobs_effective = 1
    return n_jobs_effective, _is_multithreaded


def resolve_default_scoring(scoring, estimator):
    """Pick a default scorer when the caller passed ``scoring=None``.

    Classifier -> probabilistic_multiclass_error via make_scorer(response_method='predict_proba'). Regressor -> mean_squared_error (greater_is_better=False). Raises ValueError if estimator is neither.
    """
    if scoring is not None:
        return scoring
    if is_classifier(estimator):
        logger.info("Scoring omitted, using probabilistic_multiclass_error by default.")
        # response_method='predict_proba' for sklearn 1.4+ (needs_proba is deprecated).
        return make_scorer(
            score_func=compute_probabilistic_multiclass_error,
            response_method="predict_proba",
            greater_is_better=False,
        )
    if is_regressor(estimator):
        logger.info("Scoring omitted, using mean_squared_error by default.")
        return make_scorer(score_func=mean_squared_error, greater_is_better=False)
    raise ValueError(f"Appropriate scoring not known for estimator type: {estimator}")
