"""Probes for CatBoost models whose feature declarations a rebuilt Pool cannot carry.

CatBoost registers ``cat_features`` / ``text_features`` / ``embedding_features`` AT FIT TIME. Any diagnostic that
manufactures a new frame and calls ``predict``/``predict_proba`` on it makes CatBoost rebuild a ``Pool`` from that
frame, and the rebuild forwards only the categorical declaration: a text or embedding column then arrives as a plain
numeric feature. That is not a clean Python error -- it reads out of bounds inside native code and takes the WHOLE
PROCESS down with an access violation, before anything catchable is raised (observed in CI as
"worker 'gw0' crashed", with the traceback ending inside ``catboost/core.py`` ``Pool._init``).

So the check has to happen BEFORE the call, not around it. Every probe fails open: a non-CatBoost model, an unfitted
one, or an API change all read as "not the risky case", because refusing to draw a chart on a false positive is a
worse outcome than drawing it.
"""

from __future__ import annotations

import logging
from typing import Any, List

logger = logging.getLogger(__name__)


def _indices(model: Any, getter: str) -> List[int]:
    """Feature indices from one CatBoost ``get_*_feature_indices`` accessor; empty on any probe failure."""
    fn = getattr(model, getter, None)
    if not callable(fn):
        return []
    try:
        return list(fn() or [])
    except Exception as exc:
        logger.debug("%s() probe failed (%s: %s) -- treating as not the risky case", getter, type(exc).__name__, exc)
        return []


def catboost_embedding_features(model: Any) -> List[int]:
    """Indices of the model's ``embedding_features``, or empty when there are none / it is not a fitted CatBoost."""
    return _indices(model, "get_embedding_feature_indices")


def catboost_text_features(model: Any) -> List[int]:
    """Indices of the model's ``text_features``, or empty when there are none / it is not a fitted CatBoost."""
    return _indices(model, "get_text_feature_indices")


def catboost_pool_rebuild_risk(model: Any) -> str:
    """Why re-predicting this model on a MANUFACTURED frame is unsafe, or ``""`` when it is safe.

    Returns a reason string ready to put in front of a reader, naming the feature kinds involved.
    """
    emb = catboost_embedding_features(model)
    txt = catboost_text_features(model)
    if not emb and not txt:
        return ""
    kinds = []
    if emb:
        kinds.append(f"{len(emb)} embedding feature(s)")
    if txt:
        kinds.append(f"{len(txt)} text feature(s)")
    return (
        f"the model is a CatBoost fitted with {' and '.join(kinds)}. Re-predicting on a manufactured frame makes "
        "CatBoost rebuild its Pool, and the rebuild forwards only cat_features -- so those columns arrive as plain "
        "numeric features and can crash the process inside native code rather than raise. See _catboost_guards.py."
    )


__all__ = ["catboost_embedding_features", "catboost_pool_rebuild_risk", "catboost_text_features"]
