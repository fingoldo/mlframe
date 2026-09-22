"""Val-row weights for CatBoost's eval_set.

``CatBoost*.fit`` has no ``sample_weight_eval_set`` kwarg: a val weight exists only as a ``Pool`` weight. ``_setup_eval_set``
therefore hands CatBoost its per-eval-set weights under a private fit_params key, and this module moves them onto the
eval Pools right before fit. Without it a recency-weighted or class-balanced run trained on weighted rows but
early-stopped on an unweighted val metric, so the chosen iteration optimised a different objective than training.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CB_EVAL_WEIGHTS_KEY = "_mlframe_cb_eval_sample_weight"


def _is_informative(weight: Any) -> bool:
    """Whether ``weight`` would change a metric at all: uniform weights (every row equal) do not."""
    if weight is None:
        return False
    arr = np.asarray(weight, dtype=np.float64)
    return arr.size > 0 and not np.all(arr == arr.flat[0])


def apply_cb_eval_sample_weights(fit_params: dict[str, Any], model_type_name: str | None = None) -> None:
    """Pop the private weights key and apply it to ``fit_params['eval_set']`` in place.

    A Pool entry (the reused val Pool) gets ``set_weight``; since that Pool is cached across fits, one that carried
    weights from an earlier fit is reset to ones when this fit has none. A ``(X, y)`` entry with informative weights is
    rebuilt as a weighted Pool with the fit's cat/text/embedding features. Uniform or absent weights leave the eval set
    exactly as it was. A non-CatBoost ``model_type_name`` is a no-op.
    """
    if model_type_name is not None:
        from mlframe.config import CATBOOST_MODEL_TYPES

        if model_type_name not in CATBOOST_MODEL_TYPES:
            return
    weights = fit_params.pop(CB_EVAL_WEIGHTS_KEY, None)
    es = fit_params.get("eval_set")
    if es is None:
        return
    try:
        from catboost import Pool
    except ImportError:
        return
    single = not isinstance(es, list)
    entries = [es] if single else list(es)
    weights = list(weights) if weights is not None else [None] * len(entries)
    out = []
    for entry, w in zip(entries, weights + [None] * (len(entries) - len(weights))):
        informative = _is_informative(w)
        if isinstance(entry, Pool):
            if informative:
                entry.set_weight(np.asarray(w, dtype=np.float64))
                entry._mlframe_eval_weighted = True
            elif getattr(entry, "_mlframe_eval_weighted", False):
                entry.set_weight(np.ones(entry.num_row(), dtype=np.float64))
                entry._mlframe_eval_weighted = False
            out.append(entry)
            continue
        if informative and isinstance(entry, tuple) and len(entry) == 2:
            try:
                entry = Pool(
                    data=entry[0], label=entry[1], weight=np.asarray(w, dtype=np.float64),
                    cat_features=list(fit_params.get("cat_features") or []) or None,
                    text_features=list(fit_params.get("text_features") or []) or None,
                    embedding_features=list(fit_params.get("embedding_features") or []) or None,
                )
            except Exception as exc:  # the unweighted eval still works; say that ES now ignores the weights
                logger.warning("CatBoost eval Pool with val weights could not be built (%s: %s); early stopping uses an UNWEIGHTED val metric.", type(exc).__name__, exc)
        out.append(entry)
    fit_params["eval_set"] = out[0] if single else out
